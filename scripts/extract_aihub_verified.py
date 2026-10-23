"""Extract a fully enrolled AI Hub download into a fresh, isolated directory.

Archive CRC/structure alone does not prove that the user's download selection is
finished. A separate evidence record binds either completion or a user-selected
current snapshot to the verification report. A snapshot does not claim that all
downloads have finished. Source archives and failed extraction folders are preserved.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import zipfile

from download_bdd_archives import _member_path


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as target:
        json.dump(value, target, ensure_ascii=False, indent=2, allow_nan=False)
        target.write("\n")


def relative_path(value):
    if not isinstance(value, str) or ":" in value:
        raise ValueError("Expected a portable relative archive path")
    # The verifier writes host-native Windows paths. This normalization applies
    # only to the archive manifest, never to untrusted ZIP member names.
    value = value.replace("\\", "/")
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(p in {"", ".", ".."} for p in value.split("/")):
        raise ValueError("Unsafe archive path")
    return path


def member_path(info, *, allow_root_basename=False):
    original = info.orig_filename
    # Dataset 12's supplied archives contain /<basename>. This narrowly defined
    # legacy spelling is mapped inside this archive's fresh destination only.
    if original.startswith("/") and allow_root_basename and original.count("/") == 1:
        check = copy.copy(info)
        check.orig_filename = check.filename = original[1:]
    else:
        check = info
    result = _member_path(check)
    if str(result).casefold() in {"incomplete", "extraction_complete.json"}:
        raise ValueError("Archive member collides with extraction metadata")
    return result


def verified_archives(verification, completion):
    verification = Path(verification).resolve(strict=True)
    report = json.loads(verification.read_text(encoding="utf-8"))
    evidence = json.loads(Path(completion).read_text(encoding="utf-8"))
    complete_selection = (evidence.get("download_complete") is True
                          and evidence.get("evidence_kind") in {"user_report", "downloader_manifest"})
    selected_snapshot = (evidence.get("download_complete") is False
                         and evidence.get("evidence_kind") == "user_selected_snapshot"
                         and evidence.get("use_current_verified_inventory") is True
                         and evidence.get("scope") == "verified_archives_only")
    if (not (complete_selection or selected_snapshot)
            or evidence.get("expected_inventory_confirmed") is not True
            or evidence.get("verification_sha256") != sha256(verification)
            or not isinstance(evidence.get("evidence"), str) or not evidence["evidence"].strip()):
        raise ValueError("Confirmed selection/completion evidence or explicit current-snapshot selection is required")
    if (report.get("status") != "complete" or report.get("all_observed_archives_verified") is not True
            or report.get("observed_inventory_unchanged") is not True):
        raise ValueError("Archive verification did not complete")
    root = Path(report["dataset_root"]).resolve(strict=True)
    enrolled = [row["relative_path"] for row in report["fixed_enrollment"]]
    rows = report["archives"]
    if not rows or len(set(enrolled)) != len(enrolled) or sorted(enrolled) != sorted(row["relative_path"] for row in rows):
        raise ValueError("Verified archive list differs from enrolled selection")
    seen, result = set(), []
    for row in rows:
        rel = relative_path(row["relative_path"])
        if not rel.parts[0].startswith(("12.", "13.", "14.", "15.")) or rel.suffix.lower() != ".zip":
            raise ValueError("Archive is outside the enrolled AI Hub datasets")
        key = str(rel).casefold()
        if key in seen:
            raise ValueError("Case-colliding archive names")
        seen.add(key)
        path = root.joinpath(*rel.parts)
        if path.is_symlink() or path.resolve(strict=True) != path or not path.is_relative_to(root):
            raise ValueError("Source archive resolves outside its declared location")
        if any(row.get(flag) is not True for flag in ("verified", "crc_all_members_passed", "stable_during_verification", "unchanged_since_enrollment", "stable_at_end_of_all_archives")):
            raise ValueError("Archive was incomplete or changed during verification")
        stat = path.stat()
        if stat.st_size != row["stat_after"]["bytes"] or stat.st_mtime_ns != row["stat_after"]["mtime_ns"] or sha256(path) != row["sha256"]:
            raise ValueError("Source archive changed after verification")
        result.append((rel, path, row))
    return report, evidence, result


def extract(verification, completion, output, *, max_expanded_bytes=64 * 1024**3, reserve_bytes=10 * 1024**3):
    output = Path(output).absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError("Use a new extraction directory; incomplete output is preserved")
    if type(max_expanded_bytes) is not int or max_expanded_bytes <= 0 or type(reserve_bytes) is not int or reserve_bytes < 0:
        raise ValueError("Invalid extraction space budget")
    report, evidence, archives = verified_archives(verification, completion)
    total, plans = 0, []
    for rel, source, row in archives:
        with zipfile.ZipFile(source) as zipped:
            members, seen = [], {}
            for info in zipped.infolist():
                path = member_path(info, allow_root_basename=rel.parts[0].startswith("12."))
                key = str(path).casefold()
                if key in seen:
                    raise ValueError("Duplicate or case-colliding ZIP entry")
                seen[key] = info.is_dir()
                members.append((info, path))
            for key in seen:
                for parent in PurePosixPath(key).parents:
                    if str(parent) in seen and seen[str(parent)] is False:
                        raise ValueError("ZIP entry has a file as its parent")
            expanded = sum(info.file_size for info, _ in members)
            if expanded != row["declared_expanded_bytes"] or len(members) != row["member_count"]:
                raise ValueError("ZIP inventory differs from verification")
            total += expanded
            if total > max_expanded_bytes:
                raise ValueError("Expanded data exceeds the configured budget")
            plans.append((rel, source, row, members, expanded))
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if parent.resolve() != parent or parent.is_symlink():
        raise ValueError("Extraction ancestor is redirected")
    if shutil.disk_usage(parent).free < total + reserve_bytes:
        raise ValueError("Insufficient free space for extraction and reserve")
    output.mkdir(parents=True, exist_ok=False)
    (output / "INCOMPLETE").touch(exist_ok=False)
    outputs, road_sources = [], {}
    for rel, source, row, members, expanded in plans:
        destination = output.joinpath(*rel.with_suffix("").parts)
        destination.mkdir(parents=True, exist_ok=False)
        (destination / "INCOMPLETE").touch(exist_ok=False)
        with zipfile.ZipFile(source) as zipped:
            for info, path in members:
                target = destination.joinpath(*path.parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError("Extraction target escapes its archive directory")
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zipped.open(info) as src, target.open("xb") as dst:
                        shutil.copyfileobj(src, dst, 1024**2)  # Checks every CRC.
        if sha256(source) != row["sha256"]:
            raise ValueError("Source archive changed during extraction")
        marker = {"archive_sha256": row["sha256"], "source_archive": str(source), "files": sum(not info.is_dir() for info, _ in members),
                  "expanded_bytes": expanded, "zip_crc_checked": True,
                  "root_slash_basenames_normalized": sum(info.orig_filename.startswith("/") for info, _ in members)}
        save(destination / "EXTRACTION_COMPLETE.json", marker)
        (destination / "INCOMPLETE").unlink()
        outputs.append({"archive": str(rel), "root": str(destination), **marker})
        number = int(rel.parts[0].split(".", 1)[0])
        split = "train" if rel.name == "TS.zip" and "Training" in rel.parts else "val" if rel.name == "VS.zip" and "Validation" in rel.parts else None
        if number in (13, 14, 15) and split:
            item = road_sources.setdefault(number, {"dataset_number": number, "dataset_id": f"aihub_{number}"})
            if split + "_root" in item:
                raise ValueError("Duplicate source archive for a road dataset split")
            item[split + "_root"] = str(destination)
            item[split + "_extraction_report"] = str(destination / "EXTRACTION_COMPLETE.json")
        print(json.dumps({"event": "archive_extracted", "archive": str(rel), "files": marker["files"]}, ensure_ascii=False), flush=True)
    if set(road_sources) != {13, 14, 15} or any(not all(key in row for key in ("train_root", "val_root")) for row in road_sources.values()):
        raise ValueError("Road Training/Validation archive pairs are incomplete")
    save(output / "road_source_spec.json", {"schema_version": 1, "sources": list(road_sources.values())})
    result = {"schema_version": 1, "complete": True, "verification_sha256": sha256(verification), "completion_evidence": evidence,
              "archives": outputs, "expanded_bytes": total, "road_source_spec": str(output / "road_source_spec.json"),
              "dataset_12_note": "Extracted separately; sparse line/rectangle and relation/QA labels are not a dense road mask."}
    save(output / "EXTRACTION_COMPLETE.json", result)
    (output / "INCOMPLETE").unlink()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("verification", "completion", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--max-expanded-gib", type=int, default=64)
    args = parser.parse_args()
    result = extract(args.verification, args.completion, args.output, max_expanded_bytes=args.max_expanded_gib * 1024**3)
    print(json.dumps({"complete": result["complete"], "expanded_bytes": result["expanded_bytes"]}))


if __name__ == "__main__":
    main()
