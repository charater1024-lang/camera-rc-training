"""Read-only verification of the fixed, observed AIHub 12--15 ZIP inventory.

No extraction, downloading, credentials, model loading, or GPU work. Only the
new output report is written. ZIP CRC success does not establish that the user
selected every archive offered by AIHub; an official inventory is still needed.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import zipfile


EXPECTED_OBSERVED_COUNTS = {"12": 32, "13": 4, "14": 4, "15": 4}


def utc():
    return datetime.now(timezone.utc).isoformat()


def stat_identity(path):
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns}


def inventory(root):
    """Bound directory metadata traversal; freeze exact paths before payload I/O."""
    result, other_files = [], []
    for dataset_id, expected in EXPECTED_OBSERVED_COUNTS.items():
        matches = [p for p in root.iterdir() if p.is_dir() and p.name.startswith(dataset_id + ".")]
        if len(matches) != 1:
            raise ValueError(f"Expected one dataset {dataset_id} directory")
        queue = deque([(matches[0], 0)])
        paths, visited = [], 0
        while queue:
            folder, depth = queue.popleft()
            visited += 1
            if visited > 100 or depth > 6:
                raise ValueError("Directory inventory exceeded bounded layout")
            with os.scandir(folder) as entries:
                for entry in entries:
                    if entry.is_symlink():
                        raise ValueError("Symlink/reparse-like entry requires separate review")
                    path = Path(entry.path)
                    if entry.is_dir(follow_symlinks=False):
                        queue.append((path, depth + 1))
                    elif entry.is_file(follow_symlinks=False):
                        if path.suffix.lower() == ".zip":
                            paths.append(path)
                            if len(paths) > expected:
                                raise ValueError(f"Dataset {dataset_id} inventory changed")
                        else:
                            other_files.append(str(path.relative_to(root)))
                            if len(other_files) > 100:
                                raise ValueError("Unexpected non-archive inventory")
        if len(paths) != expected:
            raise ValueError(f"Dataset {dataset_id}: expected {expected} observed ZIPs, got {len(paths)}")
        result.extend(sorted(paths))
    return tuple(result), other_files


def verify_one(path, root):
    result = {"relative_path": str(path.relative_to(root)), "started_utc": utc(),
              "stat_before": stat_identity(path), "sha256": None,
              "crc_bad_member": None, "error": None, "crc_all_members_passed": False}
    start = time.monotonic()
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            result["compressed_bytes_at_eof_before"] = stream.tell()
            stream.seek(0)
            digest, read_bytes = hashlib.sha256(), 0
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
                read_bytes += len(block)
            result.update(sha256=digest.hexdigest(), sha256_bytes_read=read_bytes)
            stream.seek(0)
            with zipfile.ZipFile(stream) as archive:
                members = archive.infolist()
                files = [item for item in members if not item.is_dir()]
                result.update(member_count=len(members), file_count=len(files),
                              declared_expanded_bytes=sum(item.file_size for item in files),
                              declared_member_compressed_bytes=sum(item.compress_size for item in files),
                              encrypted_members=sum(bool(item.flag_bits & 1) for item in files),
                              leading_separator_members=sum(item.filename.startswith(("/", "\\")) for item in members))
                result["crc_bad_member"] = archive.testzip()
                result["crc_all_members_passed"] = result["crc_bad_member"] is None
            stream.seek(0, os.SEEK_END)
            result["compressed_bytes_at_eof_after"] = stream.tell()
    except Exception as error:
        result["error"] = type(error).__name__ + ": " + str(error)
    result["stat_after"] = stat_identity(path)
    result["stable_during_verification"] = (
        result["stat_before"] == result["stat_after"]
        and result.get("compressed_bytes_at_eof_before") == result.get("compressed_bytes_at_eof_after")
        == result.get("sha256_bytes_read") == result["stat_after"]["bytes"])
    result["verified"] = bool(result["sha256"] and result["crc_all_members_passed"]
                              and result["stable_during_verification"] and not result["error"])
    result.update(finished_utc=utc(), elapsed_seconds=round(time.monotonic() - start, 3))
    return result


def write_report(path, report):
    temporary = path.with_name(path.name + ".writing")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def verify(root, output):
    root = root.resolve(strict=True)
    if output.exists():
        raise ValueError("Choose a new report output directory")
    paths, other_files = inventory(root)
    frozen = [{"relative_path": str(path.relative_to(root)), "stat_at_enrollment": stat_identity(path)} for path in paths]
    usage = shutil.disk_usage(root)
    report = {"schema_version": 1, "status": "running", "started_utc": utc(),
              "dataset_root": str(root), "expected_inventory_confirmed": False,
              "official_manifest_available": False, "downloader_completion_confirmed": False,
              "observed_archive_count": len(paths), "fixed_enrollment": frozen,
              "non_archive_files_observed": other_files,
              "enrollment_sha256": hashlib.sha256(json.dumps(frozen, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
              "disk_at_start": {"total_bytes": usage.total, "free_bytes": usage.free},
              "archives": [], "extraction_performed": False, "training_performed": False,
              "models_validated_for_control": False,
              "limits": ["Only the fixed 44 observed ZIPs are verified. Official expected archive inventory and downloader completion are unconfirmed.",
                         "CRC verifies each decompressed member; SHA256 binds the complete archive bytes. Stat/EOF stability is observed, not an exclusive writer lock.",
                         "ZIP names and embedded JSON paths are untrusted; extraction must normalize paths and verify destination containment."]}
    output.mkdir(parents=True, exist_ok=False)
    target = output / "archive_verification.json"
    write_report(target, report)
    for number, path in enumerate(paths, 1):
        result = verify_one(path, root)
        result["unchanged_since_enrollment"] = result["stat_before"] == frozen[number - 1]["stat_at_enrollment"]
        result["verified"] = result["verified"] and result["unchanged_since_enrollment"]
        report["archives"].append(result)
        report["completed_archive_count"] = number
        write_report(target, report)
        print(json.dumps({"completed": number, "total": len(paths), "archive": result["relative_path"],
                          "verified": result["verified"], "elapsed_seconds": result["elapsed_seconds"]}, ensure_ascii=True), flush=True)
    try:
        final_paths, final_other = inventory(root)
        report["observed_inventory_unchanged"] = final_paths == paths and final_other == other_files
    except Exception as error:
        report["observed_inventory_unchanged"] = False
        report["final_inventory_error"] = type(error).__name__ + ": " + str(error)
    for path, item in zip(paths, report["archives"]):
        try:
            item["stable_at_end_of_all_archives"] = stat_identity(path) == item["stat_after"]
        except OSError:
            item["stable_at_end_of_all_archives"] = False
    report["all_observed_archives_verified"] = bool(report["observed_inventory_unchanged"] and all(
        item["verified"] and item["stable_at_end_of_all_archives"] for item in report["archives"]))
    report["compressed_bytes"] = sum(item.get("sha256_bytes_read", 0) for item in report["archives"])
    report["declared_expanded_bytes"] = sum(item.get("declared_expanded_bytes", 0) for item in report["archives"])
    report["status"] = "complete" if report["all_observed_archives_verified"] else "failed"
    report["finished_utc"] = utc()
    report["disk_free_bytes_at_end"] = shutil.disk_usage(root).free
    write_report(target, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for archive_verification.json")
    args = parser.parse_args()
    report = verify(args.root, args.output)
    print(json.dumps({"status": report["status"], "all_observed_archives_verified": report["all_observed_archives_verified"],
                      "expected_inventory_confirmed": False, "report": str(args.output / 'archive_verification.json')}, ensure_ascii=True), flush=True)
    raise SystemExit(0 if report["all_observed_archives_verified"] else 1)


if __name__ == "__main__":
    main()
