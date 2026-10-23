"""Audit every source polygon JSON and publish explicit whole-frame quarantine.

CPU metadata only. Original JSON/images are unchanged. Geometry rules exactly
match prepare_aihub_road; passing this audit does not prove image/mask validity.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from prepare_aihub_road import ANNOTATION_VALIDATOR, _inventory, annotation_errors, sha256


def inspect(item):
    dataset_id, split, key, image, annotation = item
    before = annotation.stat()
    payload = annotation.read_bytes()
    after = annotation.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Annotation changed during audit: {annotation}")
    doc = json.loads(payload.decode("utf-8-sig"))
    information = doc.get("information", {}) if isinstance(doc, dict) else {}
    resolution = information.get("resolution") if isinstance(information, dict) else None
    size = tuple(resolution) if isinstance(resolution, list) and len(resolution) == 2 else (0, 0)
    errors = annotation_errors(doc, size, image.name)
    weather, clip, _ = key
    return {"dataset_id": dataset_id, "split": split, "sample_id": f"{weather}/{clip}/{image.name}",
            "image": str(image), "annotation": str(annotation),
            "annotation_sha256": hashlib.sha256(payload).hexdigest(), "errors": errors}


def audit(source_spec, output, workers=4):
    source_spec, output = Path(source_spec).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing audits; use a new ledger path")
    spec_hash = sha256(source_spec)
    spec = json.loads(source_spec.read_text(encoding="utf-8-sig"))
    if spec.get("schema_version") != 1 or not spec.get("sources"):
        raise ValueError("Source specification is incomplete")
    pairs, counts, markers = [], {}, []
    for source in sorted(spec["sources"], key=lambda s: s["dataset_id"]):
        if source["dataset_number"] not in (13, 14, 15):
            raise ValueError("Only polygon datasets 13-15 are supported")
        for split in ("train", "val"):
            root = Path(source[split + "_root"])
            root = (root if root.is_absolute() else source_spec.parent / root).resolve(strict=True)
            marker = root / "EXTRACTION_COMPLETE.json"
            proof = json.loads(marker.read_text(encoding="utf-8-sig"))
            if proof.get("zip_crc_checked") is not True or (root / "INCOMPLETE").exists():
                raise ValueError(f"Extraction is incomplete: {root}")
            inventory = _inventory(root)
            if len(inventory) * 2 != proof["files"]:
                raise ValueError(f"Source inventory changed: {root}")
            name = source["dataset_id"] + "/" + split
            if name in counts:
                raise ValueError("Duplicate dataset/split")
            counts[name] = len(inventory)
            markers.append({"path": str(marker), "sha256": sha256(marker)})
            pairs.extend((source["dataset_id"], split, *item) for item in inventory)
    excluded, errors, classes, by_source = [], Counter(), Counter(), Counter()
    inventory_digest = hashlib.sha256()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, entry in enumerate(pool.map(inspect, pairs), 1):
            inventory_digest.update(json.dumps({k:entry[k] for k in ("dataset_id", "split", "sample_id", "annotation_sha256")}, sort_keys=True).encode())
            if entry["errors"]:
                excluded.append(entry)
                by_source[entry["dataset_id"] + "/" + entry["split"]] += 1
                errors.update(e["error"] for e in entry["errors"])
                classes.update(str(e["class"]) for e in entry["errors"])
            if index % 1000 == 0 or index == len(pairs):
                print(json.dumps({"event":"annotation_audit", "completed":index, "total":len(pairs), "excluded_frames":len(excluded)}),flush=True)
    if sha256(source_spec) != spec_hash or any(sha256(m["path"]) != m["sha256"] for m in markers):
        raise ValueError("Source specification/extraction marker changed during audit")
    report = {"schema_version":1, "format":"camera_rc_aihub_annotation_quarantine_v1", "complete":True,
              "created_utc":datetime.now(timezone.utc).isoformat(), "validator":ANNOTATION_VALIDATOR,
              "policy":"exclude_entire_frame_no_label_repair", "metadata_only":True,
              "source_spec":{"path":str(source_spec),"sha256":spec_hash}, "extraction_markers":markers,
              "audited_counts":counts, "audited_json_count":len(pairs),
              "ordered_annotation_inventory_sha256":inventory_digest.hexdigest(),
              "excluded_frame_count":len(excluded), "excluded_by_source":dict(by_source),
              "error_counts":dict(errors), "error_class_counts":dict(classes), "excluded_frames":excluded,
              "limits":["Declared JSON resolution used here; actual image dimensions and mask decoding still checked during preparation.",
                        "Whole malformed frames removed from both training and evaluation with recorded identities; retained masks are unchanged.",
                        "Invalid signed polygon area includes degenerate or cancelling geometry; no geometry is repaired or inferred."],
              "training_started":False, "gpu_used":False, "source_modified":False}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x",encoding="utf-8") as f:
        json.dump(report,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-spec",required=True,type=Path)
    p.add_argument("--output",required=True,type=Path)
    p.add_argument("--workers",type=int,default=4)
    a=p.parse_args()
    if not 1 <= a.workers <= 8: p.error("Use 1..8 CPU annotation readers")
    report=audit(a.source_spec,a.output,a.workers)
    print(json.dumps({k:report[k] for k in ("audited_json_count","excluded_frame_count","excluded_by_source","error_counts","error_class_counts")},ensure_ascii=False))


if __name__=="__main__": main()
