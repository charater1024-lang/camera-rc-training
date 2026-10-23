"""Read-only CPU verification of a completed AIHub preprocessing revision.

Writes one JSON result to stdout and no files. Image overlap comparisons use
hashes in the pinned export manifests and frozen evaluation contract; official
and public images are not reopened or decoded. Changed masks are byte-verified
and decoded here. No model, trainer, inference or CUDA imports are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import ntpath
from pathlib import Path, PureWindowsPath
import posixpath
import re

import numpy as np
from PIL import Image, ImageDraw

PINNED_CONTRACT_SHA256 = "a87bf6004f930d3cb807130d7ec5944426992a64f59c3875ffaaa7e0a843ae63"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
ADDED_CLASSES = {"trafficDrum", "warningTriangle"}
MUTABLE_FIELDS = {"split", "export_split", "development_group_id", "preprocessing_provenance",
                  "mapping_sha256", "ignored_class_counts", "target", "target_sha256", "target_pixel_counts"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def path_key(path):
    require(isinstance(path, str) and bool(path), "Nonempty path required")
    if PureWindowsPath(path).is_absolute():
        return ntpath.normcase(ntpath.normpath(path))
    require(Path(path).is_absolute(), "Absolute path required")
    return posixpath.normpath(path)


def digest(value):
    require(isinstance(value, str) and SHA256.fullmatch(value) is not None, "Valid SHA256 required")
    return value


def read_verified(path, expected=None):
    path = Path(path)
    before = path.stat()
    data = path.read_bytes()
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns),
            f"File changed during read: {path}")
    actual = hashlib.sha256(data).hexdigest()
    if expected is not None:
        require(actual == digest(expected), f"SHA256 mismatch: {path}")
    return data, actual


def read_json(path, expected=None):
    data, actual = read_verified(path, expected)
    return json.loads(data.decode("utf-8-sig")), actual


def read_manifest(path, expected):
    data, _ = read_verified(path, expected)
    return [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]


def indexed(rows, key, description):
    result = {}
    for row in rows:
        value = key(row)
        require(value not in result, f"Duplicate {description}: {value}")
        result[value] = row
    return result


def row_keys(rows):
    result = {key: set() for key in ("byte", "rgb", "path", "group", "origin", "development_group", "strong_group")}
    for row in rows:
        result["byte"].add(digest(row["image_sha256"]))
        result["rgb"].add(digest(row["image_rgb_sha256"]))
        if row.get("source_image_sha256"):
            result["byte"].add(digest(row["source_image_sha256"]))
        for field in ("image", "source_image"):
            if row.get(field):
                result["path"].add(path_key(row[field]))
        if row.get("group_id"):
            result["group"].add((row["dataset_id"], row["group_id"]))
        if row.get("origin_id"):
            result["origin"].add(row["origin_id"])
        if row.get("development_group_id"):
            result["development_group"].add(row["development_group_id"])
        if row.get("dataset_id") == "vkitti2_0_3":
            result["strong_group"].add(("vkitti_scene", row["source_scene"]))
        elif row.get("dataset_id", "").startswith("aihub_"):
            tokens = row["source_clip"].split("_")
            require(len(tokens) == 3, "AIHub clip needs device/time/date tokens")
            result["strong_group"].add(("aihub_device_date", tokens[0], tokens[2]))
    return result


def disjoint(left, right, description, fields=None):
    for field in fields or left.keys():
        require(not left[field] & right[field], f"{description} overlap: {field}")


def contract_keys(contract):
    require(contract.get("complete") is True, "Official contract incomplete")
    result = {key: set() for key in ("byte", "rgb", "path", "group", "origin")}
    counts = {}
    for category, items in (("official", contract["sources"]), ("public", list(contract["public_tracks"].values()))):
        count = 0
        for item in items:
            records = item["records"]
            declared = item.get("images", item.get("expected_public_gt_images"))
            require(declared == len(records), f"{category} contract record count mismatch")
            for record in records:
                result["byte"].add(digest(record["image"]["sha256"]))
                result["rgb"].add(digest(record["image_rgb_sha256"]))
                result["path"].add(path_key(record["image"]["path"]))
                row = record["row"]
                if row.get("dataset_id") and row.get("group_id"):
                    result["group"].add((row["dataset_id"], row["group_id"]))
                if row.get("origin_id"):
                    result["origin"].add(row["origin_id"])
            count += len(records)
        counts[category] = count
    require(counts["public"] == 40, "Expected exactly 40 frozen public regression records")
    return result, counts


def pixel_counts(mask):
    require(mask.ndim == 2 and mask.dtype == np.uint8, "Mask must be 2D uint8")
    labels, values = np.unique(mask, return_counts=True)
    require(set(labels.tolist()) <= {0, 1, 255}, "Unsupported mask label")
    return {str(int(k)): int(v) for k, v in zip(labels, values)}


def clean_counts(counts):
    return {str(k): v for k, v in counts.items() if v}


def decoded_mask(path, expected):
    data, _ = read_verified(path, expected)
    with Image.open(BytesIO(data)) as image:
        result = np.array(image)
    pixel_counts(result)
    return result


def verify_changed_mask(old_row, new_row, change):
    for old_field, ledger_field in (("target", "old_target"), ("target_sha256", "old_sha256"),
                                   ("target_original", "source_annotation"), ("target_original_sha256", "source_annotation_sha256")):
        require(old_row[old_field] == change[ledger_field], f"Ledger old provenance mismatch: {old_field}")
    require(new_row["target"] == change["new_target"] and new_row["target_sha256"] == change["new_sha256"],
            "Ledger new target mismatch")
    require(new_row["split"] == change["new_split"] and new_row["development_group_id"] == change["development_group_id"],
            "Ledger split/group mismatch")
    require(new_row["sample_id"] == change["sample_id"], "Ledger sample identity mismatch")
    old = decoded_mask(old_row["target"], old_row["target_sha256"])
    new = decoded_mask(new_row["target"], new_row["target_sha256"])
    require(old.shape == new.shape == (old_row["height"], old_row["width"]), "Changed mask shape mismatch")
    require(pixel_counts(old) == clean_counts(old_row["target_pixel_counts"]), "Old pixel counts mismatch")
    require(pixel_counts(new) == clean_counts(new_row["target_pixel_counts"]), "New pixel counts mismatch")
    difference = old != new
    require(np.all(old[difference] == 255) and np.all(new[difference] == 1), "Forbidden pixel change; only 255 to 1 allowed")
    require(np.array_equal(old == 0, new == 0), "Road pixels changed")
    total = int(difference.sum())
    require(total == change["ignore_to_nonroad_pixels"], "Changed pixel count differs from ledger")
    annotation, _ = read_json(change["source_annotation"], change["source_annotation_sha256"])
    expected = [{"polygon_index": i, "class": p["class"], "polygon": p["polygon"]}
                for i, p in enumerate(annotation["annotations"]) if p["class"] in ADDED_CLASSES]
    require(expected == change["polygons"], "Ledger polygons differ from original annotation")
    canvas = Image.new("L", (old.shape[1], old.shape[0]), 0)
    draw = ImageDraw.Draw(canvas)
    for polygon in expected:
        points = polygon["polygon"]
        draw.polygon(list(zip(points[::2], points[1::2])), fill=1)
    covered = np.asarray(canvas, dtype=bool)
    require(np.array_equal(difference, covered & (old == 255)), "Changes do not match explicit ignored obstacle polygons")
    require(np.all(new[covered] == 1), "Explicit corrected polygons contain non-nonroad pixels")
    return total, len(expected)


def verify_revision(root, expected_contract_sha256=PINNED_CONTRACT_SHA256):
    root = Path(root).resolve(strict=True)
    marker, _ = read_json(root / ".complete")
    report, report_sha = read_json(root / "report.json", marker["report_sha256"])
    require(report.get("complete") is True, "Preprocessing report incomplete")
    require(report.get("format") == "aihub_training_preprocessing_revision_v1", "Unsupported preprocessing format")
    require(not (root / "failure.json").exists(), "Failure marker exists")
    policy, _ = read_json(root / "policy.json", report["policy_sha256"])
    mapping, mapping_sha = read_json(root / "mapping.json", report["mapping_sha256"])
    changes, _ = read_json(root / "changes.json", report["changes_sha256"])
    split_report, _ = read_json(root / "split_report.json", report["split_report_sha256"])
    require(ADDED_CLASSES <= set(mapping["nonroad1"]), "Correction classes absent from mapping")
    require(mapping["all_other_classes"] == 255 and mapping["official_reporting_mapping_changed"] is False,
            "Unsupported mapping scope")
    for name, item in report["inputs"].items():
        require(item["path"] == policy["inputs"][name]["path"] and item["sha256"] == policy["inputs"][name]["sha256"],
                f"Policy/report input mismatch: {name}")
        read_verified(item["path"], item["sha256"])
    require(set(report["inputs"]) == set(policy["inputs"]), "Policy/report input names differ")
    inputs = report["inputs"]
    require(inputs["official_contract"]["sha256"] == digest(expected_contract_sha256), "Protected contract anchor mismatch")
    export_report, _ = read_json(inputs["export_report"]["path"], inputs["export_report"]["sha256"])
    export_marker, _ = read_json(inputs["export_marker"]["path"], inputs["export_marker"]["sha256"])
    require(export_report.get("complete") is True and export_report.get("diagnostic_only") is False, "Source export incomplete")
    require(export_marker["report_sha256"] == inputs["export_report"]["sha256"], "Source export completion hash mismatch")
    for split, name in (("train", "train_manifest"), ("val", "reporting_manifest")):
        require(export_report["manifest_outputs"][split]["sha256"] == inputs[name]["sha256"], "Source manifest hash mismatch")
    original = read_manifest(inputs["train_manifest"]["path"], inputs["train_manifest"]["sha256"])
    reporting = read_manifest(inputs["reporting_manifest"]["path"], inputs["reporting_manifest"]["sha256"])
    contract, _ = read_json(inputs["official_contract"]["path"], expected_contract_sha256)
    official_keys, protected_counts = contract_keys(contract)
    original_by_id = indexed(original, lambda row: row["export_id"], "input export_id")
    prepared = {}
    for split in ("train", "dev"):
        item = report["manifests"][split]
        require(path_key(item["path"]) == path_key(str(root / (split + ".jsonl"))), "Manifest path outside expected revision")
        prepared[split] = read_manifest(item["path"], item["sha256"])
        require(len(prepared[split]) == item["samples"] == split_report["counts"][split], "Partition count mismatch")
        for row in prepared[split]:
            require(row["split"] == row["export_split"] == split, "Manifest split field mismatch")
            require(bool(row.get("development_group_id")), "Missing development group")
    combined = prepared["train"] + prepared["dev"]
    new_by_id = indexed(combined, lambda row: row["export_id"], "output export_id")
    require(set(original_by_id) == set(new_by_id), "Input partition missing or adding export IDs")
    require(len(combined) == report["pool_count"] == policy["expected_training_pool_count"] == split_report["counts"]["input"],
            "Pool count mismatch")
    for export_id, old in original_by_id.items():
        new = new_by_id[export_id]
        require(old["split"] == old["export_split"] == "train", "Non-training input entered partition")
        require({k:v for k,v in old.items() if k not in MUTABLE_FIELDS} ==
                {k:v for k,v in new.items() if k not in MUTABLE_FIELDS}, "Immutable row provenance changed")
        provenance = new["preprocessing_provenance"]
        require(provenance["previous_target"] == old["target"] and provenance["previous_target_sha256"] == old["target_sha256"],
                "Previous-target provenance mismatch")
        if old["dataset_id"].startswith("aihub_"):
            require(new["mapping_sha256"] == mapping_sha, "AIHub mapping hash mismatch")
            expected_ignored = {k:v for k,v in old.get("ignored_class_counts", {}).items() if k not in ADDED_CLASSES}
            require(new["ignored_class_counts"] == expected_ignored, "Unexpected ignored class change")
        else:
            require(new.get("mapping_sha256") == old.get("mapping_sha256") and
                    new.get("ignored_class_counts") == old.get("ignored_class_counts"), "Replay label metadata changed")
    train_keys, dev_keys = row_keys(prepared["train"]), row_keys(prepared["dev"])
    disjoint(train_keys, dev_keys, "Train/dev")
    protected_manifest_keys = row_keys(reporting)
    for split in ("train", "dev"):
        keys = row_keys(prepared[split])
        disjoint(keys, protected_manifest_keys, f"{split}/official manifest", ("byte", "rgb", "path", "group", "origin"))
        disjoint(keys, official_keys, f"{split}/frozen official-public contract", tuple(official_keys))
    groups = {}
    for row in prepared["dev"]:
        stat = groups.setdefault(row["development_group_id"], Counter())
        stat[row["dataset_id"]] += 1
    declared_groups = indexed(split_report["selected_groups"], lambda group: group["group_id"], "selected development group")
    require(set(groups) == set(declared_groups), "Development groups differ from split report")
    for group, by_dataset in groups.items():
        require(dict(by_dataset) == declared_groups[group]["by_dataset"] and sum(by_dataset.values()) == declared_groups[group]["rows"],
                "Development group counts mismatch")
    quarantine, _ = read_json(inputs["quarantine"]["path"], inputs["quarantine"]["sha256"])
    excluded = {(r["dataset_id"], r["sample_id"]) for r in quarantine["excluded_frames"]}
    require(not excluded & {(r["dataset_id"], r.get("sample_id")) for r in combined}, "Quarantined frame reintroduced")
    by_image = indexed(combined, lambda r: (r["dataset_id"], path_key(r["image"])), "output image identity")
    ledger = indexed(changes, lambda r: (r["dataset_id"], path_key(r["image"])), "change ledger identity")
    expected_changed = {(r["dataset_id"], path_key(r["image"])) for r in original
                        if r["dataset_id"].startswith("aihub_") and ADDED_CLASSES & set(r.get("source_class_counts", {}))}
    require(set(ledger) == expected_changed, "Change ledger does not cover exactly the explicitly annotated obstacle frames")
    changed_pixels = changed_polygons = 0
    for key, row in by_image.items():
        old = original_by_id[row["export_id"]]
        if key in ledger:
            require(path_key(row["target"]) == path_key(str(root / "masks" / row["dataset_id"] / (row["export_id"] + ".png"))),
                    "Changed mask is outside expected revision path")
            pixels, polygons = verify_changed_mask(old, row, ledger[key])
            changed_pixels += pixels
            changed_polygons += polygons
        else:
            require(row["target"] == old["target"] and row["target_sha256"] == old["target_sha256"] and
                    row["target_pixel_counts"] == old["target_pixel_counts"], "Unaffected mask path/hash/statistics changed")
    require(len(ledger) == report["changed_frames"] and changed_pixels == report["ignore_to_nonroad_pixels"] and
            changed_polygons == report["changed_polygons"], "Changed totals differ from report")
    read_verified(inputs["official_contract"]["path"], expected_contract_sha256)
    return {"complete": True, "verified_utc": datetime.now(timezone.utc).isoformat(), "revision": str(root),
            "report_sha256": report_sha, "official_contract_sha256": expected_contract_sha256,
            "counts": {"input": len(original), "train": len(prepared["train"]), "dev": len(prepared["dev"]),
                       "official_contract": protected_counts["official"], "public_contract": protected_counts["public"],
                       "changed_frames": len(ledger), "changed_polygons": changed_polygons, "changed_pixels": changed_pixels},
            "checks": {"all_output_and_pinned_input_hashes_match": True, "every_input_export_id_partitioned_exactly_once": True,
                       "train_dev_group_origin_path_byte_rgb_overlap": 0, "train_dev_official_public_frozen_hash_overlap": 0,
                       "protected_contract_unchanged": True, "all_changed_masks_byte_verified_and_decoded": True,
                       "all_changed_pixels_ignore255_to_nonroad1": True, "road_pixels_unchanged": True,
                       "ledger_matches_source_polygons_and_pixel_counts": True, "unaffected_mask_paths_hashes_statistics_preserved": True,
                       "quarantine_preserved": True},
            "limits": ["Official/public and training image overlap checks use hashes recorded in pinned manifests/contract; images are not reopened or decoded here.",
                       "Unaffected masks are compared by pinned path/hash/statistics; their files are not reopened here.",
                       "Byte/RGB identity checks do not prove absence of perceptually similar frames or independent physical routes.",
                       "Development rows were already exposed to parent models; this is not a blind test or model-performance validation."],
            "files_written": False, "gpu_used": False, "inference_performed": False, "training_performed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", type=Path, required=True)
    parser.add_argument("--expected-contract-sha256", default=PINNED_CONTRACT_SHA256)
    args = parser.parse_args()
    try:
        print(json.dumps(verify_revision(args.revision, args.expected_contract_sha256), ensure_ascii=False, indent=2, allow_nan=False))
    except Exception as exc:
        print(json.dumps({"complete": False, "error": repr(exc), "files_written": False,
                          "gpu_used": False, "inference_performed": False, "training_performed": False}, ensure_ascii=False))
        raise SystemExit(1)
