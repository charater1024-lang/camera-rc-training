from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import verify_aihub_preprocessing as verify


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def token(value):
    return hashlib.sha256(value.encode()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value), encoding="utf-8")
    return sha(path)


def lines(path, rows):
    Path(path).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return sha(path)


def fixture(tmp_path):
    root = tmp_path / "revision"
    source = tmp_path / "source"
    root.mkdir(); source.mkdir()
    original = []
    old = np.array([[255, 255, 0], [255, 255, 0], [1, 1, 0]], dtype=np.uint8)
    annotation = {"annotations": [{"class": "trafficDrum", "polygon": [0, 0, 1, 0, 1, 1, 0, 1]}]}
    ann_path = source / "annotation.json"
    dump(ann_path, annotation)
    for index in range(3):
        mask = source / f"mask{index}.png"
        Image.fromarray(old).save(mask)
        clip = f"16_080000_22100{index + 1}"
        original.append({"export_id": str(index), "dataset_id": "aihub_13", "source_clip": clip,
                         "group_id": clip, "origin_id": "aihub_commercial:" + clip,
                         "sample_id": f"sample{index}", "image": str(source / f"image{index}.jpg"),
                         "source_image": str(source / f"image{index}.jpg"), "image_sha256": token(f"image{index}"),
                         "source_image_sha256": token(f"image{index}"), "image_rgb_sha256": token(f"rgb{index}"),
                         "target": str(mask), "target_sha256": sha(mask), "target_pixel_counts": verify.pixel_counts(old),
                         "width": 3, "height": 3, "split": "train", "export_split": "train",
                         "source_class_counts": {"trafficDrum": 1} if index == 0 else {},
                         "ignored_class_counts": {"trafficDrum": 1} if index == 0 else {},
                         "mapping_sha256": token("old_mapping"), "target_original": str(ann_path),
                         "target_original_sha256": sha(ann_path)})
    mapping = {"nonroad1": ["trafficDrum", "warningTriangle"], "all_other_classes": 255,
               "official_reporting_mapping_changed": False}
    mapping_sha = dump(root / "mapping.json", mapping)
    current = deepcopy(original)
    for index, row in enumerate(current):
        row.update(split="dev" if index == 2 else "train", export_split="dev" if index == 2 else "train",
                   development_group_id=f"development{index}", mapping_sha256=mapping_sha, ignored_class_counts={},
                   preprocessing_provenance={"previous_target": original[index]["target"],
                                             "previous_target_sha256": original[index]["target_sha256"]})
    new_mask = root / "masks" / "aihub_13" / "0.png"
    new_mask.parent.mkdir(parents=True)
    new = old.copy(); new[:2, :2] = 1
    Image.fromarray(new).save(new_mask)
    current[0].update(target=str(new_mask), target_sha256=sha(new_mask), target_pixel_counts=verify.pixel_counts(new))
    changes = [{"dataset_id": "aihub_13", "sample_id": "sample0", "image": original[0]["image"],
                "source_annotation": str(ann_path), "source_annotation_sha256": sha(ann_path),
                "old_target": original[0]["target"], "old_sha256": original[0]["target_sha256"],
                "new_target": str(new_mask), "new_sha256": sha(new_mask), "ignore_to_nonroad_pixels": 4,
                "polygons": [{"polygon_index": 0, **annotation["annotations"][0]}],
                "new_split": "train", "development_group_id": "development0"}]
    reporting = [{"dataset_id": "bdd100k_drivable", "group_id": "report_clip", "origin_id": "bdd100k:report_clip",
                  "image": str(source / "official.jpg"), "image_sha256": token("official_bytes"),
                  "image_rgb_sha256": token("official_rgb")}]
    record = {"row": reporting[0], "image": {"path": reporting[0]["image"], "sha256": reporting[0]["image_sha256"]},
              "image_rgb_sha256": reporting[0]["image_rgb_sha256"]}
    public = [{"row": {}, "image": {"path": str(source / f"public{i}.jpg"), "sha256": token(f"public_byte{i}")},
               "image_rgb_sha256": token(f"public_rgb{i}")} for i in range(40)]
    contract = {"complete": True, "sources": [{"images": 1, "records": [record]}],
                "public_tracks": {"obstacle": {"expected_public_gt_images": 30, "records": public[:30]},
                                  "anomaly": {"expected_public_gt_images": 10, "records": public[30:]}}}
    inputs = {}
    for name, filename, contents, is_manifest in (
            ("train_manifest", "old_train.jsonl", original, True),
            ("reporting_manifest", "old_val.jsonl", reporting, True),
            ("official_contract", "contract.json", contract, False),
            ("quarantine", "quarantine.json", {"excluded_frames": [{"dataset_id": "aihub_13", "sample_id": "excluded"}]}, False),
            ("old_adapter", "old_adapter.json", {"pinned": True}, False)):
        path = source / filename
        (lines if is_manifest else dump)(path, contents)
        inputs[name] = {"path": str(path), "sha256": sha(path)}
    export_report = {"complete": True, "diagnostic_only": False,
                     "manifest_outputs": {"train": {"sha256": inputs["train_manifest"]["sha256"]},
                                          "val": {"sha256": inputs["reporting_manifest"]["sha256"]}}}
    export_report_path = source / "export_report.json"
    dump(export_report_path, export_report)
    inputs["export_report"] = {"path": str(export_report_path), "sha256": sha(export_report_path)}
    export_marker_path = source / ".complete"
    dump(export_marker_path, {"report_sha256": sha(export_report_path)})
    inputs["export_marker"] = {"path": str(export_marker_path), "sha256": sha(export_marker_path)}
    policy = {"inputs": inputs, "expected_training_pool_count": 3}
    split_report = {"counts": {"input": 3, "train": 2, "dev": 1},
                    "selected_groups": [{"group_id": "development2", "rows": 1, "by_dataset": {"aihub_13": 1}}]}
    report = {"complete": True, "format": "aihub_training_preprocessing_revision_v1", "inputs": inputs,
              "pool_count": 3, "changed_frames": 1, "changed_polygons": 1, "ignore_to_nonroad_pixels": 4,
              "policy_sha256": dump(root / "policy.json", policy), "mapping_sha256": mapping_sha,
              "changes_sha256": dump(root / "changes.json", changes), "split_report_sha256": dump(root / "split_report.json", split_report),
              "manifests": {}}
    for split, rows in (("train", current[:2]), ("dev", current[2:])):
        path = root / f"{split}.jsonl"
        report["manifests"][split] = {"path": str(path), "sha256": lines(path, rows), "samples": len(rows)}
    dump(root / "report.json", report)
    dump(root / ".complete", {"report_sha256": sha(root / "report.json")})
    return root, report, current, changes


def resign(root, report, current=None, changes=None):
    if current is not None:
        for split in ("train", "dev"):
            rows = [row for row in current if row["split"] == split]
            report["manifests"][split]["sha256"] = lines(root / f"{split}.jsonl", rows)
            report["manifests"][split]["samples"] = len(rows)
    if changes is not None:
        report["changes_sha256"] = dump(root / "changes.json", changes)
    dump(root / "report.json", report)
    dump(root / ".complete", {"report_sha256": sha(root / "report.json")})


def check(root, report):
    return verify.verify_revision(root, report["inputs"]["official_contract"]["sha256"])


def test_success_reads_only_and_preserves_all_files(tmp_path):
    root, report, _, _ = fixture(tmp_path)
    # No images exist: verification must use frozen hashes without image decoding.
    before = {str(p): sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    result = check(root, report)
    after = {str(p): sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    assert before == after
    assert result["complete"] and result["counts"]["public_contract"] == 40
    assert result["counts"]["changed_pixels"] == 4
    assert result["files_written"] is False


def test_artifact_tampering_fails_hash_check(tmp_path):
    root, report, _, _ = fixture(tmp_path)
    with (root / "changes.json").open("a") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        check(root, report)


def test_missing_completion_marker_fails(tmp_path):
    root, report, _, _ = fixture(tmp_path)
    (root / ".complete").rename(root / "not_complete")
    with pytest.raises(FileNotFoundError):
        check(root, report)


def test_duplicate_partition_rejected_even_with_recomputed_hashes(tmp_path):
    root, report, current, _ = fixture(tmp_path)
    current[2]["export_id"] = current[0]["export_id"]
    resign(root, report, current)
    with pytest.raises(ValueError, match="Duplicate output export_id"):
        check(root, report)


def test_development_group_overlap_rejected(tmp_path):
    root, report, current, _ = fixture(tmp_path)
    current[2]["development_group_id"] = current[0]["development_group_id"]
    resign(root, report, current)
    with pytest.raises(ValueError, match="Train/dev overlap: development_group"):
        check(root, report)


def test_road_pixel_change_rejected_even_with_consistent_new_hashes(tmp_path):
    root, report, current, changes = fixture(tmp_path)
    mask = np.array(Image.open(current[0]["target"]))
    mask[0, 2] = 1
    Image.fromarray(mask).save(current[0]["target"])
    current[0]["target_sha256"] = changes[0]["new_sha256"] = sha(current[0]["target"])
    current[0]["target_pixel_counts"] = verify.pixel_counts(mask)
    resign(root, report, current, changes)
    with pytest.raises(ValueError, match="Forbidden pixel change"):
        check(root, report)


def test_ledger_count_must_match_actual_pixels(tmp_path):
    root, report, _, changes = fixture(tmp_path)
    changes[0]["ignore_to_nonroad_pixels"] = 5
    resign(root, report, changes=changes)
    with pytest.raises(ValueError, match="pixel count differs"):
        check(root, report)


def test_unaffected_mask_must_keep_original_path(tmp_path):
    root, report, current, _ = fixture(tmp_path)
    current[1]["target"] = str(root / "different.png")
    resign(root, report, current)
    with pytest.raises(ValueError, match="Unaffected mask path"):
        check(root, report)


def test_contract_anchor_cannot_be_redefined_in_report(tmp_path):
    root, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError, match="Protected contract anchor"):
        verify.verify_revision(root, token("another_contract"))


@pytest.mark.parametrize("field", ["byte", "rgb", "origin", "path"])
def test_frozen_public_and_cross_dataset_origin_overlap_rejected(tmp_path, field):
    root, report, current, _ = fixture(tmp_path)
    item = report["inputs"]["official_contract"]
    contract, _ = verify.read_json(item["path"])
    record = contract["public_tracks"]["obstacle"]["records"][0]
    if field == "byte":
        record["image"]["sha256"] = current[0]["image_sha256"]
    elif field == "rgb":
        record["image_rgb_sha256"] = current[0]["image_rgb_sha256"]
    elif field == "path":
        record["image"]["path"] = current[0]["image"]
    else:
        # Different dataset IDs must not hide a shared canonical origin.
        contract["sources"][0]["records"][0]["row"]["origin_id"] = current[0]["origin_id"]
    item["sha256"] = dump(item["path"], contract)
    policy, _ = verify.read_json(root / "policy.json")
    policy["inputs"]["official_contract"]["sha256"] = item["sha256"]
    report["policy_sha256"] = dump(root / "policy.json", policy)
    resign(root, report)
    with pytest.raises(ValueError, match="overlap"):
        check(root, report)


def test_public_contract_requires_all_40_records(tmp_path):
    root, report, _, _ = fixture(tmp_path)
    contract, _ = verify.read_json(report["inputs"]["official_contract"]["path"])
    contract["public_tracks"]["anomaly"]["records"].pop()
    contract["public_tracks"]["anomaly"]["expected_public_gt_images"] -= 1
    with pytest.raises(ValueError, match="exactly 40"):
        verify.contract_keys(contract)
