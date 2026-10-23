"""Synthetic-only tests; no checkpoint loading, GPU or real data inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import diagnose_corrected_dev_roi as diagnostic


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")
    return sha(path)


def bundle(tmp_path, count=3):
    root = tmp_path / "prepared"
    root.mkdir()
    rows, changes = [], []
    for i in range(count):
        image = root / f"{i}.png"
        rgb = np.full((8, 8, 3), i, np.uint8)
        Image.fromarray(rgb).save(image)
        old = np.full((8, 8), 255, np.uint8)
        old[4:] = 0
        new = old.copy()
        new[0:2, 0:2] = 1
        old_file, new_file = root / f"old{i}.png", root / f"new{i}.png"
        Image.fromarray(old).save(old_file)
        Image.fromarray(new).save(new_file)
        source = root / f"annotation{i}.json"
        polygon = {"class": "trafficDrum", "polygon": [0, 0, 1, 0, 1, 1, 0, 1]}
        annotation_sha = write_json(source, {"information": {"resolution": [8, 8], "filename": image.name},
                                             "annotations": [polygon]})
        row = {"dataset_id": "aihub_13", "sample_id": image.name, "split": "dev", "export_split": "dev",
               "width": 8, "height": 8, "image": str(image), "image_sha256": sha(image),
               "image_rgb_sha256": hashlib.sha256(b"RGB:8:8:" + rgb.tobytes()).hexdigest(),
               "target": str(new_file), "target_sha256": sha(new_file),
               "target_original": str(source), "target_original_sha256": annotation_sha,
               "target_pixel_counts": {"0": 32, "1": 4, "255": 28}, "development_group_id": str(i),
               "preprocessing_provenance": {"previous_export_split": "train", "previous_target": str(old_file),
                                            "previous_target_sha256": sha(old_file),
                                            "development_role": "parent_exposed_selection_only"}}
        rows.append(row)
        changes.append({"dataset_id": row["dataset_id"], "sample_id": row["sample_id"], "image": str(image),
                        "old_target": str(old_file), "old_sha256": sha(old_file),
                        "new_target": str(new_file), "new_sha256": sha(new_file),
                        "source_annotation": str(source), "source_annotation_sha256": annotation_sha,
                        "ignore_to_nonroad_pixels": 4, "new_split": "dev", "development_group_id": str(i),
                        "polygons": [{"polygon_index": 0, **polygon}]})
    # These training paths deliberately do not exist and MUST NOT be read.
    changes.append({"dataset_id": "aihub_13", "sample_id": "train_only.png", "new_split": "train",
                    "ignore_to_nonroad_pixels": 0, "polygons": [], "old_target": "NO_TRAIN_READ"})
    mapping_sha = write_json(root / "mapping.json", {"nonroad1": ["trafficDrum", "warningTriangle"],
                           "road0": ["freespace"], "all_other_classes": 255, "official_reporting_mapping_changed": False})
    (root / "dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = {"complete": True, "format": "aihub_training_preprocessing_revision_v1", "changed_frames": count + 1,
              "changed_polygons": count, "ignore_to_nonroad_pixels": 4 * count,
              "changes_sha256": write_json(root / "changes.json", changes), "mapping_sha256": mapping_sha,
              "manifests": {"dev": {"path": str(root / "dev.jsonl"), "sha256": sha(root / "dev.jsonl"), "samples": count}}}
    report_sha = write_json(root / "report.json", report)
    write_json(root / ".complete", {"report_sha256": report_sha})
    accounting = {"frames_reprocessed": count + 1, "frames_with_pixel_changes": count,
                  "frames_without_pixel_changes": [{"dataset_id": "aihub_13", "sample_id": "train_only.png"}],
                  "ignore_to_nonroad_pixels": 4 * count}
    accounting_sha = write_json(tmp_path / "accounting.json", accounting)
    return {"root": root, "report": report, "rows": rows, "changes": changes, "accounting": accounting,
            "args": [root / "report.json", report_sha, tmp_path / "accounting.json", accounting_sha]}


def repin(b):
    root = b["root"]
    b["report"]["changes_sha256"] = write_json(root / "changes.json", b["changes"])
    (root / "dev.jsonl").write_text("".join(json.dumps(row) + "\n" for row in b["rows"]), encoding="utf-8")
    b["report"]["manifests"]["dev"]["sha256"] = sha(root / "dev.jsonl")
    b["args"][1] = write_json(root / "report.json", b["report"])
    write_json(root / ".complete", {"report_sha256": b["args"][1]})


def test_load_all_three_changed_dev_without_reading_training_files(tmp_path):
    b = bundle(tmp_path)
    frames, evidence = diagnostic.load_prepared(*b["args"])
    assert len(frames) == evidence["selected_frames"] == 3
    assert sum(int(frame["roi"].sum()) for frame in frames) == 12
    assert all(frame["classes"] == ["trafficDrum"] for frame in frames)
    assert not any("NO_TRAIN_READ" in item["path"] for item in evidence["files_checked"])


@pytest.mark.parametrize("filename", ["report.json", ".complete", "changes.json", "mapping.json", "dev.jsonl", "new0.png", "old0.png", "0.png", "annotation0.json"])
def test_tampered_files_rejected(tmp_path, filename):
    b = bundle(tmp_path)
    file = b["root"] / filename
    if filename == ".complete":
        write_json(file, {"report_sha256": "0" * 64})
    else:
        file.write_bytes(file.read_bytes() + b" ")
    with pytest.raises(ValueError, match="SHA256|marker"):
        diagnostic.load_prepared(*b["args"])


def test_accounting_hash_required(tmp_path):
    b = bundle(tmp_path)
    b["args"][3] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        diagnostic.load_prepared(*b["args"])


@pytest.mark.parametrize("mutation", ["count", "zeros", "pixels", "polygons", "duplicate", "split"])
def test_inconsistent_ledger_accounting_rejected(tmp_path, mutation):
    b = bundle(tmp_path)
    if mutation == "count":
        b["accounting"]["frames_with_pixel_changes"] -= 1
    elif mutation == "zeros":
        b["accounting"]["frames_without_pixel_changes"] = []
    elif mutation == "pixels":
        b["accounting"]["ignore_to_nonroad_pixels"] -= 1
    elif mutation == "polygons":
        b["report"]["changed_polygons"] += 1
    elif mutation == "duplicate":
        b["changes"].append(b["changes"][0])
    else:
        b["changes"][0]["new_split"] = "val"
    with pytest.raises(ValueError):
        diagnostic.check_accounting(b["report"], b["changes"], b["accounting"])


def test_no_manual_subset_above_three_or_empty(tmp_path):
    b = bundle(tmp_path, count=4)
    with pytest.raises(ValueError, match="at most three"):
        diagnostic.load_prepared(*b["args"])
    for row in b["changes"]:
        row["new_split"] = "train"
    with pytest.raises(ValueError, match="at most three"):
        diagnostic.check_accounting(b["report"], b["changes"], b["accounting"])


@pytest.mark.parametrize("mutation", ["notdev", "ancestry", "rgb", "mapping", "pixelcounts", "polygon", "group"])
def test_rehashed_semantic_mismatches_rejected(tmp_path, mutation):
    b = bundle(tmp_path)
    row, change = b["rows"][0], b["changes"][0]
    if mutation == "notdev":
        row["split"] = "train"
    elif mutation == "ancestry":
        row["preprocessing_provenance"]["previous_export_split"] = "val"
    elif mutation == "rgb":
        row["image_rgb_sha256"] = "0" * 64
    elif mutation == "mapping":
        b["report"]["mapping_sha256"] = write_json(b["root"] / "mapping.json", {
            "road0": ["freespace"], "nonroad1": ["trafficDrum", "warningTriangle"],
            "all_other_classes": 255, "official_reporting_mapping_changed": True})
    elif mutation == "pixelcounts":
        row["target_pixel_counts"]["1"] = 3
    elif mutation == "polygon":
        change["polygons"][0]["class"] = "warningTriangle"
    else:
        change["development_group_id"] = "other"
    repin(b)
    with pytest.raises(ValueError):
        diagnostic.load_prepared(*b["args"])


def test_corrected_roi_can_vanish_at_native_gt_resize():
    old = np.full((1080, 1920), 255, dtype=np.uint8)
    new = old.copy()
    new[0, 0] = 1
    roi = diagnostic.changed_roi(old, new, 1)
    reduced = np.asarray(Image.fromarray(roi.astype(np.uint8)).resize((320, 192), Image.Resampling.NEAREST))
    assert roi.sum() == 1 and reduced.sum() == 0


@pytest.mark.parametrize("old_value,new_value", [(0, 1), (1, 0), (255, 0), (1, 255)])
def test_road_or_other_label_change_rejected(old_value, new_value):
    old = np.array([[old_value]], np.uint8)
    new = np.array([[new_value]], np.uint8)
    with pytest.raises(ValueError, match="255 to 1"):
        diagnostic.changed_roi(old, new, 1)


def test_exactly_one_forward_per_frame_same_roi_and_road_denominators(tmp_path):
    frames, _ = diagnostic.load_prepared(*bundle(tmp_path)["args"])
    calls = []
    def predictor(rgb):
        calls.append(rgb)
        probability = np.zeros(rgb.shape[:2], np.float32)
        probability[0, 0] = np.float32(.65)  # Threshold equality is road.
        probability[4:6] = .8
        return probability
    baseline = diagnostic.evaluate_frames(frames, predictor)
    assert len(calls) == baseline["aggregate"]["forwards"] == 3
    assert baseline["aggregate"]["changed_roi_pixels"] == 12
    assert baseline["aggregate"]["changed_roi_false_road_pixels"] == 3
    assert baseline["aggregate"]["changed_roi_false_road_rate"] == .25
    assert baseline["aggregate"]["same_frames_road_recall"] == .5
    candidate = diagnostic.evaluate_frames(frames, lambda rgb: np.zeros(rgb.shape[:2], np.float32))
    comparison = diagnostic.differences(baseline["aggregate"], candidate["aggregate"])
    assert comparison["changed_roi_false_road_rate"]["delta_percentage_points"] == -25
    assert comparison["changed_roi_false_road_rate"]["relative_improvement_percent"] == 100
    assert comparison["same_frames_road_recall"]["relative_improvement_percent"] == -100


@pytest.mark.parametrize("kind", ["wrong_shape", "nan", "too_large", "negative", "float16"])
def test_invalid_probabilities_rejected(tmp_path, kind):
    frames, _ = diagnostic.load_prepared(*bundle(tmp_path, count=1)["args"])
    value = np.zeros((8, 8), np.float32)
    if kind == "wrong_shape":
        value = value[:2]
    elif kind == "float16":
        value = value.astype(np.float16)
    else:
        value[0, 0] = {"nan": np.nan, "too_large": 1.1, "negative": -.1}[kind]
    with pytest.raises(ValueError, match="float32 probabilities"):
        diagnostic.evaluate_frames(frames, lambda _: value)


def test_image_balanced_vs_micro_and_undefined_road_recall():
    frames = []
    for i, count in enumerate((1, 3)):
        roi = np.zeros((2, 2), bool)
        roi.flat[:count] = True
        frames.append({"row": {"dataset_id": "test", "sample_id": str(i)},
                       "rgb": np.full((2, 2, 3), i, np.uint8), "roi": roi, "road": np.zeros((2, 2), bool)})
    result = diagnostic.evaluate_frames(frames, lambda rgb: np.full((2, 2), 1 - rgb[0, 0, 0], np.float32))["aggregate"]
    assert result["changed_roi_false_road_rate"] == .25
    assert result["changed_roi_image_mean_false_road_rate"] == .5
    assert result["same_frames_road_recall"] is None
    identical = diagnostic.differences(result, result)
    assert identical["same_frames_road_recall"]["delta_percentage_points"] is None
    zero = deepcopy(result)
    zero["changed_roi_false_road_rate"] = 0
    assert diagnostic.differences(zero, result)["changed_roi_false_road_rate"]["relative_change_percent"] is None
    wrong = deepcopy(result)
    wrong["changed_roi_pixels"] += 1
    with pytest.raises(ValueError, match="denominator"):
        diagnostic.differences(result, wrong)


def test_public_allowlist_excludes_private_paths_and_no_gate_claim():
    safe_keys = ("protocol_id", "diagnostic_complete", "scope", "conditions", "coverage", "models",
                 "aggregate_results", "differences", "limitations", "runtime_comparison_performed",
                 "training_performed", "threshold_tuning_performed", "model_selection_gate_performed", "package_versions")
    report = {key: False for key in safe_keys}
    report["private_provenance"] = {"path": "D:/PRIVATE_DATA/image.png"}
    report["private_frame_results"] = {"sample_id": "private_sample"}
    aggregate = diagnostic.public_aggregate(report)
    assert set(aggregate) == set(safe_keys)
    assert "PRIVATE" not in json.dumps(aggregate) and "private_sample" not in json.dumps(aggregate)


def test_cli_has_no_device_threshold_frame_selection_or_gate_override():
    flags = diagnostic._parser().format_help()
    for forbidden in ("--device", "--threshold", "--frames", "--max-images", "--gate", "--batch"):
        assert forbidden not in flags
    assert "--candidate-training-report-sha256" in flags


@pytest.mark.parametrize("destination", ["public_tree", "existing"])
def test_cli_rejects_public_output_or_existing_evidence_before_data_read(tmp_path, destination):
    output = Path(diagnostic.__file__).resolve().parents[1] / "private_test" if destination == "public_tree" else tmp_path
    args = ["--prepared-report", "unread", "--prepared-report-sha256", "0" * 64,
            "--frame-accounting", "unread", "--frame-accounting-sha256", "0" * 64,
            "--baseline-training-report", "unread", "--baseline-training-report-sha256", "0" * 64,
            "--baseline-checkpoint", "unread", "--candidate-training-report", "unread",
            "--candidate-training-report-sha256", "0" * 64, "--candidate-checkpoint", "unread", "--output", str(output)]
    with pytest.raises(ValueError, match="private paths|already exists"):
        diagnostic.main(args)


def paired_exports(tmp_path):
    """Tiny real CPU exports + fake weight bytes; weights are never loaded."""
    from prepare_yolo_semantic import prepare
    from camera_rc.driving.yolo_semantic import verify_completion
    rows = []
    for index in range(3):
        image, target = tmp_path / f"rgb{index}.png", tmp_path / f"mask{index}.png"
        rgb = np.full((16, 16, 3), 30 + index, np.uint8)
        mask = np.ones((16, 16), np.uint8)
        mask[8:] = 0
        Image.fromarray(rgb).save(image)
        Image.fromarray(mask).save(target)
        rows.append({"dataset_id": "fixture", "group_id": str(index), "origin_id": str(index),
                     "development_group_id": str(index), "image": str(image), "target": str(target),
                     "label_space_id": "road_surface_v1", "image_sha256": sha(image), "target_sha256": sha(target),
                     "image_rgb_sha256": hashlib.sha256(b"RGB:16:16:" + rgb.tobytes()).hexdigest()})
    current = {"train": [{**rows[0], "split": "train", "export_split": "train"}],
               "dev": [{**rows[1], "split": "dev", "export_split": "dev"}]}
    revision = {"complete": True, "manifests": {}}
    for split, selected in current.items():
        manifest = tmp_path / f"prepared_{split}.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
        revision["manifests"][split] = {**diagnostic.identity(manifest), "samples": len(selected)}
    revision_path = tmp_path / "prepared_report.json"
    write_json(revision_path, revision)
    prepared = {"prepared_report": diagnostic.identity(revision_path),
                "dev_manifest_sha256": revision["manifests"]["dev"]["sha256"]}
    completed, reports, paths, configs = {}, {}, {}, {}
    for role in ("baseline", "candidate"):
        directory = tmp_path / role
        directory.mkdir()
        source_rows = {"train": rows[:2], "val": rows[2:]} if role == "baseline" else {"train": current["train"], "val": current["dev"]}
        for split, selected in source_rows.items():
            (directory / f"{split}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in selected), encoding="utf-8")
        config_path = directory / "config.json"
        write_json(config_path, {"task": "segmentation", "model": {"num_classes": 2}, "training": {"amp": False},
                   "data": {"train_manifest": "train.jsonl", "val_manifest": "val.jsonl", "label_space_id": "road_surface_v1", "image_size": [192, 320]}})
        prepare(config_path, directory / "export")
        best, last = directory / "best.pt", directory / "last.pt"
        best.write_bytes(f"fake {role} best".encode())
        last.write_bytes(f"fake {role} last".encode())
        count = len(source_rows["train"])
        report = {"schema_version": 1, "model_family": "yolo26x-sem", "task": "semantic",
                  "class_names": ["road_surface", "not_road_surface"], "road_class_id": 0,
                  "training_complete": True, "diagnostic": False, "source_config": diagnostic.identity(config_path),
                  "actual_args": {"epochs": 1, "batch": 1, "resume": False, "fraction": 1.0},
                  "history": [{"epoch": 1}], "batches_per_epoch": count,
                  "progress": {"train_batches": count, "optimizer_attempts": count, "optimizer_updates": count, "amp_skipped_updates": 0},
                  "checkpoints": {"best": diagnostic.identity(best), "last": diagnostic.identity(last)},
                  "dataset": {"provenance_path": str(directory / "export/report.json"), "provenance_sha256": sha(directory / "export/report.json"),
                              "counts": {"train": count, "val": len(source_rows["val"])}}}
        if role == "candidate":
            report["initialization"] = {"kind": "completed_best_checkpoint", "training_report": completed["baseline"]["training_report"],
                                        "checkpoint": completed["baseline"]["checkpoint"], "resume": False, "optimizer_state": "fresh"}
            report["pretrained"] = completed["baseline"]["checkpoint"]
        else:
            report["initialization"] = {"kind": "official_pretrained"}
        report_path = directory / "training_report.json"
        write_json(report_path, report)
        config = {"segmentation_training_report": str(report_path), "segmentation_training_report_sha256": sha(report_path)}
        completed[role] = verify_completion(config, best)
        reports[role], paths[role], configs[role] = report, report_path, config
    return completed, prepared, reports, paths, configs


def test_pair_uses_real_export_validation_and_exact_prepared_partitions(tmp_path):
    completed, prepared, _, _, _ = paired_exports(tmp_path)
    proof, ledger = diagnostic.verify_model_pair(completed, prepared)
    assert proof["immediate_parent_bound_to_baseline"]
    assert proof["candidate_corrected_train_and_dev_image_target_multiplicity_verified"]
    assert proof["models"]["baseline"]["direct_training_overlap_dev_images"] == 1
    assert proof["models"]["candidate"]["direct_training_overlap_dev_images"] == 0
    assert proof["prepared_counts"] == {"train": 1, "dev": 1}
    ledger.verify_unchanged()


@pytest.mark.parametrize("fault", ["swapped", "same", "different_parent", "parent_weight", "pretrained_weight", "parent_size"])
def test_pair_identity_and_parent_mismatches_rejected(tmp_path, fault):
    completed, prepared, reports, paths, configs = paired_exports(tmp_path)
    if fault == "swapped":
        completed = {"baseline": completed["candidate"], "candidate": completed["baseline"]}
    elif fault == "same":
        completed["candidate"] = deepcopy(completed["baseline"])
    else:
        initializer = reports["candidate"]["initialization"]
        if fault == "different_parent":
            initializer["training_report"] = {**initializer["training_report"], "sha256": "0" * 64}
        elif fault == "parent_weight":
            initializer["checkpoint"] = {**initializer["checkpoint"], "sha256": "0" * 64}
        elif fault == "parent_size":
            initializer["training_report"] = {**initializer["training_report"], "bytes": 1}
        else:
            reports["candidate"]["pretrained"] = {**reports["candidate"]["pretrained"], "sha256": "0" * 64}
        write_json(paths["candidate"], reports["candidate"])
        completed["candidate"]["training_report"] = diagnostic.identity(paths["candidate"])
    with pytest.raises(ValueError, match="distinct|baseline|parent|checkpoint"):
        diagnostic.verify_model_pair(completed, prepared)


@pytest.mark.parametrize("fault", ["wrong_val", "old_train_target", "old_val_target", "report_count", "val_multiplicity"])
def test_candidate_wrong_validation_old_targets_and_count_rejected(tmp_path, fault):
    completed, prepared, reports, paths, _ = paired_exports(tmp_path)
    export_path = Path(reports["candidate"]["dataset"]["provenance_path"])
    export = json.loads(export_path.read_text(encoding="utf-8"))
    split = "train" if fault == "old_train_target" else "val"
    manifest = Path(export["manifest_outputs"][split]["path"])
    exported = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if fault == "wrong_val":
        exported[0]["image_sha256"] = "0" * 64
    elif fault in ("old_train_target", "old_val_target"):
        exported[0]["input_target_sha256"] = "0" * 64
    elif fault == "val_multiplicity":
        exported.append(deepcopy(exported[0]))
    else:
        reports["candidate"]["dataset"]["counts"]["val"] += 1
    manifest.write_text("".join(json.dumps(row) + "\n" for row in exported), encoding="utf-8")
    export["manifest_outputs"][split].update(diagnostic.identity(manifest))
    write_json(export_path, export)
    marker_path = export_path.parent / ".complete"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["report_sha256"] = sha(export_path)
    write_json(marker_path, marker)
    reports["candidate"]["dataset"]["provenance_sha256"] = sha(export_path)
    write_json(paths["candidate"], reports["candidate"])
    completed["candidate"]["training_report"] = diagnostic.identity(paths["candidate"])
    with pytest.raises(ValueError, match="selection development|corrected training|count mismatch|duplicate"):
        diagnostic.verify_model_pair(completed, prepared)


def test_actual_package_versions_are_aggregate_safe(monkeypatch):
    monkeypatch.setattr(diagnostic.metadata, "version", lambda name: "fixture-" + name)
    versions = diagnostic.package_versions()
    assert set(versions) == {"python", "torch", "numpy", "Pillow", "ultralytics"}
    assert versions["ultralytics"] == "fixture-ultralytics"


def test_swapped_cli_pair_rejected_before_predictor_construction(tmp_path, monkeypatch):
    import torch
    from camera_rc.driving import yolo_semantic as adapter
    completed, prepared, _, paths, _ = paired_exports(tmp_path)
    # main() deliberately constrains its own fresh process. Restore these keys
    # after exercising it in pytest so later tests retain their environment.
    for name in ("CUDA_VISIBLE_DEVICES", "YOLO_OFFLINE", "YOLO_AUTOINSTALL"):
        monkeypatch.setenv(name, diagnostic.os.environ.get(name, ""))
    monkeypatch.setattr(diagnostic, "load_prepared", lambda *args: ([], prepared))
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    monkeypatch.setattr(torch, "set_num_interop_threads", lambda value: None)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    def forbidden_predictor(*args, **kwargs):
        pytest.fail("Invalid role/lineage must fail before predictor construction")
    monkeypatch.setattr(adapter, "VerifiedYoloSemantic", forbidden_predictor)
    output = tmp_path / "private_diagnostic"
    # CI places --basetemp inside the checkout. This test exercises lineage,
    # so give it an isolated public-root boundary with output outside that
    # boundary. Separate tests still exercise the real production privacy gate.
    fixture_public_root = tmp_path / "fixture_public_tree"
    monkeypatch.setattr(diagnostic, "__file__", str(fixture_public_root / "scripts/diagnose_corrected_dev_roi.py"))
    assert not output.resolve().is_relative_to(fixture_public_root.resolve())
    args = ["--prepared-report", prepared["prepared_report"]["path"],
            "--prepared-report-sha256", prepared["prepared_report"]["sha256"],
            "--frame-accounting", "not_read_in_mock", "--frame-accounting-sha256", "0" * 64, "--output", str(output)]
    for supplied, actual in (("baseline", "candidate"), ("candidate", "baseline")):
        args.extend([f"--{supplied}-training-report", str(paths[actual]),
                     f"--{supplied}-training-report-sha256", completed[actual]["training_report"]["sha256"],
                     f"--{supplied}-checkpoint", completed[actual]["checkpoint"]["path"]])
    with pytest.raises(ValueError, match="completed baseline parent"):
        diagnostic.main(args)
    assert not output.exists()
