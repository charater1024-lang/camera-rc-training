"""Bounded, original-resolution diagnostic of corrected development labels.

This is NOT the frozen official evaluation or a model-selection gate. It includes
every development frame with an actual 255 -> 1 change (at most three), never
selects training examples, and performs exactly one CPU FP32 forward per frame
and model. Full output contains private paths; only aggregate.json is shareable.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from io import BytesIO
import json
import os
from pathlib import Path
import platform

import numpy as np
from PIL import Image, ImageDraw

from prepare_aihub_road import annotation_errors
from verify_aihub_preprocessing import (ADDED_CLASSES, digest, indexed, path_key,
                                       pixel_counts, read_verified, require)

PROTOCOL = "corrected_parent_exposed_dev_original_roi_cpu_v1"
INPUT_SIZE = (384, 640)
CONTEXT_GT_SIZE = (192, 320)
THRESHOLD = 0.65
LIMITS = [
    "수정된 개발 프레임 최대 3장만 포함하는 진단이며 전체 데이터 성능이 아닙니다.",
    "부모 모델이 이미 본 개발 자료입니다. 독립 일반화나 통계적 유의성을 입증하지 않습니다.",
    "추가 학습과 라벨 교정의 효과를 인과적으로 분리하지 않습니다.",
    "원본 좌표의 255→1 변화 영역을 측정합니다. 기존 192×320 GT 평가와 합산하지 않습니다.",
    "nearest 축소 시 작은 변화 영역이 사라질 수 있습니다. 원본 ROI 결과는 탐지 거리/객체 재현율이 아닙니다.",
    "공식 검증/공개 장애물 40장, 임계값 튜닝, 새 모델 선택 기준에 사용하지 않습니다.",
    "포함되지 않은 클래스·주야간 조건에는 결론을 내리지 않습니다. 80km/h·물리 제어·무사고를 검증하지 않습니다.",
    "CPU 정확도 진단이며 속도 측정이나 탑재 GPU 성능 검증이 아닙니다.",
]


def identity(path, expected=None):
    data, sha = read_verified(path, expected)
    return {"path": str(Path(path).resolve()), "sha256": sha, "bytes": len(data)}


def _count(value, name):
    require(type(value) is int and value >= 0, f"Nonnegative integer required: {name}")
    return value


def frame_key(row):
    require(isinstance(row, dict) and isinstance(row.get("dataset_id"), str) and row["dataset_id"]
            and isinstance(row.get("sample_id"), str) and row["sample_id"], "Frame identity required")
    return row["dataset_id"], "sample_id", row["sample_id"]


def manifest_frame_key(row):
    """Keep all dev metadata, including VKITTI exports without sample_id.

    Corrected AI Hub frames and the change ledger still require sample_id.
    Export IDs use a separate namespace: frame_id alone repeats across virtual
    scene variants/cameras and is not a valid identity.
    """
    require(isinstance(row, dict) and isinstance(row.get("dataset_id"), str) and row["dataset_id"],
            "Frame identity required")
    if row.get("sample_id") is not None or row["dataset_id"].startswith("aihub_"):
        return frame_key(row)
    require(isinstance(row.get("export_id"), str) and row["export_id"], "Manifest export identity required")
    return row["dataset_id"], "export_id", row["export_id"]


def check_accounting(report, changes, accounting):
    require(isinstance(changes, list) and changes, "Nonempty change ledger required")
    indexed(changes, frame_key, "change ledger frame")
    for row in changes:
        _count(row.get("ignore_to_nonroad_pixels"), "changed pixels")
        require(row.get("new_split") in {"train", "dev"}, "Only train/dev ledger entries allowed")
        require(isinstance(row.get("polygons"), list), "Original annotation polygons required")
    positive = [row for row in changes if row["ignore_to_nonroad_pixels"] > 0]
    zero = {frame_key(row) for row in changes if row["ignore_to_nonroad_pixels"] == 0}
    zeros_recorded = accounting.get("frames_without_pixel_changes")
    require(isinstance(zeros_recorded, list), "Zero-change accounting required")
    indexed(zeros_recorded, frame_key, "zero-change accounting frame")
    require(zero == {frame_key(row) for row in zeros_recorded}, "Zero-change accounting mismatch")
    require(_count(accounting.get("frames_reprocessed"), "reprocessed") ==
            _count(report.get("changed_frames"), "report changed_frames") == len(changes),
            "Reprocessed frame count mismatch")
    require(_count(accounting.get("frames_with_pixel_changes"), "actual changes") == len(positive),
            "Actual changed frame count mismatch")
    pixels = sum(row["ignore_to_nonroad_pixels"] for row in changes)
    require(_count(accounting.get("ignore_to_nonroad_pixels"), "accounting pixels") ==
            _count(report.get("ignore_to_nonroad_pixels"), "report pixels") == pixels,
            "Total changed pixel accounting mismatch")
    require(_count(report.get("changed_polygons"), "polygon total") ==
            sum(len(row["polygons"]) for row in changes), "Polygon accounting mismatch")
    selected = sorted((row for row in positive if row["new_split"] == "dev"), key=frame_key)
    require(1 <= len(selected) <= 3, "Require all changed development frames, at most three")
    return selected


def changed_roi(old, new, expected_pixels):
    """Check changes before any resizing; a lost nearest-neighbor pixel is retained."""
    pixel_counts(old)
    pixel_counts(new)
    require(old.shape == new.shape, "Original mask dimensions differ")
    roi = old != new
    require(np.all(old[roi] == 255) and np.all(new[roi] == 1), "Only 255 to 1 changes allowed")
    require(np.array_equal(old == 0, new == 0), "Road supervision changed")
    require(int(roi.sum()) == _count(expected_pixels, "expected ROI") > 0, "ROI pixel count mismatch")
    return roi


def load_prepared(report_path, expected_report_sha, accounting_path, expected_accounting_sha):
    """Validate metadata globally, decode only changed dev masks/images/annotations."""
    report_path = Path(report_path).resolve(strict=True)
    root = report_path.parent
    checked = []

    def checked_bytes(path, sha=None):
        data, actual = read_verified(path, sha)
        checked.append({"path": str(Path(path).resolve()), "sha256": actual, "bytes": len(data)})
        return data

    def checked_json(path, sha=None):
        return json.loads(checked_bytes(path, sha).decode("utf-8-sig"))

    report = checked_json(report_path, digest(expected_report_sha))
    require(report.get("complete") is True and report.get("format") == "aihub_training_preprocessing_revision_v1",
            "Completed preprocessing revision required")
    require(not (root / "failure.json").exists(), "Preprocessing failure marker exists")
    marker = checked_json(root / ".complete")
    require(marker.get("report_sha256") == expected_report_sha, "Preprocessing completion marker differs")
    changes = checked_json(root / "changes.json", report["changes_sha256"])
    accounting = checked_json(accounting_path, digest(expected_accounting_sha))
    selected = check_accounting(report, changes, accounting)
    mapping = checked_json(root / "mapping.json", report["mapping_sha256"])
    require(ADDED_CLASSES <= set(mapping["nonroad1"]) and mapping.get("road0") == ["freespace"]
            and mapping.get("all_other_classes") == 255
            and mapping.get("official_reporting_mapping_changed") is False,
            "Unsupported correction mapping")
    manifest = report["manifests"]["dev"]
    require(path_key(manifest["path"]) == path_key(str(root / "dev.jsonl")), "Unexpected dev manifest path")
    rows = [json.loads(line) for line in checked_bytes(manifest["path"], manifest["sha256"])
            .decode("utf-8-sig").splitlines() if line.strip()]
    require(len(rows) == manifest["samples"], "Development sample count mismatch")
    by_id = indexed(rows, manifest_frame_key, "development frame")
    for row in rows:
        require(row.get("split") == row.get("export_split") == "dev", "Non-dev row in dev manifest")
        require(row.get("preprocessing_provenance", {}).get("previous_export_split") == "train"
                and row["preprocessing_provenance"].get("development_role") == "parent_exposed_selection_only",
                "Prepared development ancestry required")
    # Bind every change in dev to the ledger, including zero-change entries.
    ledger_dev = {frame_key(row) for row in changes if row["new_split"] == "dev"}
    metadata_changed = {manifest_frame_key(row) for row in rows if row["target_sha256"] !=
                        row["preprocessing_provenance"]["previous_target_sha256"]}
    require(metadata_changed <= ledger_dev, "Development mask change absent from ledger")
    require(ledger_dev <= set(by_id), "Ledger development frame absent from manifest")
    require(not ({frame_key(row) for row in changes if row["new_split"] == "train"} & set(by_id)),
            "Training ledger frame appears in development")
    frames = []
    for change in selected:
        row = by_id[frame_key(change)]
        previous = row["preprocessing_provenance"]
        require(row["image"] == change["image"] and row["target"] == change["new_target"]
                and row["target_sha256"] == change["new_sha256"]
                and previous["previous_target"] == change["old_target"]
                and previous["previous_target_sha256"] == change["old_sha256"]
                and row["target_original"] == change["source_annotation"]
                and row["target_original_sha256"] == change["source_annotation_sha256"]
                and row["development_group_id"] == change["development_group_id"],
                "Ledger/frame provenance mismatch")
        masks = []
        for path, sha in ((change["old_target"], change["old_sha256"]), (row["target"], row["target_sha256"])):
            with Image.open(BytesIO(checked_bytes(path, sha))) as source:
                masks.append(np.array(source))
        old, new = masks
        require(old.shape == new.shape == (row["height"], row["width"]), "Original mask dimensions mismatch")
        roi = changed_roi(old, new, change["ignore_to_nonroad_pixels"])
        require(pixel_counts(new) == {str(k): v for k, v in row["target_pixel_counts"].items() if v},
                "Prepared target counts differ")
        annotation = checked_json(change["source_annotation"], change["source_annotation_sha256"])
        require(not annotation_errors(annotation, (row["width"], row["height"]),
                                      Path(row["sample_id"]).name), "Invalid original human annotation")
        polygons = [{"polygon_index": i, "class": p["class"], "polygon": p["polygon"]}
                    for i, p in enumerate(annotation["annotations"]) if p["class"] in ADDED_CLASSES]
        require(polygons == change["polygons"] and polygons, "Ledger does not match original human polygons")
        canvas = Image.new("L", (row["width"], row["height"]), 0)
        draw = ImageDraw.Draw(canvas)
        for polygon in polygons:
            values = polygon["polygon"]
            draw.polygon(list(zip(values[::2], values[1::2])), fill=1)
        covered = np.asarray(canvas, dtype=bool)
        require(np.array_equal(roi, covered & (old == 255)) and np.all(new[covered] == 1),
                "Changed ROI differs from explicit human obstacle polygons")
        with Image.open(BytesIO(checked_bytes(row["image"], row["image_sha256"]))) as image:
            rgb = np.array(image.convert("RGB"))
        require(rgb.shape == (row["height"], row["width"], 3), "Image and original masks differ in size")
        rgb_sha = hashlib.sha256(f"RGB:{row['width']}:{row['height']}:".encode() + rgb.tobytes()).hexdigest()
        require(rgb_sha == row["image_rgb_sha256"], "Decoded RGB hash mismatch")
        native_roi = np.asarray(Image.fromarray(roi.astype(np.uint8)).resize(
            (CONTEXT_GT_SIZE[1], CONTEXT_GT_SIZE[0]), Image.Resampling.NEAREST), dtype=bool)
        frames.append({"row": row, "rgb": rgb, "roi": roi, "road": new == 0,
                       "classes": sorted({p["class"] for p in polygons}),
                       "changed_roi_pixels_at_192x320_nearest_context_only": int(native_roi.sum())})
    return frames, {"prepared_report": checked[0], "dev_manifest_sha256": manifest["sha256"],
                    "frame_accounting": accounting, "files_checked": checked,
                    "selection_rule": "All and only dev ledger frames with actual 255-to-1 changes, sorted by identity; no model outputs used",
                    "selected_frames": len(frames)}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else None


def evaluate_frames(frames, predictor):
    """One forward per frame. Threshold original-resolution probabilities only."""
    require(1 <= len(frames) <= 3, "Bounded frame count required")
    results = []
    for frame in frames:
        probability = predictor(frame["rgb"])
        require(isinstance(probability, np.ndarray) and probability.dtype == np.float32
                and probability.shape == frame["roi"].shape and np.isfinite(probability).all()
                and np.all((probability >= 0) & (probability <= 1)),
                "Require finite original-resolution float32 probabilities in [0,1]")
        predicted_road = probability >= THRESHOLD
        roi_total, road_total = int(frame["roi"].sum()), int(frame["road"].sum())
        require(roi_total > 0 and not np.any(frame["roi"] & frame["road"]), "Invalid/disjoint ROI and road GT required")
        false_road = int((predicted_road & frame["roi"]).sum())
        road_tp = int((predicted_road & frame["road"]).sum())
        results.append({"dataset_id": frame["row"]["dataset_id"], "sample_id": frame["row"]["sample_id"],
                        "original_size_hw": list(frame["roi"].shape),
                        "corrected_classes": frame.get("classes", []),
                        "changed_roi_pixels_at_192x320_nearest_context_only": frame.get(
                            "changed_roi_pixels_at_192x320_nearest_context_only"),
                        "changed_roi_pixels": roi_total, "changed_roi_false_road_pixels": false_road,
                        "changed_roi_false_road_rate": ratio(false_road, roi_total),
                        "same_frames_road_gt_pixels": road_total, "same_frames_road_tp_pixels": road_tp,
                        "same_frames_road_fn_pixels": road_total - road_tp,
                        "same_frames_road_recall": ratio(road_tp, road_total)})
    total = {key: sum(row[key] for row in results) for key in (
        "changed_roi_pixels", "changed_roi_false_road_pixels", "same_frames_road_gt_pixels",
        "same_frames_road_tp_pixels", "same_frames_road_fn_pixels")}
    total.update({"frames": len(results), "forwards": len(results),
                  "changed_roi_false_road_rate": ratio(total["changed_roi_false_road_pixels"], total["changed_roi_pixels"]),
                  "changed_roi_image_mean_false_road_rate": sum(row["changed_roi_false_road_rate"] for row in results) / len(results),
                  "same_frames_road_recall": ratio(total["same_frames_road_tp_pixels"], total["same_frames_road_gt_pixels"])})
    return {"aggregate": total, "per_frame": results}


def differences(baseline, candidate):
    for key in ("frames", "forwards", "changed_roi_pixels", "same_frames_road_gt_pixels"):
        require(baseline[key] == candidate[key], f"Comparison denominator differs: {key}")
    result = {}
    for key, lower_better in (("changed_roi_false_road_rate", True),
                              ("changed_roi_image_mean_false_road_rate", True),
                              ("same_frames_road_recall", False)):
        old, new = baseline[key], candidate[key]
        delta = new - old if old is not None and new is not None else None
        result[key] = {"baseline": old, "candidate": new, "delta_percentage_points": None if delta is None else 100 * delta,
                       "relative_change_percent": None if delta is None or old == 0 else 100 * delta / old,
                       "relative_improvement_percent": None if delta is None or old == 0 else 100 * delta / old * (-1 if lower_better else 1),
                       "lower_is_better": lower_better}
    return result


def validate_pair_lineage(reports, completed):
    """Bind the candidate's immediate initializer to this baseline, by bytes."""
    baseline, candidate = completed["baseline"], completed["candidate"]
    require(baseline["training_report"]["sha256"] != candidate["training_report"]["sha256"]
            and baseline["checkpoint"]["sha256"] != candidate["checkpoint"]["sha256"],
            "Baseline and candidate must be distinct completed models")
    initializer = reports["candidate"].get("initialization")
    require(isinstance(initializer, dict) and initializer.get("kind") == "completed_best_checkpoint",
            "Candidate requires a completed baseline parent")
    for recorded, expected, name in (
        (initializer.get("training_report"), baseline["training_report"], "parent training report"),
        (initializer.get("checkpoint"), baseline["checkpoint"], "initializer checkpoint"),
        (reports["candidate"].get("pretrained"), baseline["checkpoint"], "pretrained checkpoint"),
    ):
        require(isinstance(recorded, dict) and recorded.get("sha256") == expected["sha256"]
                and type(recorded.get("bytes")) is int and recorded["bytes"] == expected["bytes"],
                f"Candidate {name} does not match the supplied baseline")
    require(initializer.get("resume") is False and initializer.get("optimizer_state") == "fresh",
            "Require the recorded fresh fine-tune initializer")


def verify_model_pair(completed, prepared):
    """Reuse selection evaluation provenance/partition checks before inference.

    Reads manifests/configs/export and model/report bytes, never selects or
    decodes training images. A new ROI score does not weaken the selection gate.
    """
    import evaluate_selection_development as selection
    yolo, frozen = selection.yolo, selection.frozen
    ledger = yolo.IdentityLedger()
    reports = {}
    for role in ("baseline", "candidate"):
        expected = completed[role]["training_report"]
        ledger.expect(expected["path"], expected["sha256"])
        reports[role], actual = yolo._json(expected["path"], ledger)
        require(actual == expected, "Pinned training report identity differs")
    validate_pair_lineage(reports, completed)
    initializer = reports["candidate"]["initialization"]
    # Verify original lineage files too; identical copied report/weights remain
    # acceptable because their content identities must match the supplied pair.
    for item, expected in ((initializer["training_report"], completed["baseline"]["training_report"]),
                           (initializer["checkpoint"], completed["baseline"]["checkpoint"]),
                           (reports["candidate"]["pretrained"], completed["baseline"]["checkpoint"])):
        path = yolo._report_path(item["path"], Path(completed["candidate"]["training_report"]["path"]).parent)
        ledger.expect(path, expected["sha256"])
        require(ledger.identity(path)["bytes"] == expected["bytes"], "Parent lineage file size differs")
    prepared_identity = prepared["prepared_report"]
    ledger.expect(prepared_identity["path"], prepared_identity["sha256"])
    revision, _ = yolo._json(prepared_identity["path"], ledger)
    rows = {}
    for split in ("train", "dev"):
        item = revision["manifests"][split]
        ledger.expect(item["path"], item["sha256"])
        data, _ = ledger.read(item["path"])
        rows[split] = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
        require(len(rows[split]) == item["samples"] and all(
            row.get("split") == row.get("export_split") == split for row in rows[split]),
            f"Prepared {split} count or split mismatch")
    require(revision["manifests"]["dev"]["sha256"] == prepared["dev_manifest_sha256"], "Prepared development changed")
    selection.disjoint(selection.row_keys(rows["train"]), selection.row_keys(rows["dev"]), "Selection train/dev")
    proofs = {}
    for role in ("baseline", "candidate"):
        reported = completed[role]
        completion = yolo.validate_training_report(reported["training_report"]["path"], reported["checkpoint"]["path"],
                                                   reports[role]["source_config"]["path"], ledger)
        config = frozen.load_config(completion["config"]["path"])
        training, validation = frozen.training_sources(config, ledger)
        direct = yolo.validate_export_snapshot(completion, training, validation, ledger)
        overlap = selection.check_model_partition(role, direct["training_image_sha256"], rows["train"], rows["dev"])
        if role == "candidate":
            exported = {}
            for split in ("train", "val"):
                item = direct["manifests"][split]
                ledger.expect(item["path"], item["sha256"])
                data, _ = ledger.read(item["path"])
                exported[split] = [json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip()]
                require(type(reports[role]["dataset"]["counts"].get(split)) is int and
                        reports[role]["dataset"]["counts"][split] == len(exported[split]),
                        f"Candidate report/export {split} count mismatch")
            selection.check_candidate_export_partitions(exported, rows["train"], rows["dev"])
        proofs[role] = {"source_config": completion["config"], "export": completion["dataset_provenance_identity"],
                        "export_manifests": direct["manifests"], "direct_training_overlap_dev_images": overlap}
    ledger.verify_unchanged()
    return {"immediate_parent_bound_to_baseline": True, "distinct_model_reports_and_checkpoints": True,
            "candidate_corrected_train_and_dev_image_target_multiplicity_verified": True,
            "prepared_counts": {key: len(value) for key, value in rows.items()}, "models": proofs,
            "verification_scope": "Immediate parent and both direct training exports; no new image inference or blind-test claim"}, ledger


def package_versions():
    versions = {"python": platform.python_version()}
    for name in ("torch", "numpy", "Pillow", "ultralytics"):
        versions[name] = metadata.version(name)
    return versions


def public_aggregate(report):
    """Allowlist: no paths, frame IDs, source annotations, image hashes or images."""
    return {key: report[key] for key in ("protocol_id", "diagnostic_complete", "scope", "conditions", "coverage", "models",
                                        "aggregate_results", "differences", "limitations", "runtime_comparison_performed",
                                        "training_performed", "threshold_tuning_performed", "model_selection_gate_performed", "package_versions")}


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-report", type=Path, required=True)
    parser.add_argument("--prepared-report-sha256", required=True)
    parser.add_argument("--frame-accounting", type=Path, required=True)
    parser.add_argument("--frame-accounting-sha256", required=True)
    for role in ("baseline", "candidate"):
        parser.add_argument(f"--{role}-training-report", type=Path, required=True)
        parser.add_argument(f"--{role}-training-report-sha256", required=True)
        parser.add_argument(f"--{role}-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True,
                        help="Fresh PRIVATE directory outside the public source tree (e.g. local analysis or runs)")
    return parser


def main(argv=None):
    args = _parser().parse_args(argv)
    output = args.output.resolve()
    public_root = Path(__file__).resolve().parents[1]
    require(not output.is_relative_to(public_root), "Full output contains private paths; use an output outside the public source tree")
    require(not output.exists(), "Output already exists; preserve prior evidence")
    frames, prepared = load_prepared(args.prepared_report, args.prepared_report_sha256,
                                     args.frame_accounting, args.frame_accounting_sha256)
    # Imported only for the explicitly invoked CLI, after all data checks.
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    from camera_rc.driving.yolo_semantic import VerifiedYoloSemantic, verify_completion
    require(not torch.cuda.is_initialized(), "Run in a fresh CPU-only process")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    configurations, checked_models = {}, {}
    for role in ("baseline", "candidate"):
        config = {"segmentation_input_size": list(INPUT_SIZE), "drivable_class_ids": [0],
                  "segmentation_training_report": str(getattr(args, role + "_training_report")),
                  "segmentation_training_report_sha256": getattr(args, role + "_training_report_sha256"),
                  "segmentation_checkpoint_role": "best", "allow_diagnostic": False,
                  "yolo_settings_dir": str(output / "ultralytics_settings")}
        configurations[role] = config
        checked_models[role] = verify_completion(config, getattr(args, role + "_checkpoint"))
    pair_proof, pair_ledger = verify_model_pair(checked_models, prepared)
    output.mkdir(parents=True, exist_ok=False)
    try:
        results = {}
        for role in ("baseline", "candidate"):
            predictor = VerifiedYoloSemantic(configurations[role], getattr(args, role + "_checkpoint"), device="cpu", half=False)
            results[role] = evaluate_frames(frames, predictor)
            del predictor
        require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
        for item in prepared["files_checked"]:
            require(identity(item["path"], item["sha256"]) == item, "Source changed during diagnostic")
        for role in checked_models:
            require(verify_completion(configurations[role], getattr(args, role + "_checkpoint")) == checked_models[role],
                    "Model provenance changed during diagnostic")
        pair_ledger.verify_unchanged()
        report = {"protocol_id": PROTOCOL, "diagnostic_complete": True, "created_utc": datetime.now(timezone.utc).isoformat(),
                  "scope": "parent_exposed_development_label_correction_diagnostic_only",
                  "conditions": {"device": "cpu", "precision": "float32", "cpu_threads": 2, "input_size_hw": list(INPUT_SIZE),
                                 "threshold": THRESHOLD, "roi_grid": "original_mask_resolution_before_any_resize",
                                 "probability": "RGB PIL bilinear resize /255; FP32 logits bilinear to input, softmax road0, probability bilinear to original size; >= threshold",
                                 "one_forward_per_model_per_frame": True,
                                 "context_nearest_gt_size_hw": list(CONTEXT_GT_SIZE)},
                  "coverage": {"frames": len(frames), "datasets": dict(Counter(f["row"]["dataset_id"] for f in frames)),
                               "corrected_class_frame_counts": dict(Counter(c for f in frames for c in f["classes"])),
                               "day_night": "Dataset IDs only; no inferred illumination annotation",
                               "changed_roi_pixels_at_192x320_nearest_context_only": sum(f["changed_roi_pixels_at_192x320_nearest_context_only"] for f in frames)},
                  "models": {role: {"training_report_sha256": value["training_report"]["sha256"],
                                    "checkpoint_sha256": value["checkpoint"]["sha256"], "checkpoint_role": "best"}
                             for role, value in checked_models.items()},
                  "aggregate_results": {role: result["aggregate"] for role, result in results.items()},
                  "differences": differences(results["baseline"]["aggregate"], results["candidate"]["aggregate"]),
                  "limitations": LIMITS, "runtime_comparison_performed": False, "training_performed": False,
                  "threshold_tuning_performed": False, "model_selection_gate_performed": False,
                  "package_versions": package_versions(),
                  "private_provenance": prepared, "private_model_provenance": checked_models,
                  "private_pair_provenance": pair_proof,
                  "private_pair_input_identities": list(pair_ledger.identities.values()),
                  "private_implementation": {"diagnostic": identity(__file__),
                                             "runtime_adapter": identity(Path(__file__).resolve().parents[1] /
                                                                         "src/camera_rc/driving/yolo_semantic.py")},
                  "private_frame_results": {role: result["per_frame"] for role, result in results.items()}}
        for name, value in (("diagnostic.json", report), ("aggregate.json", public_aggregate(report))):
            (output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        (output / ".complete").write_text(json.dumps({"report": identity(output / "diagnostic.json"),
                                                       "aggregate": identity(output / "aggregate.json")}, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"diagnostic_complete": True, "frames": len(frames), "forwards_per_model": len(frames)}, ensure_ascii=False))
    except BaseException as exc:
        (output / "failure.json").write_text(json.dumps({"diagnostic_complete": False, "error": type(exc).__name__,
                                                        "message": str(exc)}, ensure_ascii=False, indent=2), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
