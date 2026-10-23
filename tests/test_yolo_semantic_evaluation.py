"""CPU contract tests only; fabricated logits are not model performance evidence."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location("yolo_semantic_evaluation_test", SCRIPTS / "evaluate_yolo_semantic.py")
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
from run_large_data_queue import compare_reports


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def identity(path):
    return {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}


def bundle(tmp_path):
    config, best, last = (tmp_path / name for name in ("config.json", "best.pt", "last.pt"))
    write_json(config, {"fixture": True})
    best.write_bytes(b"fake trained best weights")
    last.write_bytes(b"fake trained last weights")
    provenance = tmp_path / "dataset_report.json"
    write_json(provenance, {"complete": True})
    report = {"schema_version": 1, "model_family": "yolo26x-sem", "task": "semantic",
              "class_names": evaluation.CLASS_NAMES, "road_class_id": 0, "training_complete": True,
              "diagnostic": False, "source_config": identity(config),
              "checkpoints": {"best": identity(best), "last": identity(last)},
              "dataset": {"provenance_path": str(provenance), "provenance_sha256": identity(provenance)["sha256"]},
              "actual_args": {"task": "semantic"}, "history": [{"epoch": 1}]}
    path = tmp_path / "training_report.json"
    write_json(path, report)
    return path, report, config, best, last, provenance


def test_completed_training_report_binds_checkpoint_config_and_provenance(tmp_path):
    path, report, config, best, _, _ = bundle(tmp_path)
    result = evaluation.validate_training_report(path, best, config, evaluation.IdentityLedger())
    assert result["checkpoint"] == identity(best)
    assert result["checkpoint_roles"] == ["best"]
    assert result["identity"]["sha256"] == identity(path)["sha256"]


@pytest.mark.parametrize("field,value", [
    ("schema_version", True), ("model_family", "yolo26n-sem"), ("task", "segment"),
    ("class_names", ["not_road_surface", "road_surface"]), ("road_class_id", True),
    ("training_complete", False), ("diagnostic", True), ("history", []), ("actual_args", None),
])
def test_wrong_or_incomplete_training_report_rejected(tmp_path, field, value):
    path, report, config, best, _, _ = bundle(tmp_path)
    report[field] = value
    write_json(path, report)
    with pytest.raises(ValueError):
        evaluation.validate_training_report(path, best, config, evaluation.IdentityLedger())


@pytest.mark.parametrize("changed", ["best", "last", "config", "provenance", "unknown_checkpoint"])
def test_changed_inputs_or_unreported_checkpoint_rejected(tmp_path, changed):
    path, report, config, best, last, provenance = bundle(tmp_path)
    files = {"best": best, "last": last, "config": config, "provenance": provenance}
    if changed == "unknown_checkpoint":
        best = tmp_path / "unknown.pt"
        best.write_bytes(b"different weights")
    else:
        files[changed].write_bytes(b'{"different":true}')
    with pytest.raises(ValueError):
        evaluation.validate_training_report(path, best, config, evaluation.IdentityLedger())


def test_ledger_detects_change_after_decode_and_snapshot_mismatch(tmp_path):
    path = tmp_path / "labels.png"
    path.write_bytes(b"first")
    ledger = evaluation.IdentityLedger()
    ledger.expect(path, identity(path)["sha256"])
    ledger.read(path)
    path.write_bytes(b"other")
    with pytest.raises(ValueError, match="snapshot"):
        ledger.verify_unchanged()
    path.write_bytes(b"first")
    ledger = evaluation.IdentityLedger()
    ledger.read(path)
    path.write_bytes(b"other")
    with pytest.raises(RuntimeError, match="changed"):
        ledger.verify_unchanged()


class FakeHead:
    nc, export, bake_argmax = 2, False, False


class FakeModel:
    def __init__(self):
        self.model = [FakeHead()]
        self.names = dict(enumerate(evaluation.CLASS_NAMES))
        self.yaml = {"scale": "x", "yaml_file": "yolo26x-sem.yaml"}


def test_model_contract_allows_only_two_class_x_semantic_native_logits():
    wrapper = SimpleNamespace(task="semantic", model=FakeModel())
    assert evaluation.validate_semantic_model(wrapper, model_class=FakeModel, head_class=FakeHead) is wrapper.model


@pytest.mark.parametrize("wrong", ["task", "scale", "generation", "one_class", "names", "export", "baked_argmax", "head_type"])
def test_wrong_architecture_or_discrete_export_rejected(wrong):
    wrapper = SimpleNamespace(task="semantic", model=FakeModel())
    if wrong == "task": wrapper.task = "segment"
    if wrong == "scale": wrapper.model.yaml["scale"] = "n"
    if wrong == "generation": wrapper.model.yaml["yaml_file"] = "yolo11x-sem.yaml"
    if wrong == "one_class": wrapper.model.model[-1].nc = 1
    if wrong == "names": wrapper.model.names = {0: "nonroad", 1: "road"}
    if wrong == "export": wrapper.model.model[-1].export = True
    if wrong == "baked_argmax": wrapper.model.model[-1].bake_argmax = True
    if wrong == "head_type": wrapper.model.model[-1] = object()
    with pytest.raises(ValueError):
        evaluation.validate_semantic_model(wrapper, model_class=FakeModel, head_class=FakeHead)


def test_road_probability_is_softmax_after_logit_resize_not_sigmoid_or_resized_softmax():
    logits = torch.tensor([[[[4.0, 0.0]], [[3.0, -3.0]]]])
    actual = evaluation.restore_road_probability(logits, (2, 4), (2, 4))
    expected = F.interpolate(logits, (2, 4), mode="bilinear", align_corners=False).softmax(1)[0, 0].numpy()
    wrong = F.interpolate(logits.softmax(1)[:, :1], (2, 4), mode="bilinear", align_corners=False)[0, 0].numpy()
    assert np.allclose(actual, expected)
    assert not np.allclose(actual, wrong)
    assert actual[0, 0] == pytest.approx(torch.tensor(1.0).sigmoid().item())
    assert actual[0, 0] != pytest.approx(torch.tensor(4.0).sigmoid().item())


@pytest.mark.parametrize("bad", [torch.zeros(1, 2, 2), torch.zeros(1, 1, 2, 2),
                                 torch.zeros(1, 2, 2, 2, dtype=torch.long),
                                 (torch.zeros(1, 2, 2, 2),), torch.full((1, 2, 2, 2), float("nan"))])
def test_invalid_raw_output_never_becomes_probability(bad):
    with pytest.raises(ValueError, match="raw"):
        evaluation.logits_at_input(bad, (192, 320), 1)


@pytest.mark.parametrize("size", [(192, 320), (320, 576), (640, 640)])
def test_supported_model_input_sizes(size):
    assert evaluation.validate_input_size(size) == size


@pytest.mark.parametrize("size", [(True, 320), (191, 320), (0, 320), (192,), (192, 320.0)])
def test_invalid_input_sizes_rejected(size):
    with pytest.raises(ValueError):
        evaluation.validate_input_size(size)


def labelled_files(tmp_path, name, *, pixel=25):
    image, target = tmp_path / (name + ".png"), tmp_path / (name + "_mask.png")
    rgb = np.full((10, 20, 3), pixel, np.uint8)
    labels = np.zeros((10, 20), np.uint8)
    labels[:, 10:] = 1
    labels[0, 0] = 255
    Image.fromarray(rgb).save(image)
    Image.fromarray(labels).save(target)
    return image, target


class FakePredictor:
    def __init__(self, checkpoint=None, *, input_size=(192, 320), device="cpu", amp=False):
        self.input_size, self.device, self.amp = input_size, torch.device("cpu"), False

    def predict_logits(self, images):
        logits = torch.zeros(len(images), 2, *self.input_size)
        logits[:, 0] = 2
        return logits

    def __call__(self, rgb):
        return evaluation.restore_road_probability(self.predict_logits([rgb]), self.input_size, rgb.shape[:2])


def test_native_gt_grid_full_sample_count_ignore_and_argmax_tie(tmp_path):
    image, target = labelled_files(tmp_path, "native")
    source = {"name": "fixture", "rows": [{"image": str(image), "target": str(target)}] * 3}
    predictor = FakePredictor(input_size=(320, 576))
    result = evaluation.evaluate_native_source(source, predictor, evaluation.IdentityLedger(), batch_size=2)
    resized = np.asarray(Image.open(target).resize((320, 192), Image.Resampling.NEAREST))
    assert result["images"] == 3 and result["batches"] == 2
    assert result["partial_validation"] is False
    assert result["valid_pixels"] == 3 * int((resized != 255).sum())
    assert result["confusion_matrix"] == [[3 * int((resized == 0).sum()), 0], [3 * int((resized == 1).sum()), 0]]


def test_runtime_threshold_all_nonroad_exposes_zero_road_recall_not_false_improvement(tmp_path):
    image, target = labelled_files(tmp_path, "threshold_coverage")
    source = {"name": "fixture", "rows": [{"image": str(image), "target": str(target)}]}
    class EqualLogits(FakePredictor):
        def predict_logits(self, images):
            return torch.zeros(len(images), 2, *self.input_size)
    result = evaluation.evaluate_native_source(source, EqualLogits(), evaluation.IdentityLedger())
    threshold = result["runtime_threshold_metrics"]
    labels = np.asarray(Image.open(target).resize((320, 192), Image.Resampling.NEAREST))
    roads, nonroads = int((labels == 0).sum()), int((labels == 1).sum())
    assert threshold["confusion_matrix"] == [[0, roads], [0, nonroads]]
    assert threshold["road_recall"] == 0 and threshold["road_iou"] == 0
    assert threshold["road_precision"] is None and threshold["false_road_rate"] == 0
    assert threshold["valid_pixels"] == roads + nonroads
    # Argmax ties still choose class 0, exactly as in the historical evaluator.
    assert result["confusion_matrix"] == [[roads, 0], [nonroads, 0]]
    assert threshold["threshold"] == .65 and threshold["comparison"] == ">="
    assert "Historical TinyUNet" in threshold["comparison_limit"]


def test_runtime_threshold_exact_probability_tie_is_inclusive_and_preserves_ignore(tmp_path):
    image, target = labelled_files(tmp_path, "threshold_tie")
    source = {"name": "fixture", "rows": [{"image": str(image), "target": str(target)}]}
    class ConstantLogits(FakePredictor):
        def predict_logits(self, images):
            logits = torch.zeros(len(images), 2, *self.input_size)
            logits[:, 0] = 1
            return logits
    # Use the exact representable FP32 softmax value, not a rounded decimal.
    threshold = float(torch.tensor([1., 0.]).softmax(0)[0])
    result = evaluation.evaluate_native_source(source, ConstantLogits(), evaluation.IdentityLedger(),
                                                runtime_threshold=threshold)
    runtime = result["runtime_threshold_metrics"]
    assert runtime["threshold"] == threshold
    assert runtime["confusion_matrix"] == result["confusion_matrix"]
    assert runtime["road_recall"] == 1 and runtime["false_road_rate"] == 1
    assert runtime["valid_pixels"] == result["valid_pixels"] < 192 * 320


def test_original_obstacle_pixel_remains_in_denominator_after_small_input_prediction(tmp_path):
    image, target = labelled_files(tmp_path, "obstacle")
    mask = np.zeros((10, 20), np.uint8)
    mask[0, 1] = 1
    Image.fromarray(mask).save(target)
    discovered = {"pairs": [{"frame_id": "one", "image": image, "target": target}], "root": str(tmp_path),
                  "available_images": 1, "unlabelled_images_not_evaluated": 0, "expected_public_gt_images": 1}
    snapshot = {"training_image_sha256": set(), "training_rgb_sha256": set()}
    result = evaluation.evaluate_public_track(discovered, FakePredictor(), "obstacle", evaluation.IdentityLedger(), .65, snapshot)
    metric = result["aggregate"]["decisions"]["runtime_road"]
    assert metric["labelled_obstacle_pixels"] == 1
    assert metric["obstacle_pixels_predicted_road"] == 1
    assert result["per_image"][0]["original_image_size"] == [10, 20]


@pytest.mark.parametrize("mode", ["bytes", "decoded_rgb"])
def test_public_eval_copied_image_training_leakage_rejected(tmp_path, mode):
    image, target = labelled_files(tmp_path, "copied")
    rgb, _, image_id, _ = evaluation.read_labelled_pair(image, target, evaluation.IdentityLedger())
    snapshot = {"training_image_sha256": set(), "training_rgb_sha256": set()}
    if mode == "bytes": snapshot["training_image_sha256"].add(image_id["sha256"])
    else:
        digest = hashlib.sha256(b"RGB:20:10:")
        digest.update(rgb.tobytes())
        snapshot["training_rgb_sha256"].add(digest.hexdigest())
    with pytest.raises(ValueError, match="training"):
        evaluation._assert_public_hash_not_training(rgb, image_id, snapshot)


def complete_export(tmp_path):
    """Build a tiny real export with distinct original drives and pixel bytes."""
    from prepare_yolo_semantic import prepare
    train_image, train_target = labelled_files(tmp_path, "train", pixel=25)
    val_image, val_target = labelled_files(tmp_path, "val", pixel=70)
    config_path = tmp_path / "config.json"
    for split, image, target in (("train", train_image, train_target), ("val", val_image, val_target)):
        (tmp_path / (split + ".jsonl")).write_text(json.dumps({"image": str(image), "target": str(target),
            "dataset_id": "fixture", "group_id": split, "origin_id": split, "label_space_id": "road_surface_v1"}) + "\n")
    config = {"task": "segmentation", "model": {"num_classes": 2}, "training": {"amp": False},
              "data": {"train_manifest": "train.jsonl", "val_manifest": "val.jsonl", "label_space_id": "road_surface_v1", "image_size": [192, 320]}}
    write_json(config_path, config)
    exported = prepare(config_path, tmp_path / "export")
    best, last = tmp_path / "best.pt", tmp_path / "last.pt"
    best.write_bytes(b"best"); last.write_bytes(b"last")
    provenance = tmp_path / "export" / "report.json"
    report = {"schema_version": 1, "model_family": "yolo26x-sem", "task": "semantic", "class_names": evaluation.CLASS_NAMES,
              "road_class_id": 0, "training_complete": True, "diagnostic": False, "source_config": identity(config_path),
              "checkpoints": {"best": identity(best), "last": identity(last)}, "actual_args": {}, "history": [{"epoch": 1}],
              "dataset": {"provenance_path": str(provenance), "provenance_sha256": identity(provenance)["sha256"]}}
    training_report = tmp_path / "training_report.json"
    write_json(training_report, report)
    return config_path, best, training_report, val_target


@pytest.mark.parametrize("size", [(192, 320), (320, 576)])
def test_full_mock_evaluation_has_explicit_resolution_condition_and_no_real_validation(tmp_path, monkeypatch, size):
    config, best, training_report, _ = complete_export(tmp_path)
    monkeypatch.setattr(evaluation, "YoloSemanticPredictor", FakePredictor)
    result = evaluation.evaluate(config, best, training_report, tmp_path / "evaluation.json", device="cpu", input_size=size,
                                 runtime_threshold=.9)
    assert result["input_size"] == list(size)
    assert result["native_gt_size"] == [192, 320]
    assert result["matched_input_resolution_to_native_baseline"] is (size == (192, 320))
    assert result["validation_sources"][0]["images"] == 1
    assert not result["deployment_validated"] and not result["real_performance_validated"]
    assert not result["models_validated_for_control"] and not result["checkpoint_diagnostic"]
    assert len(result["input_content_identities"]) >= 10
    for source in result["native_sources"].values():
        assert source["runtime_threshold_metrics"]["threshold"] == .9
        assert source["runtime_threshold_metrics"]["road_recall"] == 0
    # The historical comparator must reject changed input resolutions before scores.
    if size != (192, 320):
        baseline = deepcopy(result)
        baseline["input_size"] = [192, 320]
        with pytest.raises(ValueError, match="input_size"):
            compare_reports(baseline, result)


def test_changed_original_gt_after_export_fails_before_success_report(tmp_path, monkeypatch):
    config, best, training_report, val_target = complete_export(tmp_path)
    Image.fromarray(np.zeros((10, 20), np.uint8)).save(val_target)
    monkeypatch.setattr(evaluation, "YoloSemanticPredictor", FakePredictor)
    output = tmp_path / "evaluation.json"
    with pytest.raises(ValueError, match="snapshot"):
        evaluation.evaluate(config, best, training_report, output, device="cpu")
    assert not output.exists()


def test_checkpoint_mutation_during_mock_inference_never_publishes(tmp_path, monkeypatch):
    config, best, training_report, _ = complete_export(tmp_path)
    class ChangingPredictor(FakePredictor):
        def predict_logits(self, images):
            best.write_bytes(b"changed during evaluation")
            return super().predict_logits(images)
    monkeypatch.setattr(evaluation, "YoloSemanticPredictor", ChangingPredictor)
    output = tmp_path / "evaluation.json"
    with pytest.raises(RuntimeError, match="changed"):
        evaluation.evaluate(config, best, training_report, output, device="cpu")
    assert not output.exists()


def test_export_incomplete_marker_blocks_before_predictor(tmp_path, monkeypatch):
    config, best, training_report, _ = complete_export(tmp_path)
    (tmp_path / "export" / ".incomplete").write_text("interrupted")
    def no_model(*args, **kwargs):
        pytest.fail("Incomplete provenance must fail before loading a model")
    monkeypatch.setattr(evaluation, "YoloSemanticPredictor", no_model)
    with pytest.raises(ValueError, match="complete"):
        evaluation.evaluate(config, best, training_report, tmp_path / "evaluation.json", device="cpu")


def test_full_public_subset_report_is_compatible_with_existing_comparator(tmp_path, monkeypatch):
    config, best, training_report, _ = complete_export(tmp_path)
    roots = {}
    for offset, (track, track_spec) in enumerate(evaluation.TRACKS.items()):
        root = tmp_path / track_spec["folder"]
        (root / "images").mkdir(parents=True)
        (root / "labels_masks").mkdir()
        for i in range(track_spec["public_gt_images"]):
            rgb = np.full((10, 20, 3), 100 + 40 * offset + i, np.uint8)
            target = np.zeros((10, 20), np.uint8)
            target[0, 0] = 1
            Image.fromarray(rgb).save(root / "images" / (f"frame_{i:03d}" + track_spec["extension"]))
            Image.fromarray(target).save(root / "labels_masks" / f"frame_{i:03d}_labels_semantic.png")
        roots[track] = root
    monkeypatch.setattr(evaluation, "YoloSemanticPredictor", FakePredictor)
    candidate = evaluation.evaluate(config, best, training_report, tmp_path / "candidate.json", device="cpu",
                                    obstacle_root=roots["obstacle"], anomaly_root=roots["anomaly"])
    # A fabricated equivalent-score baseline checks the legacy report contract only.
    baseline = deepcopy(candidate)
    baseline["checkpoint"] = {"path": "synthetic_native_baseline", "sha256": "a" * 64, "bytes": 1}
    baseline["model_family"] = "synthetic_TinyUNet_fixture"
    result = compare_reports(baseline, candidate)
    assert result["regressions"] == []
    assert not result["deployment_approved"]
    assert result["compared"]["obstacle"]["images"] == 30
    assert result["compared"]["anomaly"]["images"] == 10
    assert candidate["public_obstacle_anomaly_subsets"]["obstacle"]["aggregate"]["decisions"]["runtime_road"]["labelled_obstacle_pixels"] == 30


def test_existing_report_is_never_overwritten_or_model_loaded(tmp_path, monkeypatch):
    output = tmp_path / "existing.json"
    output.write_text("keep original evidence")
    monkeypatch.setattr(evaluation, "validate_training_report", lambda *args: pytest.fail("Must stop before model/provenance loading"))
    with pytest.raises(FileExistsError):
        evaluation.evaluate(tmp_path / "config.json", tmp_path / "best.pt", tmp_path / "training.json", output)
    assert output.read_text() == "keep original evidence"
