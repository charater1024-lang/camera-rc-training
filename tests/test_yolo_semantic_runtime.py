"""CPU-only runtime contract tests; synthetic logits are not driving evidence."""
import builtins
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from camera_rc.driving import yolo_semantic as adapter
from camera_rc.driving.config import DEFAULTS, check_actuation, load_runtime_config, validate_runtime_config
from camera_rc.driving.perception import CameraPerception


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    yield
    torch.set_num_threads(previous)


def save_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_bundle(tmp_path):
    checkpoint = tmp_path / "copied_best.pt"
    checkpoint.write_bytes(b"trusted fixture bytes, never deserialized")
    recorded = adapter.file_identity(checkpoint)
    # The report is copied unchanged from another training machine.
    recorded["path"] = "X:/original_training/fit/weights/best.pt"
    report = {"schema_version": 1, "model_family": "yolo26x-sem", "task": "semantic",
              "class_names": list(adapter.CLASS_NAMES), "road_class_id": 0,
              "training_complete": True, "diagnostic": False,
              "actual_args": {"epochs": 2, "batch": 4, "resume": False, "fraction": 1.0},
              "history": [{"epoch": 1.0, "metrics/mIoU": .4}, {"epoch": 2.0, "metrics/mIoU": .5}],
              "batches_per_epoch": 3,
              "progress": {"train_batches": 6, "optimizer_attempts": 3, "optimizer_updates": 2, "amp_skipped_updates": 1},
              "dataset": {"counts": {"train": 10, "val": 3}, "provenance_path": "X:/data/report.json",
                          "provenance_sha256": "a" * 64},
              "source_config": {"path": "X:/config.json", "sha256": "b" * 64, "bytes": 5},
              "checkpoints": {"best": recorded, "last": {**recorded, "path": "X:/original_training/last.pt"}}}
    path = tmp_path / "training_report.json"
    save_json(path, report)
    config = {"segmentation_backend": adapter.BACKEND, "segmentation_checkpoint": str(checkpoint),
              "segmentation_training_report": str(path), "segmentation_training_report_sha256": adapter.file_identity(path)["sha256"],
              "segmentation_checkpoint_role": "best", "segmentation_input_size": [32, 64],
              "drivable_class_ids": [0], "half": True, "min_probability": .65}
    return config, report, path, checkpoint


def repin(config, report, path):
    save_json(path, report)
    config["segmentation_training_report_sha256"] = adapter.file_identity(path)["sha256"]


class FakeHead(nn.Module):
    nc, export, bake_argmax = 2, False, False


class FakeSemantic(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([FakeHead()])
        self.names = dict(enumerate(adapter.CLASS_NAMES))
        self.yaml = {"scale": "x", "yaml_file": "yolo26x-sem.yaml"}
        self.weight = nn.Parameter(torch.ones(1))
        self.seen = None

    def forward(self, tensor):
        self.seen = tensor.detach().clone()
        # Low-resolution nonlinear spatial logits make interpolation order and
        # RGB/BGR differences observable rather than testing a constant mask.
        channels = F.adaptive_avg_pool2d(tensor, (2, 3))
        return torch.cat((channels[:, :1] * 8 - 2, channels[:, 2:3] * 5 - 1), 1)


def test_pinned_completion_allows_byte_identical_portable_weight_copy(tmp_path):
    config, _, _, checkpoint = fixture_bundle(tmp_path)
    proof = adapter.verify_completion(config, checkpoint)
    assert proof["checkpoint"]["path"] == str(checkpoint.resolve())
    assert proof["recorded_checkpoint"]["path"].startswith("X:/")
    assert proof["physical_actuation_supported"] is False
    assert proof["completed_epochs"] == 2 and proof["optimizer_updates"] == 2


@pytest.mark.parametrize("change", ["report_bytes", "checkpoint_bytes", "same_size_other_bytes", "missing_pin", "wrong_size", "role"])
def test_hash_and_role_mismatches_fail_before_loading(tmp_path, monkeypatch, change):
    config, report, path, checkpoint = fixture_bundle(tmp_path)
    def forbidden(*_):
        pytest.fail("Invalid provenance reached model loading")
    monkeypatch.setattr(adapter, "_load_semantic_model", forbidden)
    if change == "report_bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "checkpoint_bytes":
        checkpoint.write_bytes(b"changed")
    elif change == "same_size_other_bytes":
        checkpoint.write_bytes(b"!" * checkpoint.stat().st_size)
    elif change == "missing_pin":
        config.pop("segmentation_training_report_sha256")
    elif change == "role":
        config["segmentation_checkpoint_role"] = "epoch1"
    else:
        report["checkpoints"]["best"]["bytes"] += 1
        repin(config, report, path)
    with pytest.raises(ValueError):
        CameraPerception(config, "cpu")


@pytest.mark.parametrize("change", ["complete", "diagnostic", "names", "road_id_bool", "epoch_gap", "nan_history",
                                    "missing_epoch", "fraction", "no_updates", "bad_updates", "bad_batches", "source", "dataset"])
def test_incomplete_or_semantically_invalid_reports_rejected(tmp_path, change):
    config, report, path, checkpoint = fixture_bundle(tmp_path)
    if change == "complete": report["training_complete"] = False
    elif change == "diagnostic": report["diagnostic"] = True
    elif change == "names": report["class_names"].reverse()
    elif change == "road_id_bool": report["road_class_id"] = False
    elif change == "epoch_gap": report["history"][1]["epoch"] = 3
    elif change == "nan_history": report["history"][0]["metrics/mIoU"] = float("nan")
    elif change == "missing_epoch": report["history"].pop()
    elif change == "fraction": report["actual_args"]["fraction"] = .2
    elif change == "no_updates": report["progress"]["optimizer_updates"] = 0
    elif change == "bad_updates": report["progress"]["optimizer_attempts"] = 5
    elif change == "bad_batches": report["progress"]["train_batches"] = 5
    elif change == "source": report["source_config"]["sha256"] = "invalid"
    elif change == "dataset": report["dataset"]["counts"]["train"] = 20
    repin(config, report, path)
    with pytest.raises(ValueError):
        adapter.verify_completion(config, checkpoint)


@pytest.mark.parametrize("change", ["task", "names", "head", "export", "bake", "scale", "architecture"])
def test_checkpoint_model_and_head_are_checked(change):
    model = FakeSemantic()
    wrapper = SimpleNamespace(model=model, task="semantic")
    if change == "task": wrapper.task = "segment"
    elif change == "names": model.names = {0: "not_road_surface", 1: "road_surface"}
    elif change == "head": model.model[0].nc = 3
    elif change == "export": model.model[0].export = True
    elif change == "bake": model.model[0].bake_argmax = True
    elif change == "scale": model.yaml["scale"] = "n"
    elif change == "architecture": model.yaml["yaml_file"] = "yolo26x-seg.yaml"
    with pytest.raises(ValueError):
        adapter.validate_semantic_model(wrapper, model_class=FakeSemantic, head_class=FakeHead)


def test_runtime_matches_existing_offline_evaluator_without_implicit_detector(tmp_path, monkeypatch):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("runtime_equivalence_evaluator", scripts / "evaluate_yolo_semantic.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    config, _, _, _ = fixture_bundle(tmp_path)
    model = FakeSemantic()
    monkeypatch.setattr(adapter, "_load_semantic_model", lambda *_: model)
    runtime = CameraPerception(config, "cpu")
    rgb = np.random.default_rng(33).integers(0, 256, size=(47, 91, 3), dtype=np.uint8)
    # Independent evaluator functions are the contract, not a duplicated test formula.
    expected_input = evaluation.image_tensor(rgb, (32, 64)).unsqueeze(0)
    with torch.inference_mode():
        expected = evaluation.restore_road_probability(model(expected_input), (32, 64), (47, 91))
    expected[expected < .65] = 0
    result = runtime.infer(rgb)
    assert torch.equal(model.seen, expected_input)
    assert np.array_equal(result.drivable, expected)
    assert np.any(result.drivable == 0) and np.any(result.drivable > .65)
    assert result.drivable.dtype == np.float32 and result.drivable.shape == (47, 91)
    assert runtime.detector is None and result.detections == [] and runtime.half is False
    assert runtime.provenance["checkpoint_role"] == "best" and result.inference_ms > 0


@pytest.mark.parametrize("bad", [torch.zeros(1, 2, 3, 4, dtype=torch.long), torch.zeros(1, 1, 3, 4),
                                 torch.zeros(2, 2, 3, 4), (torch.zeros(1, 2, 3, 4),),
                                 torch.full((1, 2, 3, 4), float("nan"))])
def test_baked_invalid_or_nonfinite_output_rejected(bad):
    with pytest.raises(ValueError, match="raw floating"):
        adapter.restore_road_probability(bad, (32, 64), (47, 91))


@pytest.mark.parametrize("change", ["weights", "report", "nonfinite"])
def test_replacement_during_load_and_nonfinite_weights_rejected(tmp_path, monkeypatch, change):
    config, _, path, checkpoint = fixture_bundle(tmp_path)
    def load(*_):
        model = FakeSemantic()
        if change == "weights": checkpoint.write_bytes(b"replacement")
        elif change == "report": path.write_bytes(path.read_bytes() + b" ")
        else: model.weight.data.fill_(float("nan"))
        return model
    monkeypatch.setattr(adapter, "_load_semantic_model", load)
    with pytest.raises(ValueError, match="changed while|nonfinite"):
        CameraPerception(config, "cpu")


@pytest.mark.parametrize("field,value", [("segmentation_input_size", None), ("segmentation_input_size", [33, 64]),
                                        ("drivable_class_ids", [1]), ("drivable_class_ids", [False]),
                                        ("allow_diagnostic", True)])
def test_explicit_input_and_road_semantics_required(tmp_path, field, value):
    config, _, _, _ = fixture_bundle(tmp_path)
    config[field] = value
    with pytest.raises(ValueError):
        CameraPerception(config, "cpu")


def test_backend_is_explicit_and_cannot_bypass_physical_output_gate(tmp_path, monkeypatch):
    config, _, _, _ = fixture_bundle(tmp_path)
    config.pop("segmentation_backend")
    with pytest.raises(ValueError, match="explicit segmentation_backend"):
        CameraPerception(config, "cpu")
    cfg = deepcopy(DEFAULTS)
    cfg["perception"]["segmentation_backend"] = adapter.BACKEND
    check_actuation(cfg, False)
    cfg["transport"]["kind"] = "serial"
    cfg["models_validated_for_control"] = True
    cfg["transport"]["calibration_confirmed"] = True
    cfg["perception"]["offline_readiness_validated"] = True  # unsupported flag cannot bypass
    with pytest.raises(ValueError, match="dry-run/replay only"):
        check_actuation(cfg, True)
    cfg["perception"]["segmentation_backend"] = "typo"
    with pytest.raises(ValueError, match="segmentation_backend"):
        validate_runtime_config(cfg)


def test_runtime_config_resolves_portable_report_and_weight_paths(tmp_path):
    config, _, _, checkpoint = fixture_bundle(tmp_path)
    config["segmentation_checkpoint"] = checkpoint.name
    config["segmentation_training_report"] = "training_report.json"
    config["yolo_settings_dir"] = "settings"
    save_json(tmp_path / "calibration.json", {})
    path = tmp_path / "runtime.json"
    save_json(path, {"perception": config, "calibration": "calibration.json"})
    loaded = load_runtime_config(path)
    assert loaded["perception"]["segmentation_training_report"] == str((tmp_path / "training_report.json").resolve())
    assert loaded["perception"]["segmentation_checkpoint"] == str(checkpoint.resolve())
    assert loaded["perception"]["yolo_settings_dir"] == str((tmp_path / "settings").resolve())
    assert loaded["transport"]["kind"] == "dry_run"


def test_native_backend_does_not_import_ultralytics(tmp_path, monkeypatch):
    from camera_rc.models import build_model
    stored = {"task": "segmentation", "model": {"width": 4, "num_classes": 2}, "data": {"image_size": [16, 24]}}
    path = tmp_path / "native.pt"
    torch.save({"format_version": 1, "config": stored, "model": build_model(stored).state_dict()}, path)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "ultralytics" or name.startswith("ultralytics."):
            pytest.fail("Native backend imported the optional dependency")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    runtime = CameraPerception({"segmentation_checkpoint": str(path), "drivable_class_ids": [0]}, "cpu")
    assert runtime.backend == "native"
    assert runtime.infer(np.zeros((25, 39, 3), dtype=np.uint8)).drivable.shape == (25, 39)
