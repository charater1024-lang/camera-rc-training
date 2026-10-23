import hashlib

import pytest
import torch

from camera_rc.driving.export import export_runtime_checkpoint
from camera_rc.driving.perception import CameraPerception
from camera_rc.models import build_model
from camera_rc.training import _check_compatible, _read_checkpoint


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def training_checkpoint(tmp_path, task="segmentation", **changes):
    config = {"task": task, "model": {"width": 4, "num_classes": 3, "horizon": 2},
              "data": {"image_size": [16, 24], "train_manifest": "private/train.jsonl", "val_manifest": "private/val.jsonl"},
              "training": {"output_dir": "private/runs", "epochs": 3}}
    if task == "trajectory":
        config["data"]["future_offsets_s"] = [0.5, 1.0]
    model = build_model(config).eval()
    stored = {"format_version": 1, "config": config, "model": model.state_dict(), "epoch_complete": True,
              "optimizer": {"unneeded": torch.zeros(50000)}, "torch_rng": torch.get_rng_state(),
              "python_rng": [1, 2, 3], "diagnostic": False, **changes}
    path = tmp_path / "training.pt"
    torch.save(stored, path)
    return path, model, config


@pytest.mark.parametrize("task", ["segmentation", "depth", "trajectory"])
def test_compact_export_preserves_predictions_and_evaluation_contract(tmp_path, task):
    source, original, config = training_checkpoint(tmp_path, task)
    output = tmp_path / "deploy" / "model.pt"
    summary = export_runtime_checkpoint(source, output)
    stored = _read_checkpoint(output)
    _check_compatible(config, stored["config"])
    restored = build_model(stored["config"]).eval()
    restored.load_state_dict(stored["model"])
    image = torch.rand(1, 3, 16, 24)
    state = torch.tensor([[0.3, 0.1]]) if task == "trajectory" else None
    with torch.inference_mode():
        assert torch.equal(original(image, state), restored(image, state))
    assert stored["resumable"] is False and stored["diagnostic"] is False
    assert "optimizer" not in stored and "torch_rng" not in stored and "training" not in stored["config"]
    assert "train_manifest" not in stored["config"]["data"]
    assert summary["output_bytes"] < summary["source_bytes"]
    assert stored["source_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    if task == "segmentation":
        runtime = CameraPerception({"segmentation_checkpoint": str(output), "drivable_class_ids": [0]}, "cpu")
        assert runtime.height == 16 and runtime.width == 24


@pytest.mark.parametrize("provenance", ["flag", "config", "max_train_steps", "max_val_steps", "partial_epoch"])
def test_export_never_removes_diagnostic_provenance(tmp_path, provenance):
    source, _, _ = training_checkpoint(tmp_path)
    checkpoint = torch.load(source, weights_only=True)
    if provenance == "flag":
        checkpoint["diagnostic"] = True
    elif provenance == "config":
        checkpoint["config"]["diagnostic_only"] = True
    elif provenance == "partial_epoch":
        checkpoint["epoch_complete"] = False
    else:
        checkpoint["config"]["training"][provenance] = 2
    torch.save(checkpoint, source)
    output = tmp_path / "export.pt"
    assert export_runtime_checkpoint(source, output)["diagnostic"] is True
    with pytest.raises(ValueError, match="Diagnostic"):
        CameraPerception({"segmentation_checkpoint": str(output), "drivable_class_ids": [0]}, "cpu")


def test_export_refuses_overwrite_and_invalid_weights(tmp_path):
    source, _, _ = training_checkpoint(tmp_path)
    with pytest.raises(FileExistsError):
        export_runtime_checkpoint(source, source)
    output = tmp_path / "existing.pt"
    output.write_bytes(b"preserve")
    with pytest.raises(FileExistsError):
        export_runtime_checkpoint(source, output)
    assert output.read_bytes() == b"preserve"
    stored = torch.load(source, weights_only=True)
    next(iter(stored["model"].values())).flatten()[0] = float("inf")
    torch.save(stored, source)
    bad_output = tmp_path / "bad.pt"
    with pytest.raises(ValueError, match="nonfinite model weights"):
        export_runtime_checkpoint(source, bad_output)
    assert not bad_output.exists()
