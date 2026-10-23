import numpy as np
from PIL import Image
import pytest
import torch
from torch.nn import functional as F

from camera_rc.driving.perception import CameraPerception
from camera_rc.models import build_model


@pytest.fixture(autouse=True)
def limit_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def checkpoint(tmp_path, diagnostic=False, task="segmentation"):
    config = {"task": task, "model": {"width": 4, "num_classes": 3}, "data": {"image_size": [16, 24]}}
    model = build_model(config)
    path = tmp_path / "model.pt"
    torch.save({"format_version": 1, "config": config, "model": model.state_dict(), "diagnostic": diagnostic}, path)
    return path, model


def test_native_inference_matches_training_rgb_preprocessing(tmp_path):
    path, model = checkpoint(tmp_path)
    perception = CameraPerception({"segmentation_checkpoint": str(path), "drivable_class_ids": [0, 1], "half": True, "min_probability": 0.01}, "cpu")
    image = np.random.default_rng(4).integers(0, 256, size=(47, 61, 3), dtype=np.uint8)
    result = perception.infer(image)
    resized = Image.fromarray(image).resize((24, 16), Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(resized, dtype=np.float32).transpose(2, 0, 1).copy()).unsqueeze(0) / 255
    with torch.inference_mode():
        expected = model.eval()(tensor).softmax(dim=1)[:, [0, 1]].sum(1, keepdim=True)
        expected = F.interpolate(expected, size=(47, 61), mode="bilinear", align_corners=False)[0, 0].numpy()
    assert result.drivable.shape == (47, 61) and result.drivable.dtype == np.float32
    assert np.allclose(result.drivable, expected, atol=1e-6)
    assert np.isfinite(result.drivable).all() and 0 <= result.drivable.min() <= result.drivable.max() <= 1
    assert result.inference_ms > 0 and result.detections == [] and perception.half is False


def test_diagnostic_checkpoint_requires_explicit_replay_opt_in(tmp_path):
    path, _ = checkpoint(tmp_path, diagnostic=True)
    options = {"segmentation_checkpoint": str(path), "drivable_class_ids": [0]}
    with pytest.raises(ValueError, match="Diagnostic"):
        CameraPerception(options, "cpu")
    assert CameraPerception({**options, "allow_diagnostic": True}, "cpu").diagnostic


def test_uncertain_pixels_become_unknown(tmp_path):
    path, _ = checkpoint(tmp_path)
    stored = torch.load(path, weights_only=True)
    for tensor in stored["model"].values():
        tensor.zero_()
    torch.save(stored, path)
    runtime = CameraPerception({"segmentation_checkpoint": str(path), "drivable_class_ids": [0], "min_probability": 0.6}, "cpu")
    result = runtime.infer(np.zeros((24, 32, 3), dtype=np.uint8))
    assert np.array_equal(result.drivable, np.zeros((24, 32)))


def test_checkpoint_type_class_semantics_and_local_paths_required(tmp_path):
    path, _ = checkpoint(tmp_path)
    for ids in (None, [], [3], [-1], [0, 0], [True]):
        with pytest.raises(ValueError, match="drivable_class_ids"):
            CameraPerception({"segmentation_checkpoint": str(path), "drivable_class_ids": ids}, "cpu")
    with pytest.raises(FileNotFoundError, match="never downloads"):
        CameraPerception({"segmentation_checkpoint": str(tmp_path / "missing.pt"), "drivable_class_ids": [0]}, "cpu")
    path, _ = checkpoint(tmp_path, task="depth")
    with pytest.raises(ValueError, match="segmentation"):
        CameraPerception({"segmentation_checkpoint": str(path), "drivable_class_ids": [0]}, "cpu")


def test_invalid_input_and_nonfinite_weights_rejected(tmp_path):
    path, model = checkpoint(tmp_path)
    runtime = CameraPerception({"segmentation_checkpoint": str(path), "drivable_class_ids": [0]}, "cpu")
    for image in (np.zeros((20, 30, 3)), np.zeros((20, 30), dtype=np.uint8), np.zeros((1, 30, 3), dtype=np.uint8)):
        with pytest.raises(ValueError, match="uint8 RGB"):
            runtime.infer(image)
    stored = torch.load(path, weights_only=True)
    next(iter(stored["model"].values())).flatten()[0] = float("nan")
    torch.save(stored, path)
    with pytest.raises(ValueError, match="nonfinite model weights"):
        CameraPerception({"segmentation_checkpoint": str(path), "drivable_class_ids": [0]}, "cpu")
