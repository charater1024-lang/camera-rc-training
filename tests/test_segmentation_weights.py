from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from camera_rc import training
from camera_rc.metrics import task_loss


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _prediction():
    # The second pixel is confidently mistaken for road (class 0).
    return torch.tensor([[[[4.0, 4.0, 4.0]], [[-1.0, -1.0, -1.0]]]], requires_grad=True)


def test_nonroad_weight_increases_false_road_gradient_and_ignores_unlabelled():
    target = torch.tensor([[[0, 1, 255]]])
    plain, weighted = _prediction(), _prediction()
    plain_loss = task_loss("segmentation", plain, target)
    weighted_loss = task_loss("segmentation", weighted, target, torch.tensor([1.0, 2.0]))
    plain_loss.backward()
    weighted_loss.backward()
    assert weighted_loss > plain_loss
    # Weighted-mean normalization makes the factor 4/3, not 2, for this batch.
    assert weighted.grad[0, 0, 0, 1] == pytest.approx(plain.grad[0, 0, 0, 1].item() * 4 / 3)
    assert weighted.grad[0, 1, 0, 1] < plain.grad[0, 1, 0, 1] < 0
    assert torch.equal(weighted.grad[..., 2], torch.zeros_like(weighted.grad[..., 2]))


def test_none_and_equal_weights_preserve_default_loss_and_gradients():
    target = torch.tensor([[[0, 1, 255]]])
    predictions = [_prediction() for _ in range(3)]
    losses = [task_loss("segmentation", predictions[0], target),
              task_loss("segmentation", predictions[1], target, None),
              task_loss("segmentation", predictions[2], target, torch.ones(2))]
    for loss in losses:
        loss.backward()
        torch.testing.assert_close(loss, losses[0], rtol=0, atol=0)
    for prediction in predictions:
        torch.testing.assert_close(prediction.grad, predictions[0].grad, rtol=0, atol=0)


def test_weighted_all_ignore_has_zero_finite_loss_and_gradient():
    prediction = _prediction()
    loss = task_loss("segmentation", prediction, torch.full((1, 1, 3), 255), torch.tensor([1.0, 2.0]))
    loss.backward()
    assert loss.item() == 0.0
    assert torch.equal(prediction.grad, torch.zeros_like(prediction))


@pytest.mark.parametrize("weight", [
    torch.tensor([1.0]), torch.ones(1, 2), torch.tensor([1.0, 0.0]),
    torch.tensor([1.0, -1.0]), torch.tensor([1.0, float("nan")]),
    torch.tensor([1.0, float("inf")]), torch.tensor([1.0, 1e300], dtype=torch.float64),
])
def test_invalid_loss_weights_are_rejected(weight):
    with pytest.raises(ValueError, match="segmentation_class_weights"):
        task_loss("segmentation", _prediction(), torch.tensor([[[0, 1, 255]]]), weight)


def _config(output, weights=None):
    return {
        "task": "segmentation", "model": {"width": 4, "num_classes": 2},
        "data": {"image_size": [16, 20], "label_space_id": "road_surface_v1"},
        "training": {"epochs": 1, "batch_size": 1, "device": "cpu", "seed": 7,
                     "amp": False, "output_dir": str(output),
                     "segmentation_class_weights": weights},
    }


@pytest.mark.parametrize("weights", [[], [1], [1, 2, 3], "1,2", {"0": 1, "1": 2},
                                      [True, 2], [1, "2"], [1, 0], [1, -2],
                                      [1, float("nan")], [1, float("inf")]])
def test_invalid_configuration_weights_are_rejected(tmp_path, weights):
    with pytest.raises(ValueError, match="segmentation_class_weights"):
        training.validate_config(_config(tmp_path, weights))


@pytest.mark.parametrize("weights", [None, [1, 2], (1.0, 2.0)])
def test_valid_configuration_weights(tmp_path, weights):
    training.validate_config(_config(tmp_path, weights))


@pytest.mark.parametrize("task", ["depth", "trajectory"])
def test_weights_cannot_silently_affect_other_tasks(tmp_path, task):
    config = _config(tmp_path, [1, 2])
    config["task"] = task
    with pytest.raises(ValueError, match="require task=segmentation"):
        training.validate_config(config)
    with pytest.raises(ValueError, match="require task=segmentation"):
        task_loss(task, torch.ones(1, 1, 3), torch.ones(1, 1, 3), torch.ones(2))


def _loaders(config):
    generator = torch.Generator().manual_seed(120)
    examples = []
    for _ in range(3):
        image = torch.rand(3, 16, 20, generator=generator)
        target = (image[0] > 0.5).long()
        target[0] = 255
        examples.append({"image": image, "target": target})
    return DataLoader(examples, batch_size=1), DataLoader(examples[:2], batch_size=1)


def test_training_weights_are_persisted_but_validation_and_selection_are_unchanged(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", _loaders)
    calls = []

    def observed_loss(task, prediction, target, weights=None):
        calls.append(None if weights is None else weights.tolist())
        return task_loss(task, prediction, target, weights)

    monkeypatch.setattr(training, "task_loss", observed_loss)
    config = _config(tmp_path / "weighted", [1, 2])
    result = training.run_training(config)
    assert calls == [[1, 2], [1, 2], [1, 2], None, None]
    assert result["training_complete"] and not result["diagnostic"]
    checkpoint = torch.load(result["last_checkpoint"], weights_only=True)
    assert checkpoint["config"]["training"]["segmentation_class_weights"] == [1, 2]
    assert checkpoint["selection_metric"] == "miou"
    unweighted_config = deepcopy(config)
    unweighted_config["training"].pop("segmentation_class_weights")
    weighted_eval = training.evaluate_checkpoint(config, Path(result["last_checkpoint"]))
    plain_eval = training.evaluate_checkpoint(unweighted_config, Path(result["last_checkpoint"]))
    assert weighted_eval == plain_eval == result["final"]["validation"]


def test_resume_rejects_weight_changes_but_init_from_starts_new_objective(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", _loaders)
    config = _config(tmp_path / "first")
    original = training.run_training(config)
    checkpoint = Path(original["last_checkpoint"])
    before = checkpoint.read_bytes()
    config["training"].update(epochs=2, segmentation_class_weights=[1, 2])
    with pytest.raises(ValueError, match="Resume cannot change.*init_from"):
        training.run_training(config, resume=checkpoint)
    assert checkpoint.read_bytes() == before
    config["training"].update(epochs=1, output_dir=str(tmp_path / "weighted-stage"))
    stage = training.run_training(config, init_from=checkpoint)
    assert stage["training_complete"]
    assert stage["initialization"]["path"] == str(checkpoint.resolve())
    assert stage["final"]["global_step"] == 3


def test_resume_with_same_weights_matches_uninterrupted_training(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", _loaders)
    complete_config = _config(tmp_path / "complete", [1, 2])
    complete_config["training"]["epochs"] = 2
    complete = training.run_training(complete_config)
    resumed_config = _config(tmp_path / "resumed", [1, 2])
    first = training.run_training(resumed_config)
    resumed_config["training"]["epochs"] = 2
    resumed = training.run_training(resumed_config, resume=Path(first["last_checkpoint"]))
    expected = torch.load(complete["last_checkpoint"], weights_only=True)["model"]
    actual = torch.load(resumed["last_checkpoint"], weights_only=True)["model"]
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
