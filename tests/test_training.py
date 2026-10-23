from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from camera_rc.metrics import MetricAccumulator, task_loss
from camera_rc.models import build_model
from camera_rc import training


@pytest.fixture(autouse=True)
def limited_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class SyntheticDataset(Dataset):
    def __init__(self, task, count):
        self.task, self.count = task, count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        generator = torch.Generator().manual_seed(100 + index)
        image = torch.rand(3, 16, 20, generator=generator)
        if self.task == "segmentation":
            target = (image[0] * 3).long()
            target[0] = 255
        elif self.task == "depth":
            target = image[:1] + 1.0
            target[:, 0] = 0
        else:
            target = torch.tensor([[0.5, 0.1, 1.0], [1.0, 0.2, 1.0]])
        return {"image": image, "target": target, "state": torch.tensor([1.0, 0.0])}


def loaders(config):
    return (
        DataLoader(SyntheticDataset(config["task"], 3), batch_size=1, shuffle=True),
        DataLoader(SyntheticDataset(config["task"], 2), batch_size=1),
    )


def config_for(path: Path, task="segmentation", epochs=1):
    return {
        "task": task, "model": {"width": 4, "num_classes": 3, "horizon": 2},
        "data": {"image_size": [16, 20]},
        "training": {"epochs": epochs, "batch_size": 1, "grad_accum_steps": 2,
                     "device": "cpu", "seed": 7, "amp": True, "output_dir": str(path)},
    }


def test_depth_loss_and_metrics_ignore_missing_values():
    prediction = torch.tensor([[[[2.0, 100.0, 100.0]]]], requires_grad=True)
    target = torch.tensor([[[[1.0, 0.0, float("nan")]]]])
    loss = task_loss("depth", prediction, target)
    loss.backward()
    assert prediction.grad[0, 0, 0, 0] > 0
    assert torch.equal(prediction.grad[0, 0, 0, 1:], torch.zeros(2))
    metrics = MetricAccumulator("depth")
    metrics.update(prediction, target)
    assert metrics.compute() == {"rmse_m": 1.0, "abs_rel": 1.0, "valid_pixels": 1}


def test_segmentation_iou_and_ignore_index():
    prediction = torch.tensor([[[[4.0, 4.0, -1.0]], [[-1.0, -1.0, 4.0]]]])
    target = torch.tensor([[[0, 1, 255]]])
    metrics = MetricAccumulator("segmentation", 2)
    metrics.update(prediction, target)
    assert metrics.compute()["miou"] == pytest.approx(0.25)
    assert metrics.compute()["pixel_accuracy"] == pytest.approx(0.5)
    assert metrics.compute()["class_iou"] == [0.5, 0.0]
    assert metrics.compute()["confusion_matrix"] == [[1, 0], [1, 0]]
    with pytest.raises(ValueError, match="labels"):
        task_loss("segmentation", prediction, torch.tensor([[[0, 2, 255]]]))


def test_segmentation_absent_class_is_null_not_perfect_or_nan():
    metrics = MetricAccumulator("segmentation", 3)
    metrics.update(torch.tensor([[[[2.0]], [[0.0]], [[0.0]]]]), torch.tensor([[[0]]]))
    result = metrics.compute()
    assert result["class_iou"] == [1.0, None, None]
    assert result["miou"] == 1.0
    assert result["confusion_matrix"] == [[1, 0, 0], [0, 0, 0], [0, 0, 0]]


def test_trajectory_metric_units():
    target = torch.zeros(1, 2, 3)
    prediction = torch.tensor([[[3.0, 4.0, 2.0], [0.0, 2.0, 4.0]]])
    metrics = MetricAccumulator("trajectory")
    metrics.update(prediction, target)
    assert metrics.compute() == {"ade_m": 3.5, "fde_m": 2.0, "speed_mae_mps": 3.0, "samples": 1}


@pytest.mark.parametrize("task", ["segmentation", "depth", "trajectory"])
def test_train_and_evaluate_all_tasks(tmp_path, monkeypatch, task):
    monkeypatch.setattr(training, "build_dataloaders", loaders)
    config = config_for(tmp_path / task, task)
    result = training.run_training(config)
    assert result["training_complete"]
    assert result["run_minibatches"] == 3
    # Three batches with accumulation=2 must perform the final partial update.
    assert result["final"]["optimizer_steps"] == 2
    metrics = training.evaluate_checkpoint(config, Path(result["last_checkpoint"]))
    assert metrics["batches"] == 2
    assert metrics["loss"] >= 0
    checkpoint = torch.load(result["last_checkpoint"], weights_only=True)
    assert checkpoint["epoch_complete"]
    assert checkpoint["scaler"] == {}  # CPU has no enabled CUDA AMP scaler.


def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", loaders)
    complete_config = config_for(tmp_path / "complete", epochs=2)
    complete = training.run_training(complete_config)
    resume_config = config_for(tmp_path / "resumed", epochs=1)
    first = training.run_training(resume_config)
    resume_config["training"]["epochs"] = 2
    second = training.run_training(resume_config, resume=Path(first["last_checkpoint"]))
    expected = torch.load(complete["last_checkpoint"], weights_only=True)["model"]
    actual = torch.load(second["last_checkpoint"], weights_only=True)["model"]
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
    assert second["completed_epochs"] == 2
    assert second["final"]["global_step"] == 6


def test_partial_accumulation_group_has_full_weight(tmp_path, monkeypatch):
    class ConstantDepth(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.depth = torch.nn.Parameter(torch.tensor(0.5))

        def forward(self, image, state=None):
            return self.depth.expand(image.shape[0], 1, *image.shape[2:])

    def ordered_loaders(config):
        return (DataLoader(SyntheticDataset("depth", 3), batch_size=1),
                DataLoader(SyntheticDataset("depth", 2), batch_size=1))

    monkeypatch.setattr(training, "build_dataloaders", ordered_loaders)
    monkeypatch.setattr(training, "build_model", lambda config: ConstantDepth())
    config = config_for(tmp_path / "accumulation", task="depth")
    config["training"]["lr"] = 0.05
    result = training.run_training(config)
    actual = torch.load(result["last_checkpoint"], weights_only=True)["model"]["depth"]
    reference = ConstantDepth()
    optimizer = torch.optim.AdamW(reference.parameters(), lr=0.05)
    dataset = SyntheticDataset("depth", 3)
    for indices in ([0, 1], [2]):
        optimizer.zero_grad()
        images = torch.stack([dataset[i]["image"] for i in indices])
        targets = torch.stack([dataset[i]["target"] for i in indices])
        task_loss("depth", reference(images), targets).backward()
        torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.0)
        optimizer.step()
    torch.testing.assert_close(actual, reference.depth.detach(), rtol=1e-6, atol=1e-7)


def test_diagnostic_checkpoint_not_claimed_as_complete(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", loaders)
    config = config_for(tmp_path / "diagnostic", epochs=2)
    result = training.run_training(config, max_steps=1)
    assert not result["training_complete"]
    assert result["completed_epochs"] == 0
    assert result["final"]["diagnostic"]
    assert result["final"]["validation"]["partial_validation"]
    with pytest.raises(ValueError, match="mid-epoch"):
        training.run_training(config, resume=Path(result["last_checkpoint"]))


def test_configuration_and_model_validation(tmp_path):
    config = config_for(tmp_path)
    config["training"]["grad_accum_steps"] = 0
    with pytest.raises(ValueError, match="grad_accum_steps"):
        training.validate_config(config)
    with pytest.raises(ValueError, match="task"):
        build_model({"task": "unknown"})
    with pytest.raises(ValueError, match="No valid"):
        MetricAccumulator("depth").compute()


@pytest.mark.parametrize("key,value", [
    ("image_size", [32, 40]), ("depth_scale", 1000.0),
    ("max_depth_m", 50.0), ("future_offsets_s", [0.5, 1.0]),
])
def test_checkpoint_rejects_changed_data_semantics(tmp_path, key, value):
    previous = config_for(tmp_path)
    current = config_for(tmp_path)
    current["data"][key] = value
    with pytest.raises(ValueError, match=key):
        training._check_compatible(current, previous)


def test_checkpoint_accepts_equivalent_preprocessing_defaults(tmp_path):
    previous = config_for(tmp_path)
    current = config_for(tmp_path)
    previous["data"].pop("image_size")
    current["data"].update(image_size=[256, 512], depth_scale=256.0, max_depth_m=80.0)
    training._check_compatible(current, previous)


def test_resume_rejects_different_output_directory(tmp_path):
    config = config_for(tmp_path / "destination")
    with pytest.raises(ValueError, match="checkpoint directory"):
        training.run_training(config, resume=tmp_path / "source" / "last.pt")
    assert not (tmp_path / "destination").exists()


def test_bad_forecast_intervals_are_rejected(tmp_path):
    config = config_for(tmp_path, task="trajectory")
    config["data"]["future_offsets_s"] = [1.0, 0.5]
    with pytest.raises(ValueError, match="strictly increasing"):
        training.validate_config(config)


def test_amp_overflow_skips_optimizer_and_reduces_scale_then_recovers():
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", init_scale=128.0)
    before = model.weight.detach().clone()
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    model.weight.grad.fill_(float("inf"))
    assert not training._apply_optimizer_step(model, optimizer, scaler)
    torch.testing.assert_close(model.weight.detach(), before, rtol=0, atol=0)
    assert optimizer.state == {}
    assert scaler.get_scale() == 64.0
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(model(torch.ones(1, 1)).sum()).backward()
    assert training._apply_optimizer_step(model, optimizer, scaler)
    assert not torch.equal(model.weight.detach(), before)
    assert optimizer.state[model.weight]["step"].item() == 1


@pytest.mark.parametrize("nonfinite", [float("inf"), float("nan")])
def test_nonfinite_gradients_without_amp_still_fail(nonfinite):
    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    model(torch.ones(1, 1)).sum().backward()
    model.weight.grad.fill_(nonfinite)
    with pytest.raises(FloatingPointError, match="without AMP"):
        training._apply_optimizer_step(model, optimizer, scaler)
    assert optimizer.state == {}


def _cpu_amp_with_injected_overflows(monkeypatch, overflow_calls):
    real_scaler = torch.amp.GradScaler
    real_step = training._apply_optimizer_step
    calls = []

    def cpu_scaler(device, *, enabled, **kwargs):
        return real_scaler("cpu", enabled=True, **kwargs)

    def injecting_step(model, optimizer, scaler):
        calls.append(len(calls) + 1)
        if calls[-1] in overflow_calls:
            next(parameter for parameter in model.parameters() if parameter.grad is not None).grad.fill_(float("inf"))
        return real_step(model, optimizer, scaler)

    monkeypatch.setattr(training.torch.amp, "GradScaler", cpu_scaler)
    monkeypatch.setattr(training, "_apply_optimizer_step", injecting_step)
    return calls


def test_amp_skips_are_recorded_separately_from_actual_updates(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", loaders)
    calls = _cpu_amp_with_injected_overflows(monkeypatch, {1})
    config = config_for(tmp_path / "amp-recovery")
    config["training"]["amp_init_scale"] = 128.0
    result = training.run_training(config)
    assert calls == [1, 2]
    assert result["training_complete"]
    assert result["final"]["optimizer_steps"] == 1
    assert result["final"]["epoch_optimizer_steps"] == 1
    assert result["final"]["amp_skipped_steps"] == 1
    assert result["final"]["amp_scale"] == 64.0
    checkpoint = torch.load(result["last_checkpoint"], weights_only=True)
    assert checkpoint["optimizer_steps"] == 1 and checkpoint["amp_skipped_steps"] == 1
    assert checkpoint["consecutive_amp_skips"] == 0
    assert all(state["step"].item() == 1 for state in checkpoint["optimizer"]["state"].values())


def test_all_skipped_epoch_cannot_write_completed_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", loaders)
    _cpu_amp_with_injected_overflows(monkeypatch, {1, 2})
    output = tmp_path / "all-skipped"
    with pytest.raises(FloatingPointError, match="No optimizer updates.*amp_init_scale"):
        training.run_training(config_for(output))
    assert not (output / "best.pt").exists()
    assert not (output / "last.pt").exists()
    assert not (output / "summary.json").exists()


def test_repeated_amp_overflow_stops_before_unbounded_silent_skips(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", loaders)
    calls = _cpu_amp_with_injected_overflows(monkeypatch, {1, 2, 3})
    config = config_for(tmp_path / "persistent-overflow")
    config["training"].update(grad_accum_steps=1, max_consecutive_amp_skips=2)
    with pytest.raises(FloatingPointError, match="Repeated non-finite AMP gradients"):
        training.run_training(config)
    assert calls == [1, 2]


def test_nonfinite_training_loss_is_not_treated_as_amp_overflow(tmp_path, monkeypatch):
    monkeypatch.setattr(training, "build_dataloaders", loaders)
    monkeypatch.setattr(training, "task_loss", lambda *args: args[1].sum() * float("nan"))
    with pytest.raises(FloatingPointError, match="Non-finite training loss"):
        training.run_training(config_for(tmp_path / "bad-objective"))


@pytest.mark.parametrize("key,value", [
    ("amp_init_scale", 0), ("amp_init_scale", float("nan")), ("amp_init_scale", True),
    ("max_consecutive_amp_skips", 0), ("max_consecutive_amp_skips", 1.5),
])
def test_amp_numerical_configuration_is_validated(tmp_path, key, value):
    config = config_for(tmp_path)
    config["training"][key] = value
    with pytest.raises(ValueError, match=key):
        training.validate_config(config)
