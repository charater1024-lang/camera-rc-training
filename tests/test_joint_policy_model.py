"""Geometry/causality/gradient contracts for the offline joint policy baseline."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import torch

from camera_rc.joint_policy import build_joint_policy, init_encoder, parameter_count
from camera_rc.models import TinyUNet


@pytest.fixture(autouse=True)
def small_cpu_threads():
    original = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(original)


def config():
    return {"model": {"width": 4, "embed_dim": 32, "token_grid": [3, 5], "horizon": 3,
                       "encoder_chunk_size": 1},
            "data": {"expected_views": 2, "expected_history": 2, "image_size": [32, 48],
                     "future_dt_s": [.5, 1., 1.5]}}


def inputs(views=2, history=2):
    generator = torch.Generator().manual_seed(5)
    images = torch.rand((1, history, views, 3, 32, 48), generator=generator)
    k = torch.tensor([[40., 0, 23.5], [0, 40., 15.5], [0, 0, 1.]])
    k = k.expand(1, history, views, 3, 3).clone()
    transforms = torch.eye(4).expand(1, history, views, 4, 4).clone()
    transforms[..., :3, :3] = torch.tensor([[0., 0, 1], [-1, 0, 0], [0, -1, 0]])
    for view in range(views):
        transforms[:, :, view, 1, 3] = .2 * view
    times = torch.linspace(-.5, 0, history).unsqueeze(0)
    return images, k, transforms, times, torch.tensor([[.3, .01]])


def test_real_gradients_from_every_view_and_time_reach_shared_encoder():
    torch.manual_seed(8)
    model = build_joint_policy(config()).train()
    args = list(inputs())
    args[0].requires_grad_()
    output = model(*args)
    assert output.shape == (1, 3, 3) and (output[..., 2] >= 0).all()
    output.square().sum().backward()
    assert (args[0].grad.abs().sum(dim=(-3, -2, -1)) > 0).all()
    assert model.enc1[0].weight.grad.abs().sum() > 0
    assert model.geometry_embedding[0].weight.grad.abs().sum() > 0
    assert model.state_embedding[0].weight.grad.abs().sum() > 0
    assert parameter_count(model) == sum(p.numel() for p in model.parameters())


def test_camera_geometry_and_relative_age_change_joint_predictions():
    torch.manual_seed(9)
    model = build_joint_policy(config()).eval()
    original = inputs()
    with torch.no_grad():
        baseline = model(*original)
        moved = [value.clone() for value in original]
        moved[2][0, 0, 1, 1, 3] += 2
        different_geometry = model(*moved)
        changed_time = [value.clone() for value in original]
        changed_time[3][0, 0] = -.8
        different_time = model(*changed_time)
    assert (baseline - different_geometry).abs().max() > 1e-6
    assert (baseline - different_time).abs().max() > 1e-6


def test_camera_permutation_with_geometry_preserves_output():
    torch.manual_seed(10)
    model = build_joint_policy(config()).eval()
    original = inputs()
    swapped = (original[0][:, :, [1, 0]], original[1][:, :, [1, 0]], original[2][:, :, [1, 0]],
               original[3], original[4])
    with torch.no_grad():
        assert torch.allclose(model(*original), model(*swapped), atol=2e-6, rtol=1e-5)


def test_optical_center_ray_transforms_to_anchor_forward_and_keeps_metric_origin():
    model = build_joint_policy(config())
    images, k, transforms, times, state = inputs()
    transforms[0, 0, 0, :3, 3] = torch.tensor([.5, .1, .3])
    token = model.geometry_tokens(k, transforms, times, 32, 48)[0, 0, 0, 7]
    assert torch.allclose(token, torch.tensor([1., 0, 0, .5, .1, .3, -.5]), atol=1e-6)


@pytest.mark.parametrize("problem", ["future", "stale", "duplicate_time", "not_current", "nan_image", "inf_state",
                                   "negative_speed", "bad_pixels", "zero_focal", "bad_intrinsics_row", "singular_rotation",
                                   "reflection", "bad_homogeneous_row", "missing_view", "wrong_time_shape", "wrong_calibration_shape"])
def test_invalid_current_observations_are_rejected(problem):
    model = build_joint_policy(config())
    args = list(inputs())
    if problem == "future": args[3][0, 0] = .1
    elif problem == "stale": args[3][0, 0] = -1.1
    elif problem == "duplicate_time": args[3][0, 0] = 0
    elif problem == "not_current": args[3][0, -1] = -.1
    elif problem == "nan_image": args[0][0, 0, 0, 0, 0, 0] = float("nan")
    elif problem == "inf_state": args[4][0, 1] = float("inf")
    elif problem == "negative_speed": args[4][0, 0] = -.1
    elif problem == "bad_pixels": args[0][0, 0, 0, 0, 0, 0] = 2
    elif problem == "zero_focal": args[1][0, 0, 0, 0, 0] = 0
    elif problem == "bad_intrinsics_row": args[1][0, 0, 0, 2, 0] = .1
    elif problem == "singular_rotation": args[2][0, 0, 0, 0, 2] = 2
    elif problem == "reflection": args[2][0, 0, 0, 0, 2] = -1
    elif problem == "bad_homogeneous_row": args[2][0, 0, 0, 3, 0] = 1
    elif problem == "missing_view": args[:3] = [value[:, :, :1] for value in args[:3]]
    elif problem == "wrong_time_shape": args[3] = args[3].reshape(2, 1)
    elif problem == "wrong_calibration_shape": args[1] = args[1][:, :1]
    with pytest.raises(ValueError):
        model(*args)


def test_checkpointed_training_matches_regular_forward_and_gradients():
    torch.manual_seed(11)
    normal = build_joint_policy(config()).train()
    cfg = config()
    cfg["model"]["checkpoint_encoder"] = True
    checkpointed = build_joint_policy(cfg).train()
    checkpointed.load_state_dict(normal.state_dict())
    left, right = list(inputs()), list(inputs())
    left[0].requires_grad_()
    right[0].requires_grad_()
    a, b = normal(*left), checkpointed(*right)
    assert torch.allclose(a, b)
    a.square().sum().backward()
    b.square().sum().backward()
    assert torch.allclose(left[0].grad, right[0].grad, atol=1e-7)
    assert torch.allclose(normal.enc1[0].weight.grad, checkpointed.enc1[0].weight.grad, atol=1e-6)


def test_chunked_shared_encoder_matches_larger_chunk_and_mixed_precision():
    torch.manual_seed(12)
    one = build_joint_policy(config()).eval()
    cfg = config()
    cfg["model"]["encoder_chunk_size"] = 4
    four = build_joint_policy(cfg).eval()
    four.load_state_dict(one.state_dict())
    with torch.no_grad():
        assert torch.allclose(one(*inputs()), four(*inputs()), atol=2e-6, rtol=1e-5)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            mixed = one(*inputs())
    assert torch.isfinite(mixed).all()


def test_model_accepts_eight_views_and_four_past_times_but_never_missing_configured_views():
    cfg = config()
    cfg["data"].update(expected_views=8, expected_history=4)
    model = build_joint_policy(cfg).eval()
    with torch.no_grad():
        assert model(*inputs(8, 4)).shape == (1, 3, 3)


def native_checkpoint(path, diagnostic=False):
    native = TinyUNet(2, 4)
    torch.save({"format_version": 1, "model": native.state_dict(), "diagnostic": diagnostic,
                "epoch_complete": True, "config": {"task": "segmentation", "model": {"width": 4}}}, path)
    return native


def test_native_encoder_transfer_is_exact_and_leaves_joint_attention_new(tmp_path):
    path = tmp_path / "native.pt"
    native = native_checkpoint(path)
    model = build_joint_policy(config())
    joint_before = model.queries.detach().clone()
    report = init_encoder(model, path)
    assert report["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert report["matched_tensors"] > 0 and not report["source_diagnostic"]
    assert torch.equal(model.queries, joint_before)
    for name in report["matched_names"]:
        assert torch.equal(model.state_dict()[name], native.state_dict()[name])
        assert name.startswith(("enc1.", "enc2.", "enc3."))


def test_diagnostic_transfer_is_explicit_and_missing_tensor_does_not_partially_mutate(tmp_path):
    path = tmp_path / "diagnostic.pt"
    native_checkpoint(path, diagnostic=True)
    model = build_joint_policy(config())
    with pytest.raises(ValueError, match="Diagnostic"):
        init_encoder(model, path)
    assert init_encoder(model, path, allow_diagnostic=True)["source_diagnostic"]
    stored = torch.load(path, weights_only=True)
    stored["diagnostic"] = False
    del stored["model"]["enc3.4.bias"]
    torch.save(stored, path)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    with pytest.raises(ValueError, match="encoder tensor"):
        init_encoder(model, path)
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


@pytest.mark.parametrize("dt", [[1, .5, 1.5], [.5, 1], [.5, 1, float("nan")], [0, .5, 1.]])
def test_future_query_timing_is_bound_and_checked(dt):
    cfg = config()
    cfg["data"]["future_dt_s"] = dt
    with pytest.raises(ValueError, match="future_dt_s"):
        build_joint_policy(cfg)


def test_camera_names_and_configured_counts_cannot_disagree():
    cfg = config()
    cfg["data"]["camera_names"] = ["front"]
    with pytest.raises(ValueError, match="disagree"):
        build_joint_policy(cfg)


def benchmark_module():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_joint_policy.py"
    specification = importlib.util.spec_from_file_location("benchmark_joint_policy_test", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_synthetic_benchmark_can_measure_more_views_without_claiming_training(tmp_path):
    cfg = config()
    cfg["runtime"] = {"gpu_budget_gib": 8, "torch_allocator_cap_gib": 7}
    cfg["data"]["camera_names"] = ["camera_0", "camera_1"]
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    model = build_joint_policy(cfg)
    checkpoint = tmp_path / "joint.pt"
    torch.save({"format_version": 1, "model_type": "joint_camera_temporal_waypoints",
                "config": cfg, "model": model.state_dict(), "diagnostic": True}, checkpoint)
    result = benchmark_module().benchmark(path, tmp_path / "benchmark.json", checkpoint=checkpoint,
                                           device="cpu", views=8, history=2, frames=1)
    assert result["input_shape"] == [1, 2, 8, 3, 32, 48]
    assert result["output_shape"] == [1, 3, 3]
    assert result["shared_image_encoder_instances"] == 1
    assert result["provenance"]["checkpoint_expected_views"] == 2
    assert result["synthetic_overrides"]["expected_views"] == 8
    assert result["diagnostic_only"] and not result["trained_camera_layout_claim"]
    assert not result["physical_output"] and result["actual_gpu_name"] is None
    assert result["latency"]["frames"] == 1


@pytest.mark.parametrize("mismatch", ["future_times", "image_size", "stored_buffer"])
def test_benchmark_rejects_changed_checkpoint_semantics(tmp_path, mismatch):
    cfg = config()
    cfg["runtime"] = {"gpu_budget_gib": 8}
    model = build_joint_policy(cfg)
    stored_config = copy.deepcopy(cfg)
    state = model.state_dict()
    if mismatch == "future_times":
        stored_config["data"]["future_dt_s"] = [.6, 1.2, 1.8]
    elif mismatch == "image_size":
        stored_config["data"]["image_size"] = [64, 96]
    else:
        state["future_dt_s"] = torch.tensor([.6, 1.2, 1.8])
    checkpoint = tmp_path / "joint.pt"
    torch.save({"format_version": 1, "model_type": "joint_camera_temporal_waypoints",
                "config": stored_config, "model": state, "diagnostic": True}, checkpoint)
    path = tmp_path / "config.json"
    path.write_text(json.dumps(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="query times"):
        benchmark_module().benchmark(path, tmp_path / "benchmark.json", checkpoint=checkpoint,
                                     device="cpu", views=8, history=2, frames=1)
    assert not (tmp_path / "benchmark.json").exists()
