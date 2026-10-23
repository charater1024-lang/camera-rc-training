import copy
import json
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from smoke_joint_policy import create_fixture
from camera_rc.joint_policy.training import train_joint_policy, load_joint_config, validate_config
import camera_rc.joint_policy.training as training_module


def fixture_config(tmp_path):
    manifests = create_fixture(tmp_path / "raw", views=2, train_samples=2, val_samples=1)
    return {"task": "joint_policy", "diagnostic_only": False,
            "model": {"width": 4, "embed_dim": 16, "heads": 4, "token_grid": [2, 2],
                      "horizon": 5, "encoder_chunk_size": 1},
            "data": {"train_manifest": manifests["train"], "val_manifest": manifests["val"],
                     "expected_views": 2, "expected_history": 3, "image_size": [32, 48],
                     "future_dt_s": [.5, 1., 1.5, 2., 2.5]},
            "training": {"output_dir": str(tmp_path / "trained"), "device": "cpu",
                         "epochs": 1, "batch_size": 1, "num_workers": 0, "amp": False, "lr": .001}}


def test_training_updates_joint_model_and_preserves_diagnostic_provenance(tmp_path):
    config = fixture_config(tmp_path)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = train_joint_policy(config)
    finally:
        torch.set_num_threads(previous)
    assert result["optimizer_steps"] == 2
    assert result["changed_parameter_tensors"] > 0
    assert result["diagnostic"] is True  # Input provenance cannot be cleared by a config flag.
    assert result["models_validated_for_control"] is False
    assert result["test_split_evaluated"] is False
    assert result["validation_after"]["model"]["samples"] == 1
    assert result["validation_after"]["constant_velocity_baseline"]["ade_m"] == pytest.approx(0.)
    checkpoint = torch.load(result["best_checkpoint"], map_location="cpu", weights_only=True)
    assert checkpoint["model_type"] == "joint_camera_temporal_waypoints"
    assert checkpoint["diagnostic"] is True
    assert checkpoint["training_complete"] is True
    assert not (Path(result["best_checkpoint"]).parent / "INCOMPLETE").exists()


def test_train_val_leakage_rejected_before_output_creation(tmp_path):
    config = fixture_config(tmp_path)
    path = Path(config["data"]["val_manifest"])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:
        row["origin_id"] = "synthetic_joint_fixture:drive_train"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="leakage"):
        train_joint_policy(config)
    assert not Path(config["training"]["output_dir"]).exists()


@pytest.mark.parametrize("key,value", [("epochs", True), ("batch_size", 0), ("lr", float("nan")),
                                      ("grad_accum_steps", -1), ("amp", "false"), ("device", "gpu")])
def test_invalid_training_options_rejected(tmp_path, key, value):
    config = fixture_config(tmp_path)
    config["training"][key] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_relative_paths_resolve_against_config_directory(tmp_path):
    source = tmp_path / "nested"
    source.mkdir()
    path = source / "config.json"
    path.write_text(json.dumps({"data": {"train_manifest": "../train.jsonl", "test_manifest": "test.jsonl"},
                               "training": {"output_dir": "out", "init_encoder": "../encoder.pt"}}))
    config = load_joint_config(path)
    assert config["data"]["train_manifest"] == str(tmp_path / "train.jsonl")
    assert config["training"]["init_encoder"] == str(tmp_path / "encoder.pt")


def test_manifest_changed_between_validation_and_loading_is_rejected(tmp_path, monkeypatch):
    config = fixture_config(tmp_path)
    original = training_module.make_dataset
    def mutate_after_load(cfg, split):
        dataset = original(cfg, split)
        if split == "train":
            path = Path(cfg["data"]["train_manifest"])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[0]["condition"] = "changed_after_split_validation"
            path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        return dataset
    monkeypatch.setattr(training_module, "make_dataset", mutate_after_load)
    with pytest.raises(ValueError, match="Manifest changed"):
        train_joint_policy(config)
    assert not Path(config["training"]["output_dir"]).exists()


def test_incomplete_data_preparation_is_rejected(tmp_path):
    config = fixture_config(tmp_path)
    (tmp_path / "raw/.incomplete").write_text("incomplete")
    with pytest.raises(ValueError, match="incomplete preparation"):
        train_joint_policy(config)
