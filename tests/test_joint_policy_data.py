import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from camera_rc.joint_policy.data import JointSequenceDataset, resize_intrinsic, validate_split_manifests


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
spec = importlib.util.spec_from_file_location("prepare_vkitti_joint_policy", PROJECT / "scripts/prepare_vkitti_joint_policy.py")
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)
sys.path.pop(0)


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def sample(root, split="train", origin=None, diagnostic=True):
    root.mkdir(parents=True, exist_ok=True)
    frames = []
    for t in range(3):
        views = []
        for camera in range(2):
            path = root / f"{split}_{t}_{camera}.png"
            Image.new("RGB", (32, 24), (25 + t, 80 + camera, 110)).save(path)
            views.append({"camera_id": camera, "image": str(path), "image_size": [24, 32],
                          "intrinsic": [[20, .5, 15.5], [0, 22, 11.5], [0, 0, 1]],
                          "camera_to_anchor": adapter.OPTICAL_TO_ANCHOR.tolist()})
        frames.append({"timestamp_s": t * .1, "views": views})
    return {"schema_version": 1, "dataset_id": "fixture", "group_id": f"group_{split}",
            "origin_id": origin or f"origin_{split}", "split": split, "anchor_timestamp_s": .2,
            "frames": frames, "state": [2, .1], "future_dt_s": [.5, 1, 1.5, 2, 2.5],
            "targets": [[dt * 2, 0, 2] for dt in [.5, 1, 1.5, 2, 2.5]],
            "diagnostic": diagnostic, "partial_dataset": False}


def test_joint_tensor_shapes_calibration_time_and_diagnostic_provenance(tmp_path):
    row = sample(tmp_path / "images")
    path = write(tmp_path / "train.jsonl", [row])
    dataset = JointSequenceDataset(path, image_size=[48, 64])
    result = dataset[0]
    assert result["images"].shape == (3, 2, 3, 48, 64)
    assert result["intrinsics"].shape == (3, 2, 3, 3)
    assert result["camera_to_anchor"].shape == (3, 2, 4, 4)
    assert result["targets"].shape == (5, 3)
    assert torch.allclose(result["time_offsets"], torch.tensor([-.2, -.1, 0]))
    assert torch.allclose(result["intrinsics"][0, 0], torch.tensor([[40, 1, 31.5], [0, 44, 23.5], [0, 0, 1.]]))
    assert dataset.diagnostic and result["diagnostic"].item()
    assert dataset.rows[0]["origin_id"] == "origin_train"
    assert dataset.provenance["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert dataset.provenance["unique_image_files"] == 6
    assert all(isinstance(value, torch.Tensor) for value in result.values())


def test_intrinsic_nonuniform_resize_preserves_pixel_center_convention():
    result = resize_intrinsic([[100, 4, 10], [0, 120, 20], [0, 0, 1]], [100, 200], [50, 50])
    assert np.allclose(result, [[25, 1, 2.125], [0, 60, 9.75], [0, 0, 1]])


@pytest.mark.parametrize("field,value", [("state", [float("nan"), 0]), ("state", [-1, 0]),
                                        ("targets", [[1, 0, -1]] * 5), ("future_dt_s", [.5, 1, 1.5, 2, 3]),
                                        ("diagnostic", "false"), ("schema_version", True)])
def test_invalid_model_supervision_is_rejected(tmp_path, field, value):
    row = sample(tmp_path / "images")
    row[field] = value
    with pytest.raises(ValueError):
        JointSequenceDataset(write(tmp_path / "train.jsonl", [row]))


def test_stationary_zero_trajectory_is_valid_not_confused_with_empty_image(tmp_path):
    row = sample(tmp_path / "images")
    row.update(state=[0, 0], targets=[[0, 0, 0]] * 5)
    dataset = JointSequenceDataset(write(tmp_path / "train.jsonl", [row]))
    assert torch.count_nonzero(dataset[0]["targets"]) == 0


@pytest.mark.parametrize("mode", ["future", "out_of_order", "anchor_mismatch", "repeated_image", "swapped_cameras", "duplicate_camera"])
def test_future_or_fake_history_and_camera_assignments_rejected(tmp_path, mode):
    row = sample(tmp_path / "images")
    if mode == "future":
        row["frames"][0]["timestamp_s"] = .3
    elif mode == "out_of_order":
        row["frames"][1]["timestamp_s"] = 0
    elif mode == "anchor_mismatch":
        row["anchor_timestamp_s"] = .4
    elif mode == "repeated_image":
        row["frames"][1]["views"][0]["image"] = row["frames"][0]["views"][0]["image"]
    elif mode == "swapped_cameras":
        row["frames"][1]["views"].reverse()
    else:
        row["frames"][0]["views"][1]["camera_id"] = 0
    with pytest.raises(ValueError):
        JointSequenceDataset(write(tmp_path / "train.jsonl", [row]))


@pytest.mark.parametrize("mode", ["reflection", "scaled_rotation", "bad_bottom", "nan_transform", "negative_focal", "bad_dimensions"])
def test_calibration_must_be_proper_finite_and_match_image(tmp_path, mode):
    row = sample(tmp_path / "images")
    view = row["frames"][0]["views"][0]
    matrix = np.eye(4)
    if mode == "reflection":
        matrix[0, 0] = -1
    elif mode == "scaled_rotation":
        matrix[0, 0] = 2
    elif mode == "bad_bottom":
        matrix[3, 0] = 1
    elif mode == "nan_transform":
        matrix[0, 3] = np.inf
    elif mode == "negative_focal":
        view["intrinsic"][0][0] = -1
    else:
        view["image_size"] = [24, 33]
    view["camera_to_anchor"] = matrix.tolist()
    with pytest.raises(ValueError):
        JointSequenceDataset(write(tmp_path / "train.jsonl", [row]))


def test_empty_or_tampered_images_are_rejected(tmp_path):
    row = sample(tmp_path / "images")
    image = Path(row["frames"][0]["views"][0]["image"])
    Image.new("RGB", (32, 24), (0, 0, 0)).save(image)
    dataset = JointSequenceDataset(write(tmp_path / "train.jsonl", [row]))
    with pytest.raises(ValueError, match="All-zero"):
        dataset[0]
    row["frames"][0]["views"][0]["image_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA256"):
        JointSequenceDataset(write(tmp_path / "train.jsonl", [row]))


def test_optional_image_hash_is_actually_verified(tmp_path):
    row = sample(tmp_path / "images")
    for frame in row["frames"]:
        for view in frame["views"]:
            view["image_sha256"] = hashlib.sha256(Path(view["image"]).read_bytes()).hexdigest()
    dataset = JointSequenceDataset(write(tmp_path / "train.jsonl", [row]))
    assert dataset.provenance["image_files_sha256_verified"] == 6
    assert dataset.provenance["source_image_hashes_verified"]


def test_camera_slot_identity_cannot_change_between_samples(tmp_path):
    first = sample(tmp_path / "first")
    second = sample(tmp_path / "second")
    for frame in second["frames"]:
        frame["views"].reverse()
    with pytest.raises(ValueError, match="all manifest samples"):
        JointSequenceDataset(write(tmp_path / "train.jsonl", [first, second]))


@pytest.mark.parametrize("mode", ["origin", "image", "scene", "wrong_split"])
def test_holdouts_cannot_share_original_drive_or_images(tmp_path, mode):
    train = sample(tmp_path / "train", "train")
    val = sample(tmp_path / "val", "val")
    if mode == "origin":
        val["origin_id"] = train["origin_id"]
    elif mode == "image":
        val["frames"][0]["views"][0]["image"] = train["frames"][0]["views"][0]["image"]
    elif mode == "scene":
        val["group_id"] = train["group_id"]
    else:
        val["split"] = "train"
    with pytest.raises(ValueError, match="leakage|requested"):
        validate_split_manifests(write(tmp_path / "train.jsonl", [train]), write(tmp_path / "val.jsonl", [val]))


def test_reserved_test_is_included_in_leakage_checks(tmp_path):
    paths = {split: write(tmp_path / f"{split}.jsonl", [sample(tmp_path / split, split)]) for split in ("train", "val", "test")}
    report = validate_split_manifests(**paths)
    assert report["disjoint"] and report["diagnostic"]
    assert set(report["manifests"]) == {"train", "val", "test"}
    test = sample(tmp_path / "test", "test", origin="origin_train")
    write(paths["test"], [test])
    with pytest.raises(ValueError, match="Original-drive leakage"):
        validate_split_manifests(**paths)


def analytic_tables(count=30):
    intrinsics, extrinsics = {}, {}
    for frame in range(count):
        for camera in (0, 1):
            matrix = np.eye(4)
            matrix[:3, 3] = [-camera * .532725, 0, -frame * .2]
            extrinsics[(frame, camera)] = matrix
            intrinsics[(frame, camera)] = np.array([[725, 0, 620.5], [0, 725, 187], [0, 0, 1.]])
    return intrinsics, extrinsics


def test_adapter_known_motion_optical_to_anchor_stereo_and_future_targets(tmp_path):
    intrinsics, extrinsics = analytic_tables()
    row = adapter._build_rows("Scene06", "clone", 30, intrinsics, extrinsics, tmp_path, .1)[0]
    assert row["origin_id"] == "kitti:2011_09_26_drive_0018"
    assert row["split"] == "train"
    assert np.allclose(row["state"], [2, 0])
    assert np.allclose(row["targets"], [[1, 0, 2], [2, 0, 2], [3, 0, 2], [4, 0, 2], [5, 0, 2]])
    current = row["frames"][-1]["views"]
    assert np.allclose(current[0]["camera_to_anchor"], adapter.OPTICAL_TO_ANCHOR)
    assert np.allclose(np.array(current[1]["camera_to_anchor"])[:3, 3], [0, -.532725, 0])
    previous = np.array(row["frames"][0]["views"][0]["camera_to_anchor"])
    assert np.allclose(previous[:3, 3], [-.4, 0, 0])
    assert row["nominal_timing_assumed"] and row["synthetic_simulator_data"] and not row["diagnostic"]


def test_future_ground_truth_changes_cannot_change_input_state_or_geometry(tmp_path):
    intrinsics, extrinsics = analytic_tables()
    before = adapter._build_rows("Scene06", "rain", 30, intrinsics, extrinsics, tmp_path, .1)[0]
    changed = copy.deepcopy(extrinsics)
    for (frame, camera), matrix in changed.items():
        if frame > 2:
            matrix[0, 3] -= .2 * frame
            matrix[2, 3] -= frame * .5
    after = adapter._build_rows("Scene06", "rain", 30, intrinsics, changed, tmp_path, .1)[0]
    assert before["frames"] == after["frames"]
    assert before["state"] == after["state"]
    assert before["targets"] != after["targets"]


def test_yaw_rate_uses_backward_heading_with_left_positive_sign(tmp_path):
    intrinsics, extrinsics = analytic_tables()
    angle = .1
    # Previous optical forward points towards anchor right (negative left),
    # so reaching the current forward direction is a positive left yaw change.
    world_from_previous = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    center = -extrinsics[(1, 0)][:3, 3]
    extrinsics[(1, 0)][:3, :3] = world_from_previous.T
    extrinsics[(1, 0)][:3, 3] = -world_from_previous.T @ center
    row = adapter._build_rows("Scene06", "clone", 30, intrinsics, extrinsics, tmp_path, .1)[0]
    assert np.isclose(row["state"][1], 1)


@pytest.mark.parametrize("period", [0, -.1, float("nan"), .2, True])
def test_nominal_timing_must_fit_requested_offsets(period):
    with pytest.raises(ValueError):
        adapter._future_frames(period)


def test_official_adapter_requires_verified_source_records_before_output(tmp_path):
    root = tmp_path / "unverified_source"
    root.mkdir()
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        adapter.prepare(root, output, .1)
    assert not output.exists()
