import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from camera_rc.config import load_config
from camera_rc.curriculum import run_curriculum
from camera_rc.data import ManifestDataset, load_data_sources, read_manifest, validate_splits


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_multiview_training", PROJECT / "scripts/prepare_multiview_training.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def raw_dataset(root):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for split_index, split in enumerate(("train", "val")):
        for camera_index, camera in enumerate(preparation.CAMERAS):
            stem = f"{split}-{camera}"
            image, mask = root / f"{stem}.png", root / f"{stem}-mask.png"
            Image.new("RGB", (20, 16), (camera_index * 30, split_index * 50, 100)).save(image)
            labels = np.zeros((16, 20), dtype=np.uint8)
            labels[0, :2] = [1, 255]
            Image.fromarray(labels).save(mask)
            rows.append({"image": image.name, "mask": mask.name, "camera": camera,
                         "clip_id": f"session-{split}", "split": split})
    path = write_rows(root / "raw.jsonl", rows)
    return path, rows


def test_four_views_use_common_scene_identity_and_preserve_ignore(tmp_path, monkeypatch):
    source, _ = raw_dataset(tmp_path / "raw")
    monkeypatch.chdir(tmp_path)
    output = tmp_path / "prepared"
    report = preparation.prepare_multiview(source, output)
    assert report["complete"] and report["samples"] == 8
    assert not (output / ".incomplete").exists()
    train, val = [], []
    for camera in preparation.CAMERAS:
        train.extend(read_manifest(output / f"train_{camera}.jsonl"))
        val.extend(read_manifest(output / f"val_{camera}.jsonl"))
        assert report["manifests"][f"train_{camera}"]["pixel_counts"] == {0: 318, 1: 1, 255: 1}
    assert {row["group_id"] for row in train} == {"session-train"}
    assert {row["origin_id"] for row in train} == {"vehicle_multiview:session-train"}
    assert {row["dataset_id"] for row in train + val} == {"vehicle_multiview"}
    assert all(Path(row["image"]).is_absolute() for row in train + val)
    assert all(row["source_image"] == row["image"] for row in train + val)
    validate_splits(train, val)
    config = {"task": "segmentation", "model": {"num_classes": 2},
              "data": {"label_space_id": "road_surface_v1", "image_size": [192, 320]}}
    item = ManifestDataset(train, config)[0]
    assert item["image"].shape == (3, 192, 320)
    assert set(item["target"].unique().tolist()) == {0, 1, 255}
    # Original files are referenced and remain unmodified; no image cache/copies.
    assert len(list(output.iterdir())) == 9


def test_different_cameras_cannot_put_same_original_clip_in_different_splits(tmp_path):
    source, rows = raw_dataset(tmp_path / "raw")
    rows[-1]["clip_id"] = "session-train"
    write_rows(source, rows)
    with pytest.raises(ValueError, match="clip split leakage"):
        preparation.prepare_multiview(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_duplicate_image_path_is_rejected_even_in_another_camera(tmp_path):
    source, rows = raw_dataset(tmp_path / "raw")
    rows[1]["image"] = "./" + rows[0]["image"]
    write_rows(source, rows)
    with pytest.raises(ValueError, match="duplicate image"):
        preparation.prepare_multiview(source, tmp_path / "out")


@pytest.mark.parametrize("field,value,match", [
    ("camera", "back", "camera must"),
    ("split", "test", "split must"),
    ("clip_id", "", "clip_id must"),
    ("clip_id", " session ", "clip_id must"),
    ("label_space_id", "bdd_original", "label_space_id"),
    ("dataset_id", "front-camera", "dataset_id differs"),
    ("group_id", "one-frame", "group_id must"),
    ("origin_id", "front-session", "origin_id must"),
])
def test_metadata_cannot_hide_scene_or_label_mismatches(tmp_path, field, value, match):
    source, rows = raw_dataset(tmp_path / "raw")
    rows[0][field] = value
    write_rows(source, rows)
    with pytest.raises(ValueError, match=match):
        preparation.prepare_multiview(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("mode,size,value,match", [
    ("RGB", (20, 16), (0, 0, 0), "single-channel"),
    ("L", (20, 16), 2, "mask IDs"),
    ("L", (20, 16), 255, "no supervised pixels"),
    ("L", (21, 16), 0, "alignment mismatch"),
])
def test_invalid_or_unreviewed_mask_encoding_is_rejected(tmp_path, mode, size, value, match):
    source, rows = raw_dataset(tmp_path / "raw")
    Image.new(mode, size, value).save(source.parent / rows[0]["mask"])
    with pytest.raises(ValueError, match=match):
        preparation.prepare_multiview(source, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_missing_view_validation_is_not_silently_omitted(tmp_path):
    source, rows = raw_dataset(tmp_path / "raw")
    write_rows(source, rows[:-1])
    with pytest.raises(ValueError, match="val/right"):
        preparation.prepare_multiview(source, tmp_path / "out")


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    sentinel = output / "keep.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        preparation.prepare_multiview(tmp_path / "missing.jsonl", output)
    assert sentinel.read_text() == "keep"


def test_shipped_curriculum_resolves_prepared_paths_and_balances_cameras(tmp_path):
    cfg_folder = tmp_path / "configs"
    (cfg_folder / "stages").mkdir(parents=True)
    for relative in ("curriculum_multiview.json", "stages/road_public.json", "stages/road_mixed.json", "stages/multiview_vehicle.json"):
        shutil.copy2(PROJECT / "configs" / relative, cfg_folder / relative)
    source, _ = raw_dataset(tmp_path / "raw")
    preparation.prepare_multiview(source, tmp_path / "data/road/multiview")
    for dataset in ("bdd", "cityscapes"):
        for split in ("train", "val"):
            folder = tmp_path / "data/road" / f"{dataset}_{split}"
            folder.mkdir(parents=True)
            Image.new("RGB", (20, 16), (0, 20, 40)).save(folder / "image.png")
            Image.new("L", (20, 16), 0).save(folder / "mask.png")
            write_rows(folder / "manifest.jsonl", [{"image": "image.png", "target": "mask.png",
                "dataset_id": dataset, "group_id": split, "label_space_id": "road_surface_v1"}])
    plan = run_curriculum(cfg_folder / "curriculum_multiview.json")
    assert plan["validation"]["global_holdouts_checked"]
    assert [stage["samples"] for stage in plan["stages"]] == [
        {"train": 1, "val": 1}, {"train": 2, "val": 2}, {"train": 6, "val": 6}]
    stage = plan["stages"][-1]
    assert stage["init_from"]["stage"] == "road_mixed"
    config = load_config(cfg_folder / "stages/multiview_vehicle.json")
    sources, heldout = load_data_sources(config)
    assert [source["weight"] for source in sources] == [1, 1, 1, 1, 0.5, 0.5]
    assert {source["name"] for source in heldout} >= {f"vehicle_{camera}" for camera in preparation.CAMERAS}
    assert config["training"]["num_workers"] == 0
    assert config["model"] == {"num_classes": 2, "width": 16}
    assert not (tmp_path / "runs").exists()  # Plan is read-only; no training started.


def test_generated_identities_still_catch_leakage_in_combined_canonical_data(tmp_path):
    source, _ = raw_dataset(tmp_path / "raw")
    preparation.prepare_multiview(source, tmp_path / "out")
    train = read_manifest(tmp_path / "out/train_front.jsonl")
    moved_to_val = read_manifest(tmp_path / "out/train_rear.jsonl")
    with pytest.raises(ValueError, match="scene leakage"):
        validate_splits(train, moved_to_val)
