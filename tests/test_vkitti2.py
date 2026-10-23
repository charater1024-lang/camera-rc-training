import importlib.util
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
import pytest

from camera_rc.data import ManifestDataset, read_manifest, validate_splits


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_vkitti2", PROJECT / "scripts/prepare_vkitti2.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


@pytest.fixture
def source(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "SCENE_COUNTS", {"Scene01": 2, "Scene02": 2, "Scene06": 2})
    monkeypatch.setattr(preparation, "SCENE_SPLITS", {"Scene01": "val", "Scene02": "test", "Scene06": "train"})
    monkeypatch.setattr(preparation, "VARIANTS", ("clone", "rain"))
    monkeypatch.setattr(preparation, "EXPECTED_SIZE", (30, 8))
    roots = {"rgb_root": tmp_path / "rgb", "segmentation_root": tmp_path / "labels", "metadata_root": tmp_path / "textgt"}
    palette = "Category r g b\n" + "".join(f"{key} {' '.join(map(str, value))}\n" for key, value in preparation.PALETTE.items())
    mask = np.tile(np.array(list(preparation.PALETTE.values()), dtype=np.uint8), (8, 2, 1))
    counter = 0
    for scene in preparation.SCENE_COUNTS:
        for variant in preparation.VARIANTS:
            metadata = roots["metadata_root"] / scene / variant
            metadata.mkdir(parents=True)
            for name in preparation.METADATA_NAMES:
                (metadata / name).write_text(palette if name == "colors.txt" else "preserved metadata\n", encoding="utf-8")
            for camera in preparation.CAMERAS:
                rgb = roots["rgb_root"] / scene / variant / "frames/rgb" / camera
                labels = roots["segmentation_root"] / scene / variant / "frames/classSegmentation" / camera
                rgb.mkdir(parents=True)
                labels.mkdir(parents=True)
                for frame in range(2):
                    counter += 1
                    Image.new("RGB", (30, 8), (counter * 9, 23, counter * 5)).save(rgb / f"rgb_{frame:05d}.jpg", quality=100, subsampling=0)
                    Image.fromarray(mask).save(labels / f"classgt_{frame:05d}.png")
    return roots


def paths(source, scene="Scene06", variant="clone", camera="Camera_0", frame=0):
    tail = Path(scene) / variant / "frames"
    return (source["rgb_root"] / tail / "rgb" / camera / f"rgb_{frame:05d}.jpg",
            source["segmentation_root"] / tail / "classSegmentation" / camera / f"classgt_{frame:05d}.png")


def test_native_masks_all_variants_and_stereo_stay_in_original_drive(source, tmp_path):
    originals = {p: p.read_bytes() for root in source.values() for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "prepared"
    report = preparation.prepare_vkitti2(**source, output=output)
    assert report["complete"] and report["total_samples"] == 24
    assert report["source_drive_holdouts_checked"] and not report["training_started"]
    assert not report["test_used_for_training_or_validation"] and not (output / ".incomplete").exists()
    assert all(p.read_bytes() == contents for p, contents in originals.items())
    manifests = {split: read_manifest(output / f"{split}.jsonl") for split in ("train", "val", "test")}
    assert [len(x) for x in manifests.values()] == [8, 8, 8]
    validate_splits(manifests["train"], manifests["val"])
    validate_splits(manifests["train"] + manifests["val"], manifests["test"])
    config = {"task": "segmentation", "model": {"num_classes": 2}, "data": {"image_size": [192, 320], "label_space_id": "road_surface_v1"}}
    for split, rows in manifests.items():
        assert len({x["origin_id"] for x in rows}) == 1
        assert {x["variant"] for x in rows} == {"clone", "rain"}
        assert {x["camera_id"] for x in rows} == {0, 1}
        for row in rows:
            assert row["group_id"] == row["origin_id"] == "kitti:" + row["source_video"]
            assert row["dataset_id"] == "vkitti2_0_3" and row["synthetic"]
            target = np.asarray(Image.open(row["target"]))
            assert target[0, 5] == 0 and target[0, 14] == 255
            assert target[0, 0] == 1 and target[0, 13] == 1
            assert len(row["image_sha256"]) == len(row["image_rgb_sha256"]) == 64
        assert ManifestDataset(rows, config)[0]["target"].shape == (192, 320)
    assert report["metadata_files"]["Scene06/clone"]["intrinsic.txt"]["sha256"]


@pytest.mark.parametrize("problem", ["unknown_color", "indexed_mask", "mask_size", "image_size", "truncated_image", "truncated_mask", "ignored", "missing_label", "missing_pair", "extra_frame", "extra_camera", "missing_variant", "extra_scene", "bad_palette", "duplicate_palette", "missing_palette", "missing_intrinsic"])
def test_invalid_incomplete_or_unmapped_data_rejected(source, tmp_path, problem):
    image, label = paths(source)
    metadata = source["metadata_root"] / "Scene06/clone"
    if problem == "unknown_color":
        Image.new("RGB", (30, 8), (100, 60, 101)).save(label)
    elif problem == "indexed_mask":
        Image.new("P", (30, 8), 5).save(label)
    elif problem == "mask_size":
        Image.new("RGB", (29, 8), preparation.PALETTE["Road"]).save(label)
    elif problem == "image_size":
        Image.new("RGB", (29, 8), (35, 60, 100)).save(image)
    elif problem == "truncated_image":
        image.write_bytes(image.read_bytes()[:35])
    elif problem == "truncated_mask":
        label.write_bytes(label.read_bytes()[:35])
    elif problem == "ignored":
        Image.new("RGB", (30, 8), (0, 0, 0)).save(label)
    elif problem == "missing_label":
        label.unlink()
    elif problem == "missing_pair":
        image.unlink()
        label.unlink()
    elif problem == "extra_frame":
        shutil.copyfile(image, image.with_name("rgb_00002.jpg"))
    elif problem == "extra_camera":
        (image.parent.parent / "Camera_2").mkdir()
    elif problem == "missing_variant":
        shutil.rmtree(source["rgb_root"] / "Scene06/rain")
    elif problem == "extra_scene":
        (source["rgb_root"] / "Scene99").mkdir()
    elif problem == "bad_palette":
        path = metadata / "colors.txt"
        path.write_text(path.read_text().replace("Road 100 60 100", "Road 100 60 101"))
    elif problem == "duplicate_palette":
        path = metadata / "colors.txt"
        path.write_text(path.read_text() + "Road 100 60 100\n")
    elif problem == "missing_palette":
        (metadata / "colors.txt").unlink()
    else:
        (metadata / "intrinsic.txt").unlink()
    output = tmp_path / "prepared"
    with pytest.raises((ValueError, OSError)):
        preparation.prepare_vkitti2(**source, output=output)
    assert (output / ".incomplete").exists()
    assert not (output / "train.jsonl").exists()
    assert not json.loads((output / "report.json").read_text())["complete"]


def test_exact_duplicates_across_drives_rejected_but_same_drive_counted(source, tmp_path):
    first, _ = paths(source)
    second, _ = paths(source, variant="rain")
    shutil.copyfile(first, second)
    report = preparation.prepare_vkitti2(**source, output=tmp_path / "within_drive")
    assert report["within_drive_duplicates"]["file"] == 1
    foreign, _ = paths(source, scene="Scene01")
    shutil.copyfile(first, foreign)
    with pytest.raises(ValueError, match="Duplicate image across source drives"):
        preparation.prepare_vkitti2(**source, output=tmp_path / "cross_drive")


def test_cross_dataset_original_drive_leakage_is_detected(source, tmp_path):
    previous = tmp_path / "kitti_val.jsonl"
    previous.write_text(json.dumps({"origin_id": "kitti:2011_09_26_drive_0009"}) + "\n")
    preparation.prepare_vkitti2(**source, output=tmp_path / "good", related_val_manifest=[previous])
    with pytest.raises(ValueError, match="Cross-dataset source leakage"):
        preparation.prepare_vkitti2(**source, output=tmp_path / "bad", related_train_manifest=[previous])
    previous.write_text(json.dumps({"origin_id": "kitti:2011_09_26_drive_0011"}) + "\n")
    with pytest.raises(ValueError, match="Cross-dataset source leakage"):
        preparation.prepare_vkitti2(**source, output=tmp_path / "bad_test", related_train_manifest=[previous])


@pytest.mark.parametrize("content", ["", "{}\n", '{"origin_id":null}\n'])
def test_related_manifest_without_explicit_origins_is_rejected(source, tmp_path, content):
    previous = tmp_path / "missing_origin.jsonl"
    previous.write_text(content)
    with pytest.raises(ValueError):
        preparation.prepare_vkitti2(**source, output=tmp_path / "prepared", related_val_manifest=[previous])


def test_existing_or_source_nested_output_is_never_overwritten(source, tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    sentinel = existing / "keep.txt"
    sentinel.write_text("unchanged")
    with pytest.raises(FileExistsError):
        preparation.prepare_vkitti2(**source, output=existing)
    assert sentinel.read_text() == "unchanged"
    with pytest.raises(ValueError, match="separate"):
        preparation.prepare_vkitti2(**source, output=source["rgb_root"] / "prepared")


def test_real_release_scene_counts_and_heldout_origin_constants():
    assert sum(preparation.SCENE_COUNTS.values()) * len(preparation.VARIANTS) * len(preparation.CAMERAS) == 42520
    assert preparation.SOURCE_DRIVES["Scene01"] == "2011_09_26_drive_0009"
    assert preparation.SCENE_SPLITS["Scene01"] == "val"
    assert preparation.SCENE_SPLITS["Scene02"] == "test"
