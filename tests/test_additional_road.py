import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from camera_rc.data import ManifestDataset, read_manifest, validate_splits


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_additional_road", PROJECT / "scripts/prepare_additional_road.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "CamVid"
    sample_names = {
        "train": ["0016E5_00390.png", "0006R0_f00930.png", "0001TP_006690.png"],
        "val": ["0016E5_00391.png"],
        "test": ["0001TP_008550.png", "Seq05VD_f00000.png"],
    }
    counter = 0
    for split, names in sample_names.items():
        (root / split).mkdir(parents=True)
        (root / (split + "annot")).mkdir()
        for name in names:
            counter += 1
            Image.new("RGB", (32, 24), (counter, 30, 60)).save(root / split / name)
            labels = np.tile(np.arange(12, dtype=np.uint8), (32, 2)).T
            Image.fromarray(labels).save(root / (split + "annot") / name)
    monkeypatch.setattr(preparation, "CAMVID_SOURCE_COUNTS", {"train": 3, "val": 1, "test": 2})
    monkeypatch.setattr(preparation, "CAMVID_VIDEO_COUNTS", {"0016E5": 2, "0006R0": 1, "0001TP": 2, "Seq05VD": 1})
    monkeypatch.setattr(preparation, "CAMVID_SPLIT_COUNTS", {"train": 3, "val": 2, "test": 1})
    return root


def test_whole_videos_remap_native_decode_and_source_preservation(source, tmp_path):
    originals = {p: p.read_bytes() for p in source.rglob("*.png")}
    out = tmp_path / "prepared"
    report = preparation.prepare_additional_road("camvid", source, out)
    assert report["complete"] and report["global_holdouts_checked"]
    assert report["total_samples"] == 6 and not (out / ".incomplete").exists()
    assert not report["training_started"] and not report["test_used_for_training_or_validation"]
    assert all(p.read_bytes() == contents for p, contents in originals.items())
    splits = {split: read_manifest(out / (split + ".jsonl")) for split in ("train", "val", "test")}
    assert [len(splits[name]) for name in splits] == [3, 2, 1]
    validate_splits(splits["train"], splits["val"])
    validate_splits(splits["train"] + splits["val"], splits["test"])
    assert {row["source_split"] for row in splits["val"]} == {"train", "test"}
    assert {row["source_video"] for row in splits["test"]} == {"Seq05VD"}
    config = {"task": "segmentation", "model": {"num_classes": 2}, "data": {"image_size": [192, 320], "label_space_id": "road_surface_v1"}}
    for values in splits.values():
        dataset = ManifestDataset(values, config)
        for index, row in enumerate(values):
            assert row["dataset_id"] == "camvid_segnet"
            assert row["group_id"] == row["origin_id"] == "camvid:" + row["source_video"]
            assert Path(row["image"]) in originals
            actual = np.asarray(Image.open(row["target"]))
            assert actual[3, 0] == 0 and actual[11, 0] == 255
            assert actual[0, 0] == 1 and actual[10, 0] == 1
            assert dataset[index]["target"].shape == (192, 320)


@pytest.mark.parametrize("problem", ["bad_id", "color", "size", "truncated", "unlabelled", "missing", "missing_image", "extra", "unknown_video"])
def test_corrupt_incomplete_or_ambiguous_source_rejected(source, tmp_path, problem):
    label = source / "trainannot/0016E5_00390.png"
    image = source / "train/0016E5_00390.png"
    if problem == "bad_id":
        Image.new("L", (32, 24), 12).save(label)
    elif problem == "color":
        Image.new("RGB", (32, 24), (128, 64, 128)).save(label)
    elif problem == "size":
        Image.new("L", (31, 24), 3).save(label)
    elif problem == "truncated":
        image.write_bytes(image.read_bytes()[:40])
    elif problem == "unlabelled":
        Image.new("L", (32, 24), 11).save(label)
    elif problem == "missing":
        label.unlink()
    elif problem == "missing_image":
        label.unlink()
        image.unlink()
    elif problem == "extra":
        (source / "train/unexpected.txt").write_text("extra")
    else:
        label.rename(label.with_name("unknown_000001.png"))
        image.rename(image.with_name("unknown_000001.png"))
    out = tmp_path / "prepared"
    with pytest.raises((ValueError, OSError)):
        preparation.prepare_additional_road("camvid", source, out)
    assert (out / ".incomplete").exists()
    assert not json.loads((out / "report.json").read_text(encoding="utf-8"))["complete"]
    assert not (out / "train.jsonl").exists()


@pytest.mark.parametrize("reencode", [False, True])
def test_duplicates_across_holdout_detected_by_file_or_decoded_pixels(source, tmp_path, reencode):
    image = source / "train/0016E5_00390.png"
    holdout = source / "test/Seq05VD_f00000.png"
    if reencode:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("distinct encoding", "same pixels")
        with Image.open(image) as opened:
            opened.save(holdout, pnginfo=metadata)
        assert image.read_bytes() != holdout.read_bytes()
    else:
        holdout.write_bytes(image.read_bytes())
    with pytest.raises(ValueError, match="Duplicate image"):
        preparation.prepare_additional_road("camvid", source, tmp_path / "out")


def test_rgb_grayscale_ids_accepted_and_unknown_values_rejected(source, tmp_path):
    label = source / "trainannot/0016E5_00390.png"
    with Image.open(label) as opened:
        converted = opened.convert("RGB")
    converted.save(label)
    assert preparation.prepare_additional_road("camvid", source, tmp_path / "out")["complete"]


def test_existing_or_nested_output_never_modified(source, tmp_path):
    existing = tmp_path / "out"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        preparation.prepare_additional_road("camvid", source, existing)
    assert marker.read_text() == "keep"
    with pytest.raises(ValueError, match="separate"):
        preparation.prepare_additional_road("camvid", source, source / "out")
    assert not (source / "out").exists()


def test_group_leakage_guard(source, tmp_path, monkeypatch):
    original = preparation._camvid_pairs
    def wrong_split(root):
        pairs = original(root)
        image, label, source_split, video, split = pairs[0]
        pairs[0] = (image, label, source_split, video, "test")
        return pairs
    monkeypatch.setattr(preparation, "_camvid_pairs", wrong_split)
    with pytest.raises(ValueError, match="Source-video leakage"):
        preparation.prepare_additional_road("camvid", source, tmp_path / "out")


@pytest.fixture
def kitti_source(tmp_path, monkeypatch):
    root = tmp_path / "kitti"
    (root / "training/image_2").mkdir(parents=True)
    (root / "training/gt_image_2").mkdir()
    (root / "testing/image_2").mkdir(parents=True)
    train_drive, val_drive = "2011_09_26_drive_0001", "2011_09_26_drive_0009"
    count = 0
    for category in ("um", "umm", "uu"):
        lines = []
        for index, drive in enumerate((train_drive, val_drive)):
            count += 1
            lines.append(f"{drive} {count:010d}\n")
            Image.new("RGB", (32, 24), (count, 40, 80)).save(root / "training/image_2" / f"{category}_{index:06d}.png")
            target = np.zeros((24, 32, 3), dtype=np.uint8)
            target[8:, :, 0] = 255
            target[16:, :, 2] = 255
            # B outside valid area must remain ignored, not become road.
            target[:4, :, 2] = 255
            Image.fromarray(target).save(root / "training/gt_image_2" / f"{category}_road_{index:06d}.png")
            if category == "um":
                Image.new("RGB", (32, 24), (255, 0, 255)).save(root / "training/gt_image_2" / f"um_lane_{index:06d}.png")
        (root / "training" / (category + "_mapping.txt")).write_text("".join(lines), encoding="utf-8")
    # Unlabelled testing images must not be recruited into either split.
    Image.new("RGB", (32, 24), (255, 255, 255)).save(root / "testing/image_2/um_000000.png")
    monkeypatch.setattr(preparation, "KITTI_CATEGORY_COUNTS", {"um": 2, "umm": 2, "uu": 2})
    monkeypatch.setattr(preparation, "KITTI_DRIVE_COUNT", 2)
    monkeypatch.setattr(preparation, "KITTI_VAL_DRIVES", frozenset({val_drive}))
    monkeypatch.setattr(preparation, "KITTI_SPLIT_COUNTS", {"train": 3, "val": 3})
    return root


def test_kitti_valid_region_rgb_mapping_and_category_transcending_groups(kitti_source, tmp_path):
    out = tmp_path / "out"
    report = preparation.prepare_additional_road("kitti_road", kitti_source, out)
    assert report["complete"] and report["total_samples"] == 6
    assert not report["ego_lane_labels_used"] and not (out / "test.jsonl").exists()
    assert len(report["mapping_sha256"]) == 3
    train, val = read_manifest(out / "train.jsonl"), read_manifest(out / "val.jsonl")
    validate_splits(train, val)
    assert len({row["group_id"] for row in train}) == len({row["group_id"] for row in val}) == 1
    for row in train + val:
        assert row["dataset_id"] == "kitti_road" and row["source_split"] == "training"
        assert "_road_" in row["target_original"]
        target = np.asarray(Image.open(row["target"]))
        assert target[0, 0] == target[5, 0] == 255
        assert target[10, 0] == 1 and target[20, 0] == 0


def test_kitti_separate_mapping_root_does_not_require_copying(kitti_source, tmp_path):
    mapping_root = tmp_path / "separate_mapping"
    (mapping_root / "training").mkdir(parents=True)
    for path in (kitti_source / "training").glob("*_mapping.txt"):
        path.rename(mapping_root / "training" / path.name)
    originals = {path: path.read_bytes() for path in mapping_root.rglob("*.txt")}
    out = tmp_path / "out"
    report = preparation.prepare_additional_road("kitti_road", kitti_source, out, mapping_root=mapping_root)
    assert report["complete"] and report["mapping_root"] == str(mapping_root.resolve())
    assert all(path.read_bytes() == content for path, content in originals.items())
    assert not list((kitti_source / "training").glob("*_mapping.txt"))
    with pytest.raises(ValueError, match="separate"):
        preparation.prepare_additional_road("kitti_road", kitti_source, mapping_root / "out", mapping_root=mapping_root)


@pytest.mark.parametrize("problem", ["mapping_missing", "mapping_short", "mapping_bad", "duplicate_origin", "missing_road", "wrong_rgb", "rgb_green", "unexpected_file"])
def test_kitti_invalid_mapping_or_labels_fail_closed(kitti_source, tmp_path, problem):
    mapping = kitti_source / "training/um_mapping.txt"
    label = kitti_source / "training/gt_image_2/um_road_000000.png"
    if problem == "mapping_missing":
        mapping.unlink()
    elif problem == "mapping_short":
        mapping.write_text("2011_09_26_drive_0001 0000000001\n")
    elif problem == "mapping_bad":
        mapping.write_text("bad\nwrong\n")
    elif problem == "duplicate_origin":
        other = kitti_source / "training/umm_mapping.txt"
        other.write_bytes(mapping.read_bytes())
    elif problem == "missing_road":
        label.unlink()
    elif problem == "wrong_rgb":
        Image.new("L", (32, 24), 1).save(label)
    elif problem == "rgb_green":
        Image.new("RGB", (32, 24), (255, 1, 255)).save(label)
    else:
        Image.new("RGB", (32, 24)).save(label.with_name("uu_lane_000000.png"))
    out = tmp_path / "out"
    with pytest.raises((ValueError, OSError)):
        preparation.prepare_additional_road("kitti_road", kitti_source, out)
    assert (out / ".incomplete").exists() and not (out / "train.jsonl").exists()
