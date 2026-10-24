"""Explicit ego-lane annotation preparation on small synthetic CPU fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image, PngImagePlugin
import pytest

from camera_rc.config import load_config
from camera_rc.lane_training import LABEL_SPACE_ID, prepare_lane_dataset


PROJECT = Path(__file__).resolve().parents[1]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _annotations(root):
    rows = []
    for split, color in (("train", (30, 70, 100)), ("val", (100, 70, 30))):
        path = root / f"{split}.png"
        Image.new("RGB", (32, 24), color).save(path)
        rows.append({"image": path.name, "group_id": f"drive-{split}", "split": split,
                     "dataset_id": "synthetic_lane_fixture", "diagnostic_only": True,
                     "label_space_id": LABEL_SPACE_ID,
                     "valid_polygon": [[2, 2], [29, 2], [29, 21], [2, 21]],
                     "ego_lane_polygon": [[12, 4], [19, 4], [19, 21], [12, 21]]})
    path = root / "annotations.jsonl"
    _write(path, rows)
    return path, rows


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_explicit_masks_preserve_inputs_and_publish_auditable_completion(tmp_path):
    annotations, rows = _annotations(tmp_path)
    before = {path: path.read_bytes() for path in tmp_path.iterdir() if path.is_file()}
    output = tmp_path / "prepared"
    report = prepare_lane_dataset(annotations, output, image_size=(24, 32))
    assert all(path.read_bytes() == content for path, content in before.items())
    assert report["preparation_complete"] and report["label_space_id"] == "ego_lane_v1"
    assert report["diagnostic_only"] and report["samples"] == {"train": 1, "val": 1}
    assert report["groups"] == {"train": 1, "val": 1}
    assert not any(report[key] for key in ("training_executed", "model_adopted",
                                         "automatic_model_replacement", "accuracy_measured",
                                         "train_val_group_leakage"))
    assert not (output / ".incomplete").exists()
    marker = json.loads((output / ".complete").read_text(encoding="utf-8"))
    assert marker["preparation_complete"]
    assert marker["report_sha256"] == _sha(output / "report.json")
    assert report["annotation_sha256"] == _sha(annotations)
    for relative, digest in report["file_sha256"].items():
        assert _sha(output / relative) == digest
    for index, source in enumerate(report["sources"]):
        assert source["image_sha256"] == _sha(tmp_path / rows[index]["image"])
        assert source["mask_sha256"] == _sha(Path(source["mask"]))
        with Image.open(source["image"]) as image:
            rgb = image.convert("RGB")
            assert source["rgb_sha256"] == hashlib.sha256(b"RGB:32x24\0" + rgb.tobytes()).hexdigest()
        with Image.open(source["mask"]) as image:
            mask = np.asarray(image)
        assert set(np.unique(mask)) == {0, 1, 255}
        assert mask[0, 0] == 255 and mask[3, 3] == 0 and mask[10, 16] == 1
        counts = {str(label): int((mask == label).sum()) for label in (0, 1, 255)}
        assert report["pixel_counts"][source["split"]] == counts
        assert report["resized_pixel_counts"][source["split"]] == counts
    config = load_config(output / "train_config.json")
    assert config["model"] == {"num_classes": 2, "width": 16}
    assert config["data"]["image_size"] == [24, 32]
    assert config["data"]["label_space_id"] == LABEL_SPACE_ID
    assert config["diagnostic_only"] and config["training"]["device"] == "cpu"
    assert config["training"]["segmentation_class_weights"] == [1., 2.]
    assert config["training"]["output_dir"] == str(output / "training_run")
    assert not Path(config["training"]["output_dir"]).exists()
    assert not list(output.rglob("*.pt"))
    for split in ("train", "val"):
        manifest = Path(config["data"][f"{split}_manifest"])
        row = json.loads(manifest.read_text(encoding="utf-8"))
        assert row["task"] == "segmentation" and row["label_space_id"] == LABEL_SPACE_ID
        assert row["group_id"] == f"drive-{split}" and row["diagnostic_only"]
        assert row["source_provenance"]["annotation_sha256"] == _sha(annotations)


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "already_exists"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("my existing result", encoding="utf-8")
    with pytest.raises(FileExistsError, match="new output directory"):
        prepare_lane_dataset(tmp_path / "missing.jsonl", output)
    assert marker.read_text(encoding="utf-8") == "my existing result"
    assert list(output.iterdir()) == [marker]


@pytest.mark.parametrize("variant, message", [
    ("road_space", "road-surface"), ("missing_polygon", "explicit ego_lane_polygon"),
    ("group_leak", "group_id leakage"), ("no_val", "nonempty train and val"),
    ("mixed_dataset", "Mixed dataset_id"), ("diagnostic_string", "diagnostic_only must be boolean"),
])
def test_preflight_refuses_ambiguous_or_leaking_annotations(tmp_path, variant, message):
    annotations, rows = _annotations(tmp_path)
    if variant == "road_space":
        rows[0]["label_space_id"] = "road_surface_v1"
    elif variant == "missing_polygon":
        rows[0].pop("ego_lane_polygon")
        rows[0]["target"] = "road_surface.png"
    elif variant == "group_leak":
        rows[1]["group_id"] = rows[0]["group_id"]
    elif variant == "no_val":
        rows[1]["split"] = "train"
    elif variant == "mixed_dataset":
        rows[1]["dataset_id"] = "second_source"
    else:
        rows[0]["diagnostic_only"] = "true"
    _write(annotations, rows)
    output = tmp_path / "refused"
    with pytest.raises(ValueError, match=message):
        prepare_lane_dataset(annotations, output)
    assert not output.exists()


@pytest.mark.parametrize("variant, message", [
    ("path", "Duplicate image path"), ("byte", "Duplicate image byte SHA256"),
    ("rgb", "Duplicate decoded RGB SHA256"),
])
def test_duplicate_images_across_splits_leave_only_incomplete_output(tmp_path, variant, message):
    annotations, rows = _annotations(tmp_path)
    train, val = tmp_path / "train.png", tmp_path / "val.png"
    if variant == "path":
        rows[1]["image"] = "train.png"
    elif variant == "byte":
        val.write_bytes(train.read_bytes())
    else:
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("different_container", "same RGB pixels")
        with Image.open(train) as image:
            image.save(val, pnginfo=metadata)
        assert train.read_bytes() != val.read_bytes()
    _write(annotations, rows)
    output = tmp_path / "incomplete"
    with pytest.raises(ValueError, match=message):
        prepare_lane_dataset(annotations, output)
    assert (output / ".incomplete").is_file()
    assert not (output / ".complete").exists()
    assert not (output / "report.json").exists()
    assert not (output / "train_config.json").exists()
    assert not list(output.glob("*.jsonl"))


@pytest.mark.parametrize("polygon, message", [
    ([[12, 4], [19, 4]], "at least three"),
    ([[12, 4], [19, 4], [19, 30]], "outside image"),
    ([[12, 4], [19, 4], [float("nan"), 20]], "finite numeric"),
    ([[12, 4], [19, 4], [19, 4], [12, 20]], "distinct vertices"),
    ([[4, 4], [8, 8], [12, 12]], "nonzero area"),
    ([[4, 4], [20, 18], [4, 18], [18, 4]], "self-intersections"),
    ([[0, 0], [5, 0], [5, 5], [0, 5]], "entirely inside"),
])
def test_invalid_polygons_never_receive_a_completion_marker(tmp_path, polygon, message):
    annotations, rows = _annotations(tmp_path)
    rows[0]["ego_lane_polygon"] = polygon
    _write(annotations, rows)
    output = tmp_path / "invalid_polygon"
    with pytest.raises(ValueError, match=message):
        prepare_lane_dataset(annotations, output)
    assert (output / ".incomplete").is_file()
    assert not (output / ".complete").exists()


def test_ego_edges_cannot_cross_outside_concave_valid_region(tmp_path):
    annotations, rows = _annotations(tmp_path)
    rows[0]["valid_polygon"] = [[2, 2], [29, 2], [29, 21], [21, 21],
                                [21, 8], [10, 8], [10, 21], [2, 21]]
    # All ego vertices fit within the U-shaped valid region, but its top edge
    # crosses the unannotated notch between the two legs.
    rows[0]["ego_lane_polygon"] = [[5, 15], [26, 15], [16, 4]]
    _write(annotations, rows)
    output = tmp_path / "crossing"
    with pytest.raises(ValueError, match="entirely inside"):
        prepare_lane_dataset(annotations, output)
    assert (output / ".incomplete").exists() and not (output / ".complete").exists()


def test_diagnostic_row_taints_whole_config_and_cli_does_not_start_training(tmp_path, capsys):
    annotations, rows = _annotations(tmp_path)
    rows[1]["diagnostic_only"] = False
    _write(annotations, rows)
    spec = importlib.util.spec_from_file_location("lane_preparation_cli", PROJECT / "scripts/prepare_lane_training.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    output = tmp_path / "cli_prepared"
    loaded_before = set(sys.modules)
    assert cli.main(["--annotations", str(annotations), "--output", str(output),
                     "--image-size", "24", "32"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["diagnostic_only"] and not printed["training_executed"] and not printed["model_adopted"]
    assert load_config(output / "train_config.json")["diagnostic_only"]
    assert not (set(sys.modules) - loaded_before) & {"torch", "camera_rc.training", "ultralytics"}
    assert not (output / "training_run").exists()
