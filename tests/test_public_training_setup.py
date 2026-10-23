import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from camera_rc.config import load_config
from camera_rc.curriculum import run_curriculum
from camera_rc.data import read_manifest


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_public_training", PROJECT / "scripts/prepare_public_training.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def dataset(root, *, city=False, overlap=False):
    images, masks = root / "images", root / "masks"
    for split in ("train", "val"):
        suffix = "train" if overlap else split
        image_dir = images / split / ("aachen" if city else "")
        mask_dir = masks / split / ("aachen" if city else "")
        image_dir.mkdir(parents=True)
        mask_dir.mkdir(parents=True)
        stem = f"aachen_{'000000' if suffix == 'train' else '000001'}_000019" if city else f"clip-{suffix}"
        image = image_dir / (stem + ("_leftImg8bit.png" if city else ".jpg"))
        mask = mask_dir / (stem + ("_gtFine_labelIds.png" if city else "_drivable_id.png"))
        Image.new("RGB", (32, 24), (20, 40, 60)).save(image)
        labels = np.full((24, 32), 7 if city else 0, np.uint8)
        labels[:8] = 11 if city else 2
        labels[:, 0] = 255 if city else 1
        Image.fromarray(labels).save(mask)
    return images, masks


def test_bdd_only_produces_independent_ready_curriculum_without_training(tmp_path):
    images, masks = dataset(tmp_path / "bdd")
    original = {p: p.read_bytes() for root in (images, masks) for p in root.rglob("*") if p.is_file()}
    output = tmp_path / "prepared"
    report = preparation.prepare_public(images, masks, output)
    assert report["complete"] and report["stages"] == 1
    assert not report["training_started"] and not report["deployment_validated"]
    assert not (output / ".incomplete").exists()
    assert not (output / "training").exists()
    assert all(path.read_bytes() == value for path, value in original.items())
    config = load_config(output / "stages/road_public.json")
    assert config["model"] == {"num_classes": 2, "width": 16}
    assert config["data"]["image_size"] == [192, 320]
    assert config["training"]["device"] == "cuda"
    assert "train_sources" not in config["data"]
    assert "train_manifest" in config["data"]
    rows = read_manifest(output / "road/bdd_train/manifest.jsonl")
    assert rows[0]["dataset_id"] == "bdd100k_drivable"
    assert rows[0]["group_id"] == "clip-train"
    assert rows[0]["target_original"] in {str(p) for p in original}
    assert set(np.asarray(Image.open(rows[0]["target"])).ravel()) == {0, 1}
    assert run_curriculum(output / "curriculum.json")["mode"] == "plan"


def test_generated_two_stage_config_runs_real_cpu_training_and_transfers_weights(tmp_path):
    images, masks = dataset(tmp_path / "bdd")
    city_images, city_masks = dataset(tmp_path / "city", city=True)
    output = tmp_path / "prepared"
    report = preparation.prepare_public(
        images, masks, output, cityscapes_images=city_images, cityscapes_labels=city_masks,
        device="cpu", epochs=1, mixed_epochs=1,
    )
    assert report["stages"] == 2 and report["global_holdouts_checked"]
    city_rows = read_manifest(output / "road/cityscapes_val/manifest.jsonl")
    assert city_rows[0]["origin_id"] == "cityscapes:aachen_000001"
    assert 255 in np.asarray(Image.open(city_rows[0]["target"]))
    # Keep these synthetic test checkpoints explicitly ineligible for deployment.
    for name in ("road_public", "road_mixed"):
        path = output / f"stages/{name}.json"
        config = json.loads(path.read_text())
        config["diagnostic_only"] = True
        path.write_text(json.dumps(config), encoding="utf-8")
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = run_curriculum(output / "curriculum.json", execute=True)
    finally:
        torch.set_num_threads(previous_threads)
    assert result["training_complete"]
    assert result["diagnostic"]
    checkpoint = torch.load(output / "training/stages/road_mixed/last.pt", map_location="cpu", weights_only=True)
    assert checkpoint["optimizer_steps"] >= 1
    assert Path(checkpoint["initialization"]["path"]).name == "best.pt"
    assert Path(checkpoint["initialization"]["path"]).parent.name == "road_public"
    state = json.loads((output / "training/state.json").read_text(encoding="utf-8"))
    first, second = state["stages"]
    assert first["status"] == second["status"] == "complete"
    assert second["initialization"] == first["checkpoint"]


def test_existing_output_never_overwritten(tmp_path):
    output = tmp_path / "prepared"
    output.mkdir()
    marker = output / "mine"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        preparation.prepare_public(tmp_path / "missing", tmp_path / "missing", output)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("city", [False, True])
def test_scene_overlap_preserves_failure_marker_and_no_ready_curriculum(tmp_path, city):
    images, masks = dataset(tmp_path / "bdd", overlap=not city)
    options = {}
    if city:
        ci, cm = dataset(tmp_path / "city", city=True, overlap=True)
        options.update(cityscapes_images=ci, cityscapes_labels=cm)
    output = tmp_path / "prepared"
    with pytest.raises(ValueError, match="group IDs overlap"):
        preparation.prepare_public(images, masks, output, **options)
    assert (output / ".incomplete").is_file()
    assert not (output / "curriculum.json").exists()
    assert not json.loads((output / "report.json").read_text())["complete"]


@pytest.mark.parametrize("bad", ["rgb", "missing", "size", "empty_labels"])
def test_invalid_inputs_do_not_publish_curriculum(tmp_path, bad):
    images, masks = dataset(tmp_path / "bdd")
    mask = next((masks / "train").glob("*.png"))
    if bad == "rgb":
        Image.new("RGB", (32, 24)).save(mask)
    elif bad == "size":
        Image.new("L", (33, 24), 0).save(mask)
    elif bad == "missing":
        mask.unlink()
    else:
        ci, cm = dataset(tmp_path / "city", city=True)
        for path in cm.rglob("*.png"):
            Image.new("L", (32, 24), 255).save(path)
    output = tmp_path / "prepared"
    options = {"cityscapes_images": ci, "cityscapes_labels": cm} if bad == "empty_labels" else {}
    with pytest.raises((ValueError, FileNotFoundError)):
        preparation.prepare_public(images, masks, output, **options)
    assert (output / ".incomplete").exists()
    assert not (output / "curriculum.json").exists()


def test_optional_cityscapes_requires_both_roots(tmp_path):
    images, masks = dataset(tmp_path / "bdd")
    with pytest.raises(ValueError, match="both"):
        preparation.prepare_public(images, masks, tmp_path / "out", cityscapes_images=images)


def test_output_inside_original_dataset_rejected(tmp_path):
    images, masks = dataset(tmp_path / "bdd")
    with pytest.raises(ValueError, match="separate"):
        preparation.prepare_public(images, masks, images / "converted")


def test_legacy_bdd_background_is_not_learned_as_road(tmp_path):
    images, masks = dataset(tmp_path / "bdd")
    labels = np.full((24, 32), 1, np.uint8)
    labels[:8] = 0
    labels[:, 0] = 2
    for mask in masks.rglob("*.png"):
        Image.fromarray(labels).save(mask)
    output = tmp_path / "prepared"
    report = preparation.prepare_public(images, masks, output, bdd_encoding="legacy_2018", workers=2)
    row = read_manifest(output / "road/bdd_train/manifest.jsonl")[0]
    converted = np.asarray(Image.open(row["target"]))
    assert converted[0, 10] == 1  # Legacy 0/background -> not_road_surface
    assert converted[15, 10] == 0  # Legacy 1/direct -> road_surface
    assert converted[15, 0] == 0  # Legacy 2/alternative -> road_surface
    assert report["bdd_input_encoding"] == "legacy_2018"
    assert report["preparation_workers"] == 2
    assert (output / "label_maps/bdd_2018_to_road.json").is_file()
