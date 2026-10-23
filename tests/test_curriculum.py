from copy import deepcopy
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from camera_rc.config import load_config
from camera_rc.curriculum import run_curriculum
from camera_rc.data import build_dataloaders, load_data_sources, remap_masks, prepare_cityscapes, read_manifest
from camera_rc.training import run_training


@pytest.fixture(autouse=True)
def limited_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")
    return path


def manifest(root, name, count=2, group=None, origin=None):
    rows = []
    for index in range(count):
        image = root / f"{name}-{index}.png"
        mask = root / f"{name}-{index}-mask.png"
        pixels = np.zeros((16, 20, 3), dtype=np.uint8)
        pixels[:, :, index % 3] = 150
        Image.fromarray(pixels).save(image)
        Image.fromarray(np.full((16, 20), index % 3, np.uint8)).save(mask)
        row = {"image": str(image), "target": str(mask), "dataset_id": name,
               "group_id": group or name, "task": "segmentation"}
        if origin is not None:
            row["origin_id"] = origin
        rows.append(row)
    path = root / f"{name}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def config(root, name, train, val):
    return write_json(root / f"{name}.json", {
        "task": "segmentation", "model": {"width": 4, "num_classes": 3},
        "data": {"train_manifest": str(train), "val_manifest": str(val),
                 "label_space_id": "road-three-classes-v1", "image_size": [16, 20]},
        "training": {"output_dir": str(root / f"standalone-{name}"), "epochs": 1,
                     "batch_size": 1, "num_workers": 0, "device": "cpu", "seed": 42,
                     "amp": False, "lr": 0.001}})


def curriculum(root, configs, **overrides):
    value = {"output_dir": "curriculum-run", "stages": [
        {"name": f"stage-{index}", "config": str(path), **({"init_from": "previous"} if index else {})}
        for index, path in enumerate(configs)]}
    value.update(overrides)
    return write_json(root / "curriculum.json", value)


def test_relative_sources_and_weights_only_init_paths(tmp_path):
    path = write_json(tmp_path / "config.json", {"task": "segmentation", "data": {
        "train_sources": [{"manifest": "a.jsonl", "weight": 2}],
        "val_manifests": ["b.jsonl", {"manifest": "c.jsonl", "name": "heldout"}]},
        "training": {"init_from": "weights/best.pt"}})
    loaded = load_config(path)
    assert loaded["data"]["train_sources"][0]["manifest"] == str(tmp_path / "a.jsonl")
    assert loaded["data"]["val_manifests"][1]["manifest"] == str(tmp_path / "c.jsonl")
    assert loaded["training"]["init_from"] == str(tmp_path / "weights" / "best.pt")


def test_weighted_sampler_balances_sources_not_frame_counts(tmp_path):
    first, second, val = manifest(tmp_path, "a", 2), manifest(tmp_path, "b", 6), manifest(tmp_path, "val", 1)
    cfg = load_config(config(tmp_path, "train", first, val))
    cfg["data"].pop("train_manifest")
    cfg["data"]["train_sources"] = [{"manifest": str(first), "weight": 2}, {"manifest": str(second), "weight": 1}]
    cfg["training"]["samples_per_epoch"] = 12
    train, validation = build_dataloaders(cfg)
    assert train.sampler.replacement
    assert train.sampler.num_samples == 12
    assert train.sampler.weights[:2].sum().item() == pytest.approx(2)
    assert train.sampler.weights[2:].sum().item() == pytest.approx(1)
    assert len(validation.dataset) == 1
    assert train.dataset[0]["image"].shape == (3, 16, 20)


@pytest.mark.parametrize("weight", [0, -1, float("nan"), True])
def test_invalid_source_weights_fail(tmp_path, weight):
    train, val = manifest(tmp_path, "a"), manifest(tmp_path, "val", 1)
    cfg = load_config(config(tmp_path, "cfg", train, val))
    cfg["data"].pop("train_manifest")
    cfg["data"]["train_sources"] = [{"manifest": str(train), "weight": weight}]
    with pytest.raises(ValueError, match="weight"):
        load_data_sources(cfg)


def test_global_heldout_origin_catches_derived_dataset_leakage(tmp_path):
    a_train = manifest(tmp_path, "a-train")
    a_val = manifest(tmp_path, "a-val", 1, origin="canonical-source/heldout-drive")
    b_train = manifest(tmp_path, "derived-b-train", origin="canonical-source/heldout-drive")
    b_val = manifest(tmp_path, "b-val", 1)
    plan_path = curriculum(tmp_path, [config(tmp_path, "a", a_train, a_val), config(tmp_path, "b", b_train, b_val)])
    with pytest.raises(ValueError, match="Global cross-stage"):
        run_curriculum(plan_path)
    assert not (tmp_path / "curriculum-run").exists()


def test_plan_readonly_and_semantic_mismatch_fails(tmp_path):
    train, val = manifest(tmp_path, "a"), manifest(tmp_path, "val", 1)
    first = config(tmp_path, "first", train, val)
    second = config(tmp_path, "second", train, val)
    path = curriculum(tmp_path, [first, second])
    plan = run_curriculum(path)
    assert plan["mode"] == "plan" and not plan["training_complete"]
    assert plan["validation"]["global_holdouts_checked"]
    assert not (tmp_path / "curriculum-run").exists()
    modified = json.loads(second.read_text())
    modified["data"]["label_space_id"] = "different-class-order"
    write_json(second, modified)
    with pytest.raises(ValueError, match="label_space_id"):
        run_curriculum(path)


def test_two_stage_training_resets_optimizer_and_rechecks_old_holdouts(tmp_path):
    a, av = manifest(tmp_path, "a"), manifest(tmp_path, "av", 1)
    b, bv = manifest(tmp_path, "b", 3), manifest(tmp_path, "bv", 1)
    path = curriculum(tmp_path, [config(tmp_path, "one", a, av), config(tmp_path, "two", b, bv)])
    result = run_curriculum(path, execute=True)
    assert result["training_complete"]
    first, second = result["stages"]
    assert second["initialization"] == first["checkpoint"]
    assert set(second["holdout_metrics"]) == {str(av), str(bv)}
    initial = torch.load(first["checkpoint"]["path"], weights_only=True)
    trained = torch.load(second["checkpoint"]["path"], weights_only=True)
    assert trained["global_step"] == 3  # New stage, not old 2 + new 3.
    assert trained["optimizer_steps"] == 3
    assert any(not torch.equal(initial["model"][key], trained["model"][key]) for key in initial["model"])
    with pytest.raises(FileExistsError, match="exists"):
        run_curriculum(path, execute=True)
    resumed = run_curriculum(path, execute=True, resume_run=True)
    assert resumed["training_complete"] and len(resumed["stages"]) == 2
    modified = json.loads(path.read_text())
    modified["stages"][1]["name"] = "changed"
    write_json(path, modified)
    with pytest.raises(ValueError, match="changed"):
        run_curriculum(path, execute=True, resume_run=True)


def test_init_requires_semantics_and_cannot_resume(tmp_path):
    train, val = manifest(tmp_path, "a"), manifest(tmp_path, "val", 1)
    cfg = load_config(config(tmp_path, "first", train, val))
    first = run_training(cfg)
    new = deepcopy(cfg)
    new["training"]["output_dir"] = str(tmp_path / "new")
    new["data"]["label_space_id"] = "different"
    with pytest.raises(ValueError, match="label_space_id"):
        run_training(new, init_from=first["best_checkpoint"])
    new = deepcopy(cfg)
    with pytest.raises(ValueError, match="cannot be combined"):
        run_training(new, resume=first["last_checkpoint"], init_from=first["best_checkpoint"])
    new["data"].pop("label_space_id")
    with pytest.raises(ValueError, match="explicit"):
        run_training(new, init_from=first["best_checkpoint"])


def test_partial_stage_stops_chain_without_completion_claim(tmp_path):
    train, val = manifest(tmp_path, "a", 3), manifest(tmp_path, "val", 2)
    first = config(tmp_path, "first", train, val)
    cfg = json.loads(first.read_text())
    cfg["training"]["max_train_steps"] = 1
    write_json(first, cfg)
    second = config(tmp_path, "second", train, val)
    result = run_curriculum(curriculum(tmp_path, [first, second]), execute=True)
    assert not result["training_complete"]
    assert result["stages"][0]["status"] == "incomplete"
    assert len(result["stages"]) == 1
    checkpoint = torch.load(result["stages"][0]["checkpoint"]["path"], weights_only=True)
    assert checkpoint["diagnostic"]


def test_synthetic_provenance_saved_in_checkpoint(tmp_path):
    train, val = manifest(tmp_path, "a"), manifest(tmp_path, "val", 1)
    cfg = load_config(config(tmp_path, "first", train, val))
    cfg["diagnostic_only"] = True
    result = run_training(cfg)
    checkpoint = torch.load(result["last_checkpoint"], weights_only=True)
    assert checkpoint["diagnostic"] and result["diagnostic"]


def test_detection_plan_requires_exact_scene_manifests(tmp_path):
    train, val = manifest(tmp_path, "a"), manifest(tmp_path, "val", 1)
    for manifest_path in (train, val):
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
        for row in rows:
            label = Path(row["image"]).with_suffix(".txt")
            label.write_text("0 0.5 0.5 0.5 0.5\n")
            row.update(target=str(label), task="detection")
        manifest_path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    train_images = [row["image"] for row in map(json.loads, train.read_text().splitlines())]
    val_images = [row["image"] for row in map(json.loads, val.read_text().splitlines())]
    (tmp_path / "train.txt").write_text("\n".join(train_images))
    (tmp_path / "val.txt").write_text("\n".join(val_images))
    dataset = write_json(tmp_path / "dataset.yaml", {"path": str(tmp_path), "train": "train.txt", "val": "val.txt", "names": ["car"]})
    cfg = write_json(tmp_path / "det.json", {"task": "detection", "model": "yolo11n.yaml", "data": str(dataset), "epochs": 1})
    path = curriculum(tmp_path, [], stages=[{"name": "det", "config": str(cfg), "label_space_id": "car-v1",
                                           "train_manifests": [str(train)], "val_manifests": [str(val)]}])
    assert run_curriculum(path)["stages"][0]["samples"] == {"train": 2, "val": 1}
    (tmp_path / "train.txt").write_text(train_images[0])
    with pytest.raises(ValueError, match="exactly"):
        run_curriculum(path)


def test_remap_unknown_ignore_alignment_and_original_identity(tmp_path):
    source = manifest(tmp_path, "original", 1, origin="shared-original-scene")
    row = json.loads(source.read_text().strip())
    pixels = np.full((16, 20), 7, np.uint8)
    pixels[0, :3] = [8, 99, 255]
    Image.fromarray(pixels).save(row["target"])
    mapping = write_json(tmp_path / "map.json", {"label_space_id": "road_surface_v1", "num_classes": 2,
                                               "mapping": {"7": 0, "8": 1, "255": 255}})
    result = remap_masks(source, mapping, tmp_path / "mapped")
    converted = read_manifest(result["manifest"])[0]
    assert converted["image"] == row["image"]
    assert converted["group_id"] == row["group_id"] and converted["origin_id"] == row["origin_id"]
    assert converted["target_original"] == row["target"]
    assert converted["label_space_id"] == "road_surface_v1"
    with Image.open(converted["target"]) as mask:
        assert mask.size == (20, 16)
        assert np.asarray(mask)[0, :4].tolist() == [1, 255, 255, 0]
    assert result["undeclared_ids_mapped_to_ignore"] == {99: 1}
    assert not (tmp_path / "mapped" / ".incomplete").exists()
    with pytest.raises(FileExistsError):
        remap_masks(source, mapping, tmp_path / "mapped")
    Image.new("L", (17, 16)).save(row["target"])
    with pytest.raises(ValueError, match="alignment"):
        remap_masks(source, mapping, tmp_path / "bad_alignment")


def test_remap_preserves_ignore_semantics(tmp_path):
    mapping = write_json(tmp_path / "bad-map.json", {"label_space_id": "road", "num_classes": 2, "mapping": {"255": 0}})
    with pytest.raises(ValueError, match="ignore ID 255"):
        remap_masks(tmp_path / "not-read.jsonl", mapping, tmp_path / "never-created")
    assert not (tmp_path / "never-created").exists()


def test_cityscapes_pairs_labelids_and_retains_sequence(tmp_path):
    rgb_root, mask_root = tmp_path / "leftImg8bit", tmp_path / "gtFine"
    for split, city in (("train", "aachen"), ("val", "frankfurt")):
        (rgb_root / split / city).mkdir(parents=True)
        (mask_root / split / city).mkdir(parents=True)
        for frame in (19, 38):
            prefix = f"{city}_000000_{frame:06d}"
            Image.new("RGB", (20, 16)).save(rgb_root / split / city / f"{prefix}_leftImg8bit.png")
            Image.new("L", (20, 16), 7).save(mask_root / split / city / f"{prefix}_gtFine_labelIds.png")
    result = prepare_cityscapes(rgb_root, mask_root, tmp_path / "cityscapes")
    assert result["train"]["samples"] == 2 and result["train"]["groups"] == 1
    train = read_manifest(tmp_path / "cityscapes" / "train.jsonl")
    assert train[0]["origin_id"] == train[1]["origin_id"] == "cityscapes:aachen_000000"
    assert train[0]["target"].endswith("_gtFine_labelIds.png")
