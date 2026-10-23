import hashlib
import importlib.util
import json
from pathlib import Path

from PIL import Image
import pytest
import torch

from camera_rc.config import load_config
from camera_rc.data import build_dataloaders, read_manifest


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_large_data_stage", PROJECT / "scripts/prepare_large_data_stage.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def make_row(folder, identity, origin, dataset):
    folder.mkdir(parents=True, exist_ok=True)
    image, mask = folder / f"{identity}.jpg", folder / f"{identity}.png"
    Image.new("RGB", (16, 16), (25, 60, 100)).save(image)
    Image.new("L", (16, 16), 0).save(mask)
    return {"image": str(image), "target": str(mask), "group_id": origin, "origin_id": origin,
            "dataset_id": dataset, "label_space_id": "road_surface_v1", "task": "segmentation"}


def write_manifest(root, key, rows, path=None):
    path = path or root / (key + ".jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report_path = root / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else {"complete": True, "label_space_id": "road_surface_v1", "manifests": {}}
    report["manifests"][key] = {"manifest": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "samples": len(rows)}
    write_json(report_path, report)
    return path


@pytest.fixture
def stage(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "BDD_TRAIN_COUNT", 2)
    monkeypatch.setattr(preparation, "BDD_VAL_COUNT", 1)
    project = tmp_path / "project"
    bdd = project / "data/public_v2"
    bdd_train = [make_row(bdd / "images", f"btrain{i}", f"bdd:train{i}", "bdd100k_drivable") for i in range(2)]
    bdd_val = [make_row(bdd / "images", "bval", "bdd:val", "bdd100k_drivable")]
    train_path = write_manifest(bdd, "bdd_train", bdd_train, bdd / "road/bdd_train/manifest.jsonl")
    val_path = write_manifest(bdd, "bdd_val", bdd_val, bdd / "road/bdd_val/manifest.jsonl")
    config = {"task": "segmentation", "model": {"num_classes": 2, "width": 16},
              "data": {"label_space_id": "road_surface_v1", "image_size": [192, 320],
                       "train_manifest": str(train_path), "val_manifests": [{"name": "bdd", "manifest": str(val_path)}]},
              "training": {"output_dir": str(project / "runs/bdd100k_2018_baseline_v1"), "epochs": 1}}
    write_json(project / "configs/local/bdd2018_baseline_v1.json", config)
    checkpoint_path = project / "runs/bdd100k_2018_baseline_v1/best.pt"
    checkpoint_path.parent.mkdir(parents=True)
    checkpoint = {"format_version": 1, "epoch": 0, "epoch_complete": True, "diagnostic": False,
                  "initialization": None, "validation": {"partial_validation": False}, "config": config,
                  "model": {}, "optimizer_steps": 1}
    torch.save(checkpoint, checkpoint_path)
    new = project / "data/vkitti"
    train = [make_row(new / "images", f"newtrain{i}", "kitti:source20", "vkitti2") for i in range(2)]
    val = [make_row(new / "images", "newval", "kitti:source01", "vkitti2")]
    test = [make_row(new / "images", "newtest", "kitti:source02", "vkitti2")]
    for split, rows in (("train", train), ("val", val), ("test", test)):
        write_manifest(new, split, rows)
    old = project / "data/old_kitti"
    # Old training is deliberately related to a held-out scene and must never
    # enter this new stage. Real+synthetic versions within val are permitted.
    write_manifest(old, "train", [make_row(old / "images", "oldtrain", "kitti:source02", "kitti_road")])
    write_manifest(old, "val", [make_row(old / "images", "oldval", "kitti:source01", "kitti_road")])
    return {"project": project, "name": "large_v1", "source_root": [new],
            "validation_root": [old], "init_from": checkpoint_path}


def test_full_bdd_all_new_once_native_loader_and_reserved_test_exclusion(stage):
    report = preparation.prepare_large_data_stage(**stage)
    config_path = Path(report["config"])
    config = load_config(config_path)
    rows = read_manifest(config["data"]["train_manifest"])
    assert len(rows) == 4 and all("sample_id" not in row for row in rows)
    assert report["train_counts"] == {"bdd100k_drivable": 2, "vkitti2": 2}
    assert report["validation_counts"] == {"bdd100k_drivable": 1, "vkitti2": 1, "kitti_road": 1}
    assert report["reserved_test_counts"] == {"vkitti2": 1}
    assert report["complete"] and not report["training_started"] and not report["deployed"]
    assert report["all_stage_images_once_per_epoch"] and not report["sampling_with_replacement"]
    assert not report["diagnostic"] and not report["test_used_for_training_or_validation"]
    assert config["training"]["epochs"] == 1
    assert config["training"]["segmentation_class_weights"] == [1.0, 2.0]
    assert config["training"]["lr"] == 0.0002 and config["training"]["amp"] is True
    assert config["training"]["batch_size"] == 16 and config["training"]["num_workers"] == 2
    assert "train_sources" not in config["data"] and "max_train_steps" not in config["training"]
    assert all("test.jsonl" not in source["manifest"] for source in config["data"]["val_manifests"])
    assert report["config_sha256"] == preparation._sha256(config_path)
    assert report["initial_checkpoint"]["sha256"] == preparation._sha256(stage["init_from"])
    assert report["train_manifest_sha256"] == preparation._sha256(config["data"]["train_manifest"])
    assert not (stage["project"] / "data/large_v1/.incomplete").exists()
    train_loader, _ = build_dataloaders(config)
    assert sorted(list(train_loader.sampler)) == list(range(4))
    assert not train_loader.drop_last


@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_corrupted_source_manifest_hash_is_rejected(stage, split):
    path = stage["source_root"][0] / f"{split}.jsonl"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="SHA256"):
        preparation.prepare_large_data_stage(**stage)
    assert not (stage["project"] / "data/large_v1").exists()


@pytest.mark.parametrize("which", ["new_marker", "new_report", "old_marker", "bdd_marker", "bdd_nested_marker", "sample_count", "report_label"])
def test_incomplete_or_inconsistent_prepared_inputs_rejected(stage, which):
    new, old = stage["source_root"][0], stage["validation_root"][0]
    bdd = stage["project"] / "data/public_v2"
    if which.endswith("marker"):
        folder = {"new_marker": new, "old_marker": old, "bdd_marker": bdd, "bdd_nested_marker": bdd / "road/bdd_train"}[which]
        (folder / ".incomplete").write_text("incomplete")
    else:
        path = new / "report.json"
        report = json.loads(path.read_text())
        if which == "new_report":
            report["complete"] = False
        elif which == "sample_count":
            report["manifests"]["train"]["samples"] += 1
        else:
            report["label_space_id"] = "wrong"
        write_json(path, report)
    with pytest.raises(ValueError):
        preparation.prepare_large_data_stage(**stage)


@pytest.mark.parametrize("problem", ["origin_train_val", "origin_train_test", "origin_val_test", "path_train_val", "repeated_train", "source_alias", "missing_origin", "wrong_label"])
def test_cross_dataset_path_and_origin_leaks_rejected(stage, problem):
    root = stage["source_root"][0]
    train, val, test = [read_manifest(root / f"{s}.jsonl") for s in ("train", "val", "test")]
    if problem == "origin_train_val":
        train[0]["origin_id"] = val[0]["origin_id"]
    elif problem == "origin_train_test":
        train[0]["origin_id"] = test[0]["origin_id"]
    elif problem == "origin_val_test":
        val[0]["origin_id"] = test[0]["origin_id"]
    elif problem == "path_train_val":
        train[0]["image"] = val[0]["image"]
    elif problem == "repeated_train":
        train.append(train[0].copy())
    elif problem == "source_alias":
        train[0]["source_image"] = test[0]["image"]
    elif problem == "missing_origin":
        train[0].pop("origin_id")
    else:
        train[0]["label_space_id"] = "wrong"
    for split, values in (("train", train), ("val", val), ("test", test)):
        write_manifest(root, split, values)
    with pytest.raises(ValueError):
        preparation.prepare_large_data_stage(**stage)


@pytest.mark.parametrize("problem", ["diagnostic", "partial_epoch", "partial_val", "finetuned", "init_chain", "wrong_source", "weighted_sources", "max_steps", "wrong_val", "model"])
def test_non_original_bdd_checkpoint_rejected(stage, problem):
    path = stage["init_from"]
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if problem == "diagnostic":
        checkpoint["diagnostic"] = True
    elif problem == "partial_epoch":
        checkpoint["epoch_complete"] = False
    elif problem == "partial_val":
        checkpoint["validation"]["partial_validation"] = True
    elif problem == "finetuned":
        checkpoint["initialization"] = {"path": "old.pt"}
    elif problem == "init_chain":
        checkpoint["config"]["training"]["init_from"] = "old.pt"
    elif problem == "wrong_source":
        checkpoint["config"]["data"]["train_manifest"] = str(stage["source_root"][0] / "train.jsonl")
    elif problem == "weighted_sources":
        checkpoint["config"]["data"]["train_sources"] = [{"manifest": "old", "weight": 1}]
    elif problem == "max_steps":
        checkpoint["config"]["training"]["max_train_steps"] = 1
    elif problem == "wrong_val":
        checkpoint["config"]["data"]["val_manifests"][0]["manifest"] = str(stage["source_root"][0] / "val.jsonl")
    else:
        checkpoint["config"]["model"]["width"] = 32
    torch.save(checkpoint, path)
    with pytest.raises(ValueError):
        preparation.prepare_large_data_stage(**stage)


def test_multiple_new_sources_and_unlabelled_test_without_manifest(stage):
    root = stage["project"] / "data/acdc"
    write_manifest(root, "train", [make_row(root / "images", "train", "acdc:train", "acdc")])
    write_manifest(root, "val", [make_row(root / "images", "val", "acdc:val", "acdc")])
    path = root / "report.json"
    report = json.loads(path.read_text())
    report["heldout_test_unlabelled"] = {"count": 2000, "targets_available": False}
    write_json(path, report)
    stage["source_root"].append(root)
    stage["epochs"] = 2
    result = preparation.prepare_large_data_stage(**stage)
    assert result["train_counts"]["acdc"] == 1
    assert result["total_training_samples"] == 5 and result["epochs"] == 2
    assert "acdc" not in result["reserved_test_counts"]
    assert result["source_manifests"][2]["heldout_test_unlabelled"] == {"count": 2000, "targets_available": False}


@pytest.mark.parametrize("reserved", ["data", "config", "run"])
def test_existing_artifacts_never_overwritten(stage, reserved):
    paths = {"data": stage["project"] / "data/large_v1", "config": stage["project"] / "configs/local/large_v1.json", "run": stage["project"] / "runs/large_v1"}
    path = paths[reserved]
    if reserved == "config":
        path.write_text("keep")
    else:
        path.mkdir()
        (path / "keep.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        preparation.prepare_large_data_stage(**stage)
    assert (path if reserved == "config" else path / "keep.txt").read_text() == "keep"


@pytest.mark.parametrize("weights", [[0, 2], [1, float("nan")], [1, float("inf")], [True, 2], [1, 2, 3]])
def test_invalid_loss_weights_rejected(stage, weights):
    with pytest.raises(ValueError):
        preparation.prepare_large_data_stage(**stage, segmentation_class_weights=weights)


def test_repeated_prepared_roots_and_path_escape_names_rejected(stage):
    with pytest.raises(ValueError, match="unique"):
        preparation.prepare_large_data_stage(**{**stage, "validation_root": stage["source_root"]})
    with pytest.raises(ValueError, match="Stage name"):
        preparation.prepare_large_data_stage(**{**stage, "name": "../escape"})


def test_full_bdd_count_cannot_be_silently_reduced(stage):
    root = stage["project"] / "data/public_v2"
    path = root / "road/bdd_train/manifest.jsonl"
    write_manifest(root, "bdd_train", read_manifest(path)[:1], path)
    with pytest.raises(ValueError, match="full original BDD"):
        preparation.prepare_large_data_stage(**stage)


def test_legacy_bdd_clip_origins_are_namespaced_only_in_new_stage_copy(stage):
    root = stage["project"] / "data/public_v2"
    original_bytes = {}
    for split in ("train", "val"):
        path = root / f"road/bdd_{split}/manifest.jsonl"
        rows = read_manifest(path)
        for row in rows:
            row.pop("origin_id")
        write_manifest(root, "bdd_" + split, rows, path)
        original_bytes[path] = path.read_bytes()
    report = preparation.prepare_large_data_stage(**stage)
    stage_rows = read_manifest(stage["project"] / "data/large_v1/train.jsonl")
    for row in stage_rows:
        if row["dataset_id"] == "bdd100k_drivable":
            assert row["origin_id"] == "bdd100k:" + row["group_id"]
    assert all(path.read_bytes() == contents for path, contents in original_bytes.items())
    assert report["legacy_bdd_origin_normalization"]
