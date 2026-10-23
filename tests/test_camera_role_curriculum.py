import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from camera_rc.config import load_config
from camera_rc.data import load_data_sources, read_manifest, build_validation_loaders
from camera_rc.training import validate_config
import camera_rc.training as training


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_camera_role_curriculum", PROJECT / "scripts/prepare_camera_role_curriculum.py")
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(content), encoding="utf-8")
    return path


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def row(root, name, kind, split, **metadata):
    root.mkdir(parents=True, exist_ok=True)
    image, target = root / f"{name}.png", root / f"{name}_labels.png"
    Image.new("RGB", (32, 24), (10, 20, 30)).save(image)
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[0] = 1
    mask[1, 0] = 255
    Image.fromarray(mask).save(target)
    dataset = preparation.DATASETS.get(kind, "vehicle_actual")
    result = {"image": str(image), "target": str(target), "task": "segmentation",
              "label_space_id": "road_surface_v1", "dataset_id": dataset,
              "group_id": f"{kind}_{split}", "origin_id": f"{kind}:{split}", "split": split}
    if kind == "vehicle":
        result.update(clip_id=f"vehicle_{split}", origin_id=f"vehicle_actual:vehicle_{split}",
                      rig_id="proposed_eight_exterior_cameras_v1", condition="day_clear")
    result.update(metadata)
    return result


def dataset(tmp_path, kind, split_items, name=None, weight=1):
    name = name or kind
    manifests = {}
    originals = {}
    for split, items in split_items.items():
        originals[split] = [row(tmp_path / name, f"{split}_{i}", kind, split, **item) for i, item in enumerate(items)]
        path = tmp_path / name / f"{split}.jsonl"
        write_rows(path, originals[split])
        manifests[split] = str(path)
    return {"name": name, "kind": kind, "weight": weight, "manifests": manifests}, originals


def prepare(tmp_path, sources, **options):
    source = write(tmp_path / "spec.json", {"sources": sources})
    return preparation.prepare_curriculum(source, tmp_path / "output", **options)


def test_public_data_stays_actual_views_and_never_claims_eight_rig_roles(tmp_path):
    bdd, _ = dataset(tmp_path, "bdd", {"train": [{}, {}], "val": [{}]})
    views = [{"camera_id": camera} for camera in sorted(preparation.A2D2_VIEWS)]
    a2d2, originals = dataset(tmp_path, "a2d2", {"train": views, "val": views, "test": [{"camera_id": "cam_front_center"}]})
    report = prepare(tmp_path, [bdd, a2d2])
    assert report["samples"] == {"train": 8, "val": 7, "test": 1}
    assert report["complete"] and report["image_masks_decoded_and_validated"]
    assert not report["all_roles_have_real_rig_train_and_val_samples"]
    assert len(report["missing_real_rig_train_val"]) == 16
    assert not report["synchronized_multiview_samples_created"]
    assert not report["models_validated_for_control"]
    config = load_config(report["stage"])
    validate_config(config)
    train, val = load_data_sources(config)
    assert sum(len(source["rows"]) for source in train) == 8
    observed_val = [r["image"] for source in val for r in source["rows"]]
    assert len(observed_val) == len(set(observed_val)) == 7
    assert all("camera_role" not in r for source in train + val for r in source["rows"])
    actual_views = {r["source_view"] for source in train for r in source["rows"] if r["dataset_id"] == "a2d2"}
    assert actual_views == preparation.A2D2_VIEWS
    assert not any(r["image"] == originals["test"][0]["image"] for source in train + val for r in source["rows"])
    assert len(build_validation_loaders(config)) == 7
    assert all(r["origin_id"] == "a2d2:val" for source in val for r in source["rows"] if r["dataset_id"] == "a2d2")


def test_weights_balance_each_source_then_actual_view_and_condition(tmp_path):
    bdd, _ = dataset(tmp_path, "bdd", {"train": [{}] * 9, "val": [{}]}, weight=2)
    examples = [{"camera_id": camera, "variant": variant} for camera in (0, 1) for variant in ("rain", "fog")]
    vk, _ = dataset(tmp_path, "vkitti2", {"train": examples, "val": examples})
    report = prepare(tmp_path, [bdd, vk])
    train, _ = load_data_sources(load_config(report["stage"]))
    assert sorted(s["weight"] for s in train) == [.25, .25, .25, .25, 2]
    assert sum(s["weight"] for s in train) == 3
    assert report["samples_per_epoch"] == 13
    # The sampler is explicit replacement sampling, not an assertion all rows are seen.
    assert "With replacement" in report["sampling_policy"]


def test_existing_evaluator_reports_bad_condition_separately(tmp_path, monkeypatch):
    examples = [{"condition": "rain"}, {"condition": "fog"}]
    source, rows = dataset(tmp_path, "acdc", {"train": examples, "val": examples})
    Image.new("L", (32, 24), 0).save(rows["val"][0]["target"])
    Image.new("L", (32, 24), 1).save(rows["val"][1]["target"])
    report = prepare(tmp_path, [source])
    config = load_config(report["stage"])
    config["training"].update(device="cpu", num_workers=0)

    class AlwaysRoad(torch.nn.Module):
        def forward(self, image, state=None):
            logits = torch.zeros((image.shape[0], 2, *image.shape[2:]), device=image.device)
            logits[:, 0] = 1
            return logits

    monkeypatch.setattr(training, "build_model", lambda config: AlwaysRoad())
    checkpoint = tmp_path / "fixture.pt"
    torch.save({"format_version": 1, "config": config, "model": {}}, checkpoint)
    metrics = training.evaluate_sources(config, checkpoint)
    assert len(metrics) == 2
    by_condition = {name.rsplit("__", 1)[1]: value for name, value in metrics.items()}
    assert by_condition["rain"]["pixel_accuracy"] == 1
    assert by_condition["fog"]["pixel_accuracy"] == 0
    assert by_condition["fog"]["confusion_matrix"][1][0] == 192 * 320
    assert all(value["batches"] == 1 and not value["partial_validation"] for value in metrics.values())


def test_actual_vehicle_roles_need_both_splits_and_original_clip_identity(tmp_path):
    examples = [{"camera_role": role} for role in preparation.ROLES]
    source, originals = dataset(tmp_path, "vehicle", {"train": examples, "val": examples, "test": examples})
    report = prepare(tmp_path, [source], require_vehicle_coverage=True)
    assert report["all_roles_have_real_rig_train_and_val_samples"]
    assert not report["models_validated_for_control"]
    assert not report["synchronized_multiview_samples_created"]
    assert all(report["rig_role_coverage"][role]["val"]["samples"] == 1 for role in preparation.ROLES)
    for entry in report["manifests"]:
        assert read_manifest(entry["manifest"])[0]["origin_id"] == originals[entry["split"]][0]["origin_id"]


def test_missing_rig_role_is_gap_not_filled_from_front_images(tmp_path):
    bdd, _ = dataset(tmp_path, "bdd", {"train": [{}], "val": [{}]})
    vehicle, _ = dataset(tmp_path, "vehicle", {"train": [{"camera_role": "front_main"}], "val": [{"camera_role": "front_main"}]})
    with pytest.raises(ValueError, match="val/rear"):
        prepare(tmp_path, [bdd, vehicle], require_vehicle_coverage=True)
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("changed_split", ["val", "test"])
def test_same_original_scene_cannot_cross_cameras_or_conditions_into_holdout(tmp_path, changed_split):
    examples = {s: [{"camera_id": 0 if s == "train" else 1, "variant": "fog" if s == "train" else "rain"}]
                for s in ("train", "val", "test")}
    source, rows = dataset(tmp_path, "vkitti2", examples)
    rows[changed_split][0]["origin_id"] = rows["train"][0]["origin_id"]
    write_rows(Path(source["manifests"][changed_split]), rows[changed_split])
    with pytest.raises(ValueError, match="origin leakage"):
        prepare(tmp_path, [source])
    assert not (tmp_path / "output").exists()


def test_duplicate_image_in_different_slice_is_not_double_counted(tmp_path):
    source, rows = dataset(tmp_path, "a2d2", {"train": [{"camera_id": "cam_front_left"}, {"camera_id": "cam_front_right"}],
                                             "val": [{"camera_id": "cam_front_center"}]})
    rows["train"][1]["image"] = rows["train"][0]["image"]
    write_rows(Path(source["manifests"]["train"]), rows["train"])
    with pytest.raises(ValueError, match="Duplicate image"):
        prepare(tmp_path, [source])


@pytest.mark.parametrize("kind,metadata,match", [
    ("bdd", {"camera_role": "front_main"}, "Public data"),
    ("a2d2", {"camera_id": "front_near"}, "camera_id"),
    ("vkitti2", {"camera_id": True, "variant": "fog"}, "camera_id"),
    ("vkitti2", {"camera_id": 0, "variant": "night"}, "variant"),
    ("acdc", {"condition": "sunny"}, "condition"),
    ("vehicle", {"camera_role": "cabin"}, "camera_role"),
    ("vehicle", {"camera_role": "rear", "group_id": "rear-frame"}, "original clip_id"),
    ("vehicle", {"camera_role": "rear", "rig_id": "another_rig"}, "rig_id"),
    ("bdd", {"label_space_id": "raw_bdd"}, "explicitly remapped"),
    ("bdd", {"dataset_id": "a2d2"}, "dataset_id"),
])
def test_unknown_or_fabricated_semantics_are_rejected(tmp_path, kind, metadata, match):
    source, _ = dataset(tmp_path, kind, {"train": [metadata], "val": [metadata]})
    with pytest.raises(ValueError, match=match):
        prepare(tmp_path, [source])


def test_conditions_not_available_in_source_are_not_invented(tmp_path):
    source, _ = dataset(tmp_path, "bdd", {"train": [{}], "val": [{}]})
    report = prepare(tmp_path, [source])
    assert {m["condition"] for m in report["manifests"]} == {"unlabelled_condition"}
    assert not (tmp_path / "output/.incomplete").exists()


def test_invalid_mask_is_rejected_before_committing_output(tmp_path):
    source, rows = dataset(tmp_path, "bdd", {"train": [{}], "val": [{}]})
    Image.new("L", (32, 24), 2).save(rows["train"][0]["target"])
    with pytest.raises(ValueError, match="single-channel"):
        prepare(tmp_path, [source])
    assert not (tmp_path / "output").exists()


def test_metadata_only_mode_does_not_claim_image_validation(tmp_path):
    source, _ = dataset(tmp_path, "bdd", {"train": [{}], "val": [{}]})
    report = prepare(tmp_path, [source], validate_images=False)
    assert not report["image_masks_decoded_and_validated"]


def test_incomplete_source_is_not_consumed(tmp_path):
    source, _ = dataset(tmp_path, "bdd", {"train": [{}], "val": [{}]})
    (tmp_path / "bdd/.incomplete").write_text("incomplete")
    with pytest.raises(ValueError, match="incomplete preparation"):
        prepare(tmp_path, [source])


def test_existing_output_is_never_overwritten(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    marker = output / "keep"
    marker.write_text("keep")
    with pytest.raises(FileExistsError):
        preparation.prepare_curriculum(tmp_path / "missing", output)
    assert marker.read_text() == "keep"


@pytest.mark.parametrize("budget", [0, True, 1.5, 4])
def test_explicit_sampling_budget_is_positive_and_bounded_by_source_rows(tmp_path, budget):
    source, _ = dataset(tmp_path, "bdd", {"train": [{}, {}], "val": [{}]})
    path = write(tmp_path / "spec.json", {"sources": [source], "samples_per_epoch": budget})
    with pytest.raises(ValueError, match="samples_per_epoch"):
        preparation.prepare_curriculum(path, tmp_path / "output")


def test_shipped_rig_is_a_proposal_with_all_control_validation_false():
    rig = json.loads((PROJECT / "configs/camera_rig_8_roles.json").read_text())
    assert tuple(role["id"] for role in rig["roles"]) == preparation.ROLES
    assert not rig["models_validated_for_control"]
    assert not rig["calibration_validated"]
    assert not rig["capture_synchronization_validated"]
    assert not rig["tesla_replica"]
