from copy import deepcopy
import hashlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from aihub_preprocess_split import partition_rows


def row(index, dataset="bdd100k_drivable", group=None):
    group = group or f"clip{index}"
    result = {"dataset_id": dataset, "group_id": group, "origin_id": "bdd100k:" + group,
              "image": f"D:/fixture/images/{index}.jpg", "source_image": f"D:/fixture/raw/{index}.jpg",
              "target": f"D:/fixture/masks/{index}.png", "split": "train", "export_split": "train",
              "image_sha256": hashlib.sha256(f"bytes{index}".encode()).hexdigest(),
              "image_rgb_sha256": hashlib.sha256(f"rgb{index}".encode()).hexdigest(), "sample_id": str(index)}
    if dataset.startswith("aihub_"):
        result.update(source_clip=group, source_split="Training", origin_id="aihub_commercial:" + group)
    if dataset == "vkitti2_0_3":
        result.update(source_scene=group, source_video="drive_" + group,
                      group_id="kitti:drive_" + group, origin_id="kitti:drive_" + group,
                      camera_id=index % 2, variant="clone" if index % 2 else "rain", frame_id=index)
    return result


def test_deterministic_complete_nonmutating_partition():
    rows = [row(i) for i in range(100)]
    before = deepcopy(rows)
    train, dev, report = partition_rows(rows)
    assert rows == before
    assert len(train) == 90 and len(dev) == 10
    assert all(r["split"] == "dev" for r in dev)
    assert all(r["export_split"] == "train" for r in dev)
    assert (train, dev, report) == partition_rows(list(reversed(rows)))
    assert partition_rows(rows, seed=7)[1] != dev
    assert report["split_identity_checks"]["group_origin_path_byte_rgb_overlap"] == 0
    assert report["official_validation_modified"] is False


def test_vkitti_whole_scene_not_weather_camera_or_frame():
    rows = [row(i, "vkitti2_0_3", scene) for i, scene in enumerate(["Scene06"]*18 + ["Scene18"]*24 + ["Scene20"]*58)]
    train, dev, report = partition_rows(rows)
    assert len(dev) == 18
    assert {r["source_scene"] for r in dev} == {"Scene06"}
    assert not {r["source_video"] for r in train} & {r["source_video"] for r in dev}
    assert report["actual_dev_fraction"] == .18


def test_aihub_equipment_day_across_datasets_is_atomic():
    rows = []
    for day in range(1, 11):
        for ds in ("aihub_13", "aihub_14", "aihub_15"):
            for hour in (8, 9):
                rows.append(row(len(rows), ds, f"16_{hour:02d}3000_2210{day:02d}"))
    train, dev, report = partition_rows(rows)
    def days(items): return {r["source_clip"].split("_")[-1] for r in items}
    assert not days(train) & days(dev)
    assert len(dev) == 6
    assert len(report["selected_groups"]) == 1
    assert set(report["selected_groups"][0]["by_dataset"]) == {"aihub_13", "aihub_14", "aihub_15"}
    assert (train, dev, report) == partition_rows(list(reversed(rows)))


@pytest.mark.parametrize("field", ["image_sha256", "image_rgb_sha256", "source_image"])
def test_duplicate_chain_cannot_cross_split(field):
    rows = [row(i) for i in range(30)]
    rows[1][field] = rows[0][field]
    rows[2]["image_rgb_sha256"] = rows[1]["image_rgb_sha256"]
    train, dev, _ = partition_rows(rows)
    assigned = {r["sample_id"]: r["split"] for r in train+dev}
    assert assigned["0"] == assigned["1"] == assigned["2"]


def test_target_class_counts_never_influence_assignment():
    rows = [row(i) for i in range(50)]
    original = partition_rows(rows)
    for r in rows:
        r["target_pixel_counts"] = {"0": 0, "1": 99, "255": 99999}
        r["metrics"] = {"iou": 0.9999}
    current = partition_rows(rows)
    assert [r["sample_id"] for r in original[1]] == [r["sample_id"] for r in current[1]]


@pytest.mark.parametrize("field,value", [("split", "val"), ("export_split", "val"),
    ("image", "relative.jpg"), ("target", "relative.png"), ("image_sha256", None),
    ("image_rgb_sha256", "bad"), ("origin_id", "wrong"), ("dataset_id", "unrecognized")])
def test_invalid_inputs_fail_closed(field, value):
    rows = [row(i) for i in range(10)]
    rows[0][field] = value
    with pytest.raises(ValueError):
        partition_rows(rows)


def test_duplicate_image_path_is_not_silently_removed():
    rows = [row(i) for i in range(10)]
    rows[1]["image"] = rows[0]["image"]
    with pytest.raises(ValueError, match="Repeated image path"):
        partition_rows(rows)


def test_single_indivisible_group_refused():
    with pytest.raises(ValueError, match="at least two"):
        partition_rows([row(i, "vkitti2_0_3", "Scene06") for i in range(10)])


@pytest.mark.parametrize("seed,fraction", [(True,.1),(42,0),(42,.5),(42,float("nan")),(42,True)])
def test_bad_selection_options(seed, fraction):
    with pytest.raises(ValueError):
        partition_rows([row(i) for i in range(10)], seed, fraction)


def test_aihub_invalid_date_or_nontraining_rejected():
    for group, split in (("16_080000_221099", "Training"), ("16_080000_221001", "Validation")):
        rows = [row(i, "aihub_13", group) for i in range(2)]
        rows[0]["source_split"] = split
        with pytest.raises(ValueError):
            partition_rows(rows)


def test_exact_duplicate_across_dataset_stays_one_component():
    rows = [row(i) for i in range(50)]
    rows += [row(100+i, "aihub_13", f"16_080000_2210{i+1:02d}") for i in range(10)]
    rows[-1]["image_rgb_sha256"] = rows[0]["image_rgb_sha256"]
    train, dev, report = partition_rows(rows)
    result = {r["sample_id"]: r for r in train+dev}
    assert result["0"]["development_group_id"] == result["109"]["development_group_id"]
    assert result["0"]["split"] == result["109"]["split"]
    assert report["duplicate_hash_links"]["rgb"] == 1


def test_repeated_normalized_absolute_path_rejected():
    rows = [row(i) for i in range(10)]
    rows[0]["image"] = "/fixture/b/image.jpg"
    rows[1]["image"] = "/fixture/a/../b/image.jpg"
    with pytest.raises(ValueError, match="Repeated image path"):
        partition_rows(rows)
