import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("build_aihub_cohort", Path(__file__).parents[1] / "scripts/build_aihub_cohort.py")
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def row(i, group="g", ds=mod.VKITTI, scene="Scene06", variant="clone", camera=0):
    return {"image": f"image{i}.jpg", "target": f"mask{i}.png", "dataset_id": ds,
            "group_id": group, "origin_id": "origin:" + group,
            "sample_id": str(i), "source_scene": scene, "variant": variant,
            "camera_id": camera, "frame_id": i, "task": "segmentation", "label_space_id": "road_surface_v1"}


def test_selection_order_independent_and_preserves_identity():
    rows = [row(i) for i in range(30)]
    a, stats = mod.select_replay(rows, 9, 42, True)
    b, other = mod.select_replay(list(reversed(rows)), 9, 42, True)
    assert a == b and stats == other
    assert len({mod.row_identity(r) for r in a}) == 9
    assert all(r in rows and r["origin_id"] == "origin:g" for r in a)
    assert mod.select_replay(rows, 9, 43, True)[0] != a


def test_proportional_strata_exact_count():
    rows = [row(i, scene="large") for i in range(7)] + [row(i + 7, scene="small") for i in range(3)]
    selected, stats = mod.select_replay(rows, 5, 42, True)
    assert len(selected) == 5
    assert sum(s["selected"] for s in stats) == 5
    assert all(abs(s["selected"] - s["available"] / 2) <= 0.5 for s in stats)


@pytest.mark.parametrize("count", [0, -1, 11, True])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        mod.select_replay([row(i) for i in range(10)], count, 42)


def test_duplicate_or_missing_stratum():
    with pytest.raises(ValueError, match="Duplicate"):
        mod.select_replay([row(1), row(1)], 1, 42)
    bad = row(1)
    del bad["variant"]
    with pytest.raises(ValueError, match="metadata"):
        mod.select_replay([bad], 1, 42, True)


def dump(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def manifest(path, rows):
    for i, r in enumerate(rows):
        for field in ("image", "target"):
            file = path.parent / (path.stem + str(i) + (".jpg" if field == "image" else ".png"))
            file.write_bytes(b"metadata fixture only")
            r[field] = str(file)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def setup(tmp_path):
    old_train = [row(i, "bdd" + str(i), mod.BDD) for i in range(3)] + [row(i + 10, "vktrain") for i in range(3)]
    old_val = [row(30, "bddval", mod.BDD), row(31, "vkval")]
    manifest(tmp_path / "old_train.jsonl", old_train)
    manifest(tmp_path / "old_val.jsonl", old_val)
    config = tmp_path / "baseline.json"
    dump(config, {"task": "segmentation", "model": {"num_classes": 2},
                  "data": {"label_space_id": "road_surface_v1", "image_size": [192, 320],
                           "train_manifest": "old_train.jsonl", "val_manifest": "old_val.jsonl"},
                  "training": {"init_from": "should_not_copy.pt", "epochs": 1}})
    ai = tmp_path / "aihub"
    ai.mkdir()
    manifests = {}
    for split in ("train", "val"):
        r = row(100 if split == "train" else 200, "ai" + split, "aihub_13")
        r.update(source_dataset_number=13, split=split, source_split="Training" if split == "train" else "Validation")
        p = ai / (split + ".jsonl")
        manifest(p, [r])
        manifests[split] = {"manifest": str(p), "sha256": mod.sha256(p), "samples": 1, "all_samples_decoded": True}
    report = {"complete": True, "diagnostic": False, "format": "camera_rc_aihub_road_v1", "label_space_id": "road_surface_v1",
              "cross_source_clip_leakage_checked": True, "exact_and_decoded_rgb_duplicates_checked": True, "manifests": manifests}
    dump(ai / "report.json", report)
    dump(ai / ".complete", {"report_sha256": mod.sha256(ai / "report.json")})
    return config, ai


def test_build_preserves_all_holdouts_and_input_indices(tmp_path):
    config, ai = setup(tmp_path)
    output = tmp_path / "cohort"
    report = mod.build(config, ai, output, 2, 2, 42)
    assert report["train"]["samples"] == 5
    assert report["val"]["samples"] == 3
    assert report["all_prior_training_rows_checked"] == 6
    assert not (output / ".incomplete").exists()
    new = json.loads((output / "config.json").read_text())
    assert "init_from" not in new["training"]
    assert new["training"]["epochs"] == 3
    val = [json.loads(x) for x in (output / "val.jsonl").read_text().splitlines()]
    assert {r["group_id"] for r in val} == {"bddval", "vkval", "aival"}
    assert all("cohort_input_manifest_sha256" in r for r in val)
    with pytest.raises(FileExistsError):
        mod.build(config, ai, output, 2, 2, 42)


def test_aihub_tamper_rejected(tmp_path):
    config, ai = setup(tmp_path)
    with (ai / "train.jsonl").open("a") as f:
        f.write("\n")
    with pytest.raises(ValueError, match="differs"):
        mod.build(config, ai, tmp_path / "cohort", 2, 2)


def test_full_old_split_checked_before_replay_selection(tmp_path):
    config, ai = setup(tmp_path)
    p = tmp_path / "old_val.jsonl"
    rows = [json.loads(x) for x in p.read_text().splitlines()]
    rows[0]["origin_id"] = "origin:bdd2"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="leakage"):
        mod.build(config, ai, tmp_path / "cohort", 1, 1)


def test_unsupported_dataset12_excluded(tmp_path):
    config, ai = setup(tmp_path)
    p = ai / "train.jsonl"
    r = json.loads(p.read_text())
    r["source_dataset_number"] = 12
    p.write_text(json.dumps(r) + "\n", encoding="utf-8")
    report = json.loads((ai / "report.json").read_text())
    report["manifests"]["train"]["sha256"] = mod.sha256(p)
    dump(ai / "report.json", report)
    dump(ai / ".complete", {"report_sha256": mod.sha256(ai / "report.json")})
    with pytest.raises(ValueError, match="Only supported"):
        mod.build(config, ai, tmp_path / "cohort", 2, 2)
