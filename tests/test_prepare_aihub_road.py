import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from camera_rc.data import read_manifest, validate_splits


@pytest.fixture
def adapter():
    path = Path(__file__).parents[1] / "scripts/prepare_aihub_road.py"
    spec = importlib.util.spec_from_file_location("aihub_adapter_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def polygon(name, x1=0, y1=0, x2=16, y2=12):
    return {"class": name, "polygon": [x1, y1, x2, y1, x2, y2, x1, y2]}


def annotation(name="frame.jpg"):
    return {
        "information": {"filename": name, "resolution": [16, 12]},
        "annotations": [
            polygon("background"), polygon("freespace", 0, 5, 16, 12),
            polygon("whiteLane", 5, 5, 9, 12), polygon("vehicle", 7, 6, 13, 10),
        ],
    }


@pytest.fixture
def source_fixture(tmp_path):
    spec = {"schema_version": 1, "sources": []}
    assets = []
    for number, dataset_id in ((13, "aihub_city"), (14, "aihub_highway")):
        source = {"dataset_number": number, "dataset_id": dataset_id}
        for split, day in (("train", "240101"), ("val", "240102")):
            root = tmp_path / f"추출_{number}_{split}"
            clip = f"{number}_120000_{day}"
            image = root / "1.맑음" / "images" / clip / f"{clip}_01.jpg"
            label = root / "1.맑음" / "labels" / clip / f"{clip}_01.json"
            image.parent.mkdir(parents=True)
            label.parent.mkdir(parents=True)
            Image.new("RGB", (16, 12), (number * 10, 40 if split == "train" else 95, 80)).save(image)
            label.write_text(json.dumps(annotation(image.name)), encoding="utf-8")
            marker = root / "EXTRACTION_COMPLETE.json"
            marker.write_text(json.dumps({"archive_sha256": str(number % 10) * 64,
                                         "zip_crc_checked": True, "files": 2,
                                         "expanded_bytes": image.stat().st_size + label.stat().st_size,
                                         "source_archive": f"source_{number}_{split}.zip"}), encoding="utf-8")
            source[split + "_root"] = str(root)
            source[split + "_extraction_report"] = str(marker)
            assets.append({"number": number, "split": split, "image": image, "label": label, "root": root,
                           "marker": marker, "clip": clip})
        spec["sources"].append(source)
    path = tmp_path / "sources.json"

    def save():
        path.write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")

    save()
    return path, spec, assets, save


def test_mapping_background_unknown_and_objects_ignore_annotation_order(adapter):
    data = annotation()
    expected, counts = adapter.rasterize(data, (16, 12), "frame.jpg")
    reverse = dict(data, annotations=list(reversed(data["annotations"])))
    actual, reverse_counts = adapter.rasterize(reverse, (16, 12), "frame.jpg")
    np.testing.assert_array_equal(actual, expected)
    assert counts == reverse_counts
    assert expected[1, 1] == 255  # background cannot become negative supervision
    assert expected[8, 2] == 0
    assert expected[8, 6] == 255  # ambiguous road paint overrides freespace
    assert expected[8, 8] == 1  # explicit obstacle overrides road and ambiguous paint
    assert expected[8, 11] == 1
    assert expected.dtype == np.uint8


def test_unknown_class_is_ignored_never_assumed_nonroad(adapter):
    data = annotation()
    data["annotations"].append(polygon("new_unreviewed_class", 0, 6, 3, 9))
    values, counts = adapter.rasterize(data, (16, 12), "frame.jpg")
    assert values[7, 1] == 255 and counts["new_unreviewed_class"] == 1


def test_all_geometry_errors_are_reported_with_polygon_identity(adapter):
    data = annotation()
    data["annotations"] += [{"class": "whiteLane", "polygon": [1, 5, 4, 5, 8, 5]},
                            {"class": "vehicle", "polygon": [0, 0, -1, 5, 4, 5]}]
    errors = adapter.annotation_errors(data, (16, 12), "frame.jpg")
    assert [(e["polygon_index"], e["class"]) for e in errors] == [(4, "whiteLane"), (5, "vehicle")]
    assert "zero-area" in errors[0]["error"] and "outside" in errors[1]["error"]


def quarantine_fixture(adapter, source_fixture, tmp_path):
    source_spec, spec, assets, _ = source_fixture
    asset = assets[0]
    doc = json.loads(asset["label"].read_text())
    doc["annotations"].append({"class":"whiteLane", "polygon":[1,5,4,5,8,5]})
    asset["label"].write_text(json.dumps(doc),encoding="utf-8")
    marker = json.loads(asset["marker"].read_text())
    marker["expanded_bytes"] = asset["label"].stat().st_size + asset["image"].stat().st_size
    asset["marker"].write_text(json.dumps(marker))
    ledger = {"schema_version":1,"complete":True,"format":"camera_rc_aihub_annotation_quarantine_v1",
              "validator":adapter.ANNOTATION_VALIDATOR,"policy":"exclude_entire_frame_no_label_repair",
              "source_spec":{"sha256":adapter.sha256(source_spec)},
              "audited_counts":{s["dataset_id"]+"/"+split:1 for s in spec["sources"] for split in ("train","val")},
              "excluded_frames":[{"dataset_id":"aihub_city","split":"train",
                                  "sample_id":f"1.맑음/{asset['clip']}/{asset['image'].name}",
                                  "image":str(asset["image"]),"annotation":str(asset["label"]),
                                  "annotation_sha256":adapter.sha256(asset["label"]),
                                  "errors":adapter.annotation_errors(doc,(16,12),asset["image"].name)}]}
    path = tmp_path/"quarantine.json"
    path.write_text(json.dumps(ledger),encoding="utf-8")
    return path, ledger


def test_explicit_quarantine_removes_whole_frame_and_preserves_sources(adapter, source_fixture, tmp_path):
    ledger_path, ledger = quarantine_fixture(adapter, source_fixture, tmp_path)
    source_spec, _, assets, _ = source_fixture
    before = adapter.sha256(assets[0]["label"])
    with pytest.raises(ValueError, match="zero-area"):
        adapter.prepare(source_spec,tmp_path/"strict_failed")
    report = adapter.prepare(source_spec,tmp_path/"quarantined",ledger_path)
    assert report["total_samples"] == 3
    assert report["manifests"]["train"]["samples"] == 1
    assert report["quarantine"]["excluded_frames"] == ledger["excluded_frames"]
    assert adapter.sha256(assets[0]["label"]) == before
    assert not list((tmp_path/"quarantined"/"masks"/"aihub_city"/"train").glob("*.png"))


@pytest.mark.parametrize("change", ["hash", "reason", "count", "unknown_frame", "validator"])
def test_changed_or_forged_quarantine_fails(adapter, source_fixture, tmp_path, change):
    p, ledger = quarantine_fixture(adapter,source_fixture,tmp_path)
    if change=="hash": ledger["excluded_frames"][0]["annotation_sha256"]="a"*64
    if change=="reason": ledger["excluded_frames"][0]["errors"][0]["error"]="manual exclusion"
    if change=="count": ledger["audited_counts"]["aihub_city/train"]=99
    if change=="unknown_frame": ledger["excluded_frames"][0]["sample_id"]="unknown.jpg"
    if change=="validator": ledger["validator"]="unreviewed"
    p.write_text(json.dumps(ledger),encoding="utf-8")
    with pytest.raises(ValueError): adapter.prepare(source_fixture[0],tmp_path/"invalid_quarantine",p)
    assert not (tmp_path/"invalid_quarantine"/".complete").exists()


def test_annotation_audit_scans_every_frame_and_binds_errors(adapter, source_fixture, tmp_path, monkeypatch):
    quarantine_fixture(adapter,source_fixture,tmp_path)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts"))
    import audit_aihub_annotations
    report = audit_aihub_annotations.audit(source_fixture[0],tmp_path/"full_audit.json",2)
    assert report["audited_json_count"] == 4 and report["excluded_frame_count"] == 1
    assert report["excluded_by_source"] == {"aihub_city/train":1}
    assert report["error_class_counts"] == {"whiteLane":1}
    prepared=adapter.prepare(source_fixture[0],tmp_path/"from_audit",tmp_path/"full_audit.json")
    assert prepared["total_samples"]==3


def test_ledger_cannot_quarantine_a_now_valid_frame(adapter, source_fixture, tmp_path):
    p, ledger = quarantine_fixture(adapter,source_fixture,tmp_path)
    asset = source_fixture[2][0]
    asset["label"].write_text(json.dumps(annotation(asset["image"].name)),encoding="utf-8")
    marker = json.loads(asset["marker"].read_text())
    marker["expanded_bytes"] = asset["label"].stat().st_size + asset["image"].stat().st_size
    asset["marker"].write_text(json.dumps(marker))
    ledger["excluded_frames"][0]["annotation_sha256"] = adapter.sha256(asset["label"])
    p.write_text(json.dumps(ledger),encoding="utf-8")
    with pytest.raises(ValueError,match="valid frame"):
        adapter.prepare(source_fixture[0],tmp_path/"valid_frame_rejected",p)


@pytest.mark.parametrize("value", [-1, 16.01, float("nan"), float("inf"), True])
def test_invalid_coordinates_fail_without_clipping(adapter, value):
    data = annotation()
    data["annotations"][1]["polygon"][0] = value
    with pytest.raises(ValueError, match="coordinates"):
        adapter.rasterize(data, (16, 12), "frame.jpg")


def test_boundary_coordinates_and_dimensions_validated(adapter):
    data = annotation()
    values, _ = adapter.rasterize(data, (16, 12), "frame.jpg")
    assert values[11, 15] == 0  # observed right/bottom image-edge convention accepted
    with pytest.raises(ValueError, match="resolution"):
        adapter.rasterize(data, (15, 12), "frame.jpg")
    with pytest.raises(ValueError, match="filename"):
        adapter.rasterize(data, (16, 12), "different.jpg")


def test_degenerate_and_unsupervised_targets_fail(adapter):
    data = annotation()
    data["annotations"] = [polygon("background")]
    with pytest.raises(ValueError, match="no supervised"):
        adapter.rasterize(data, (16, 12), "frame.jpg")
    data["annotations"] = [{"class": "freespace", "polygon": [0, 0, 5, 5, 10, 10]}]
    with pytest.raises(ValueError, match="zero-area"):
        adapter.rasterize(data, (16, 12), "frame.jpg")


def test_complete_output_retains_official_splits_and_source_provenance(adapter, source_fixture, tmp_path):
    source_spec, spec, assets, _ = source_fixture
    before = {str(path): adapter.sha256(path) for asset in assets
              for path in (asset["image"], asset["label"], asset["marker"])}
    output = tmp_path / "prepared"
    report = adapter.prepare(source_spec, output)
    assert report["complete"] is True and report["total_samples"] == 4
    assert report["drive_or_day_independence_claimed"] is False
    assert report["training_started"] is False and report["gpu_used"] is False
    assert not (output / ".incomplete").exists()
    marker = json.loads((output / ".complete").read_text())
    assert marker["report_sha256"] == adapter.sha256(output / "report.json")
    train, val = read_manifest(output / "train.jsonl"), read_manifest(output / "val.jsonl")
    validate_splits(train, val)
    for split, rows in (("train", train), ("val", val)):
        assert len(rows) == 2
        assert report["manifests"][split]["sha256"] == adapter.sha256(output / f"{split}.jsonl")
        for row in rows:
            assert row["source_split"] == ("Training" if split == "train" else "Validation")
            assert row["origin_id"] == "aihub_commercial:" + row["group_id"]
            assert row["target_original_sha256"] == adapter.sha256(row["target_original"])
            assert row["image_sha256"] == adapter.sha256(row["image"])
            assert row["target_sha256"] == adapter.sha256(row["target"])
            assert row["ignored_class_counts"] == {"whiteLane": 1}
            with Image.open(row["target"]) as mask:
                assert mask.mode == "L" and mask.size == (16, 12)
                assert sorted(np.unique(mask).tolist()) == [0, 1, 255]
    assert before == {path: adapter.sha256(path) for path in before}


@pytest.mark.parametrize("kind", ["incomplete", "bad_marker", "missing_label", "unsupported12"])
def test_incomplete_and_invalid_sources_never_publish(adapter, source_fixture, tmp_path, kind):
    source_spec, spec, assets, save = source_fixture
    asset = assets[0]
    if kind == "incomplete":
        (asset["root"] / "INCOMPLETE").touch()
    elif kind == "bad_marker":
        marker = json.loads(asset["marker"].read_text())
        marker["zip_crc_checked"] = False
        asset["marker"].write_text(json.dumps(marker))
    elif kind == "missing_label":
        asset["label"].unlink()
    else:
        spec["sources"][0]["dataset_number"] = 12
        save()
    output = tmp_path / "invalid"
    with pytest.raises(ValueError):
        adapter.prepare(source_spec, output)
    assert (output / ".incomplete").is_file() and not (output / ".complete").exists()
    assert json.loads((output / "report.json").read_text())["complete"] is False
    assert not (output / "train.jsonl").exists()


def test_cross_dataset_clip_collision_rejected_before_decoding(adapter, source_fixture, tmp_path):
    source_spec, _, assets, _ = source_fixture
    # A different dataset ID must not conceal a shared original clip.
    first, last = assets[0], assets[-1]
    for kind in ("images", "labels"):
        parent = last["root"] / "1.맑음" / kind
        old = parent / last["clip"]
        new = parent / first["clip"]
        old.rename(new)
        path = next(new.iterdir())
        path.rename(new / (first["clip"] + "_02" + path.suffix))
    with pytest.raises(ValueError, match="clip overlap"):
        adapter.prepare(source_spec, tmp_path / "clip_leak")


def test_reencoded_duplicate_pixels_cross_split_rejected(adapter, source_fixture, tmp_path):
    source_spec, _, assets, _ = source_fixture
    first, val = assets[0], assets[1]
    # Losslessly encode identical decoded JPEG pixels as a new PNG under val's own frame ID.
    new = val["image"].with_suffix(".png")
    with Image.open(first["image"]) as image:
        image.save(new)
    val["image"].unlink()
    obj = json.loads(val["label"].read_text())
    obj["information"]["filename"] = new.name
    val["label"].write_text(json.dumps(obj))
    marker = json.loads(val["marker"].read_text())
    marker["expanded_bytes"] = new.stat().st_size + val["label"].stat().st_size
    val["marker"].write_text(json.dumps(marker))
    assert adapter.sha256(first["image"]) != adapter.sha256(new)
    with pytest.raises(ValueError, match="Duplicate image bytes/pixels"):
        adapter.prepare(source_spec, tmp_path / "pixel_leak")


def test_missing_whole_pair_is_detected_against_extraction_inventory(adapter, source_fixture, tmp_path):
    source_spec, _, assets, _ = source_fixture
    asset = assets[0]
    extra_image = asset["image"].with_name(asset["clip"] + "_02.jpg")
    extra_label = asset["label"].with_name(asset["clip"] + "_02.json")
    Image.new("RGB", (16, 12), (70, 20, 200)).save(extra_image)
    extra_label.write_text(json.dumps(annotation(extra_image.name)), encoding="utf-8")
    marker = json.loads(asset["marker"].read_text())
    marker["files"] = 4
    marker["expanded_bytes"] += extra_image.stat().st_size + extra_label.stat().st_size
    asset["marker"].write_text(json.dumps(marker))
    # Both members disappear after completion, leaving one perfectly paired frame.
    asset["image"].unlink()
    asset["label"].unlink()
    with pytest.raises(ValueError, match="inventory differs"):
        adapter.prepare(source_spec, tmp_path / "partial_but_paired")


def test_stable_manifest_order_and_mask_bytes_independent_of_output(adapter, source_fixture, tmp_path):
    source_spec, _, _, _ = source_fixture
    a, b = tmp_path / "first", tmp_path / "second"
    adapter.prepare(source_spec, a)
    adapter.prepare(source_spec, b)
    for split in ("train", "val"):
        rows_a, rows_b = read_manifest(a / f"{split}.jsonl"), read_manifest(b / f"{split}.jsonl")
        for left, right in zip(rows_a, rows_b):
            assert {k: v for k, v in left.items() if k != "target"} == {
                k: v for k, v in right.items() if k != "target"}
            assert adapter.sha256(left["target"]) == adapter.sha256(right["target"])
    with pytest.raises(FileExistsError):
        adapter.prepare(source_spec, a)
