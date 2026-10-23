import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import pytest

from camera_rc.data import ManifestDataset, read_manifest, validate_splits


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
spec = importlib.util.spec_from_file_location("download_a2d2", SCRIPTS / "download_a2d2.py")
downloader = importlib.util.module_from_spec(spec)
sys.modules["download_a2d2"] = downloader
spec.loader.exec_module(downloader)
spec = importlib.util.spec_from_file_location("prepare_a2d2", SCRIPTS / "prepare_a2d2.py")
preparer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparer)


def record(key, content):
    return {"key": key, "bytes": len(content), "md5": hashlib.md5(content, usedforsecurity=False).hexdigest()}


@pytest.mark.parametrize("key", ["../escape", "a/../../escape", "C:/escape", "/abs", "a\\b", "a//b", "a/./b", "a/CON.png", "a/NUL", "a/name.", "a/name ", "a/x:y", "a/x?y"])
def test_reject_unsafe_object_paths(key):
    with pytest.raises(ValueError):
        downloader.safe_relative(key)


def test_new_nested_path_has_strict_existing_ancestor(tmp_path):
    result = downloader.target_path(tmp_path, "new/dir/file.png")
    assert result == tmp_path / "new/dir/file.png"
    assert not result.exists()


def test_symlink_escape_is_rejected(tmp_path):
    root, outside = tmp_path / "source", tmp_path / "outside"
    root.mkdir(); outside.mkdir()
    try:
        (root / "escape").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation unavailable on this host")
    with pytest.raises(ValueError, match="escapes"):
        downloader.target_path(root, "escape/image.png")


class Response:
    def __init__(self, obj, content, offset=0):
        self.url = downloader.BASE + obj["key"]
        self.status_code = 206 if offset else 200
        self.headers = {"ETag": '"' + obj["md5"] + '"', "Content-Length": str(len(content))}
        if offset:
            self.headers["Content-Range"] = f"bytes {offset}-{obj['bytes'] - 1}/{obj['bytes']}"
        self.content = content

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def raise_for_status(self):
        pass

    def iter_content(self, size):
        yield self.content


class Session:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.reply


def test_resume_enforces_range_etag_md5_and_keeps_verified_file(tmp_path, monkeypatch):
    content = b"ten-bytes!"
    obj = record("folder/file.png", content)
    (tmp_path / "folder").mkdir()
    partial = tmp_path / "folder/file.png.part"
    partial.write_bytes(content[:3])
    session = Session(Response(obj, content[3:], 3))
    monkeypatch.setattr(downloader, "session", lambda: session)
    result = downloader.download_one(tmp_path, obj)
    assert (tmp_path / obj["key"]).read_bytes() == content
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert not partial.exists()
    assert session.calls[0][1]["headers"]["Range"] == "bytes=3-"
    assert session.calls[0][1]["headers"]["If-Match"] == '"' + obj["md5"] + '"'
    assert downloader.download_one(tmp_path, obj)["existing"]
    assert len(session.calls) == 1


@pytest.mark.parametrize("problem", ["range", "status", "etag", "length", "redirect", "checksum"])
def test_bad_response_never_promoted(tmp_path, monkeypatch, problem):
    content = b"ten-bytes!"
    obj = record("file.png", content)
    partial = tmp_path / "file.png.part"
    partial.write_bytes(content[:3])
    reply = Response(obj, content[3:], 3)
    if problem == "range": reply.headers["Content-Range"] = "bytes 0-6/10"
    if problem == "status": reply.status_code = 200
    if problem == "etag": reply.headers["ETag"] = '"' + "f" * 32 + '"'
    if problem == "length": reply.headers["Content-Length"] = "99"
    if problem == "redirect": reply.url = "https://example.com/untrusted"
    if problem == "checksum": reply.content = b"X" * len(reply.content)
    monkeypatch.setattr(downloader, "session", lambda: Session(reply))
    with pytest.raises(ValueError):
        downloader.download_one(tmp_path, obj)
    assert not (tmp_path / "file.png").exists()
    assert partial.exists()


def test_existing_corrupt_download_is_not_silently_reused(tmp_path):
    (tmp_path / "file.png").write_bytes(b"different")
    with pytest.raises(ValueError, match="identity mismatch"):
        downloader.download_one(tmp_path, record("file.png", b"expected!"))


def test_semantic_mapping_keeps_road_markings_and_ignores_ambiguous_areas():
    classes = {"#ff00ff": "RD normal street", "#ffc125": "Solid line", "#8000ff": "Dashed line",
               "#d23273": "Zebra crossing", "#b432b4": "Drivable cobblestone", "#eee9bf": "Slow drive area",
               "#c87dd2": "Painted driv. instr.", "#ff0080": "Obstacles / trash", "#8b636c": "Non-drivable street",
               "#6e6e00": "Speed bumper", "#9696c8": "Parking area", "#60458f": "Blurred area"}
    lookup, known, mapping = preparer.label_tables(classes, strict=False)
    colors = [tuple(bytes.fromhex(color[1:])) for color in classes] + [(0, 0, 0), (1, 2, 3)]
    target, unknown = preparer.convert_mask(np.array([colors], dtype=np.uint8), lookup, known)
    assert target.tolist() == [[0] * 7 + [1, 1, 255, 255, 255, 255, 255]]
    assert unknown == {"#010203": 1}
    assert mapping["#ff0080"]["target"] == 1


def test_date_split_keeps_recordings_cameras_and_nearby_sessions_together():
    assert preparer.split_for_sequence("20181008_095521") == "val"
    assert preparer.split_for_sequence("20181107_132300") == "test"
    assert preparer.split_for_sequence("20181107_133445") == "test"
    assert preparer.split_for_sequence("20181204_191844") == "train"


@pytest.fixture
def source(tmp_path, monkeypatch):
    root = tmp_path / "raw"
    root.mkdir()
    classes = {"#ff00ff": "RD normal street", "#ff0000": "Car 1", "#60458f": "Blurred area"}
    extra_names = sorted((preparer.ROAD_NAMES | preparer.IGNORE_NAMES) - set(classes.values()))
    extra_names += ["Extra nonroad " + str(i) for i in range(55 - len(classes) - len(extra_names))]
    for i, name in enumerate(extra_names): classes[f"#{i + 1000:06x}"] = name
    objects = []
    for key in downloader.METADATA:
        path = root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(classes).encode() if key.endswith("class_list.json") else b"Publisher metadata."
        path.write_bytes(content)
        objects.append(record(key, content))
    for i, sequence in enumerate(("20180807_145028", "20181008_095521", "20181107_132300")):
        for kind in ("camera", "label"):
            key = f"camera_lidar_semantic/{sequence}/{kind}/cam_front_center/{sequence.replace('_', '')}_{kind}_frontcenter_000000091.png"
            path = root / key
            path.parent.mkdir(parents=True, exist_ok=True)
            if kind == "camera": image = Image.new("RGB", (48, 32), (10 + i, 30, 50))
            else:
                values = np.zeros((32, 48, 3), dtype=np.uint8)
                values[:, :16] = [255, 0, 255]
                values[:, 16:32] = [255, 0, 0]
                values[:, 32:] = [96, 69, 143]
                image = Image.fromarray(values)
            image.save(path)
            objects.append(record(key, path.read_bytes()))
    monkeypatch.setattr(downloader, "EXPECTED_PAIRS", 3)
    monkeypatch.setattr(downloader, "EXPECTED_SEQUENCES", 3)
    monkeypatch.setattr(downloader, "EXPECTED_BYTES", sum(o["bytes"] for o in objects if o["key"] not in downloader.METADATA))
    monkeypatch.setattr(preparer, "EXPECTED_SPLITS", {"train": 1, "val": 1, "test": 1})
    inventory = {"source": downloader.BASE, "objects": objects}
    path = root / "source_inventory.json"
    path.write_text(json.dumps(inventory), encoding="utf8")
    (root / "DOWNLOAD_COMPLETE.json").write_text(json.dumps({"complete": True, "inventory_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}), encoding="utf8")
    return root


def test_prepare_native_manifests_holdouts_and_original_preservation(source, tmp_path):
    before = {p: p.read_bytes() for p in source.rglob("*.png")}
    out = tmp_path / "prepared"
    report = preparer.prepare(source, out, image_size=(32, 48), workers=2)
    assert report["complete"] and not (out / ".incomplete").exists()
    assert not report["synchronized_multiview_samples"]
    assert all(path.read_bytes() == content for path, content in before.items())
    splits = {split: read_manifest(value["manifest"]) for split, value in report["manifests"].items()}
    validate_splits(splits["train"], splits["val"])
    validate_splits(splits["train"] + splits["val"], splits["test"])
    config = {"task": "segmentation", "model": {"num_classes": 2}, "data": {"image_size": [192, 320], "label_space_id": "road_surface_v1"}}
    for split, rows in splits.items():
        assert report["manifests"][split]["samples"] == 1
        assert set(np.asarray(Image.open(rows[0]["target"])).ravel()) == {0, 1, 255}
        assert tuple(ManifestDataset(rows, config)[0]["target"].shape) == (192, 320)
        assert rows[0]["original_size_hw"] == [32, 48]
        assert Path(rows[0]["source_image"]).is_file()
        assert Path(rows[0]["source_target"]).is_file()
        assert rows[0]["source_image_key"].startswith("camera_lidar_semantic/")


@pytest.mark.parametrize("problem", ["missing_marker", "changed_inventory", "corrupt_source", "existing_output", "unpaired"])
def test_prepare_fails_closed_on_incomplete_or_changed_sources(source, tmp_path, problem):
    out = tmp_path / "prepared"
    if problem == "missing_marker": (source / "DOWNLOAD_COMPLETE.json").unlink()
    if problem == "changed_inventory":
        with (source / "source_inventory.json").open("a") as f: f.write("\n")
    if problem == "corrupt_source": next(source.rglob("*_camera_*.png")).write_bytes(b"broken")
    if problem == "existing_output": out.mkdir()
    if problem == "unpaired":
        path = source / "source_inventory.json"
        inventory = json.loads(path.read_text())
        inventory["objects"] = [x for x in inventory["objects"] if not x["key"].endswith("_label_frontcenter_000000091.png")]
        path.write_text(json.dumps(inventory))
    with pytest.raises((ValueError, FileNotFoundError)):
        preparer.prepare(source, out, image_size=(32, 48), workers=1)
    assert not (out / "train.jsonl").exists()
