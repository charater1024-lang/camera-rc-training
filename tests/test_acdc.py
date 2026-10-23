import hashlib
import json
from pathlib import Path
import sys
import zipfile

import pytest
import requests
import numpy as np
from PIL import Image


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import download_acdc as downloader
import prepare_acdc as adapter


class FakeResponse:
    def __init__(self, data=None, body=b"", status=200, headers=None, failure=None):
        self.data, self.body, self.status_code = data, body, status
        self.headers, self.failure = headers or {}, failure
        self.url = downloader.BASE + "/api/downloadPackage/PRIVATE_TOKEN/file.zip"

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def json(self):
        return self.data

    def iter_content(self, size):
        yield self.body
        if self.failure:
            raise self.failure


class FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def get(self, url, **kwargs):
        self.requests.append((url, kwargs))
        return next(self.responses)


def _spec(body):
    return {"packageId": "public-id", "name": "test.zip", "max_bytes": 1000,
            "md5": hashlib.md5(body, usedforsecurity=False).hexdigest()}


def test_download_unknown_length_verifies_publisher_md5_and_never_persists_token(tmp_path, capsys):
    body = b"verified content"
    session = FakeSession([FakeResponse({"token": "PRIVATE_TOKEN"}), FakeResponse(body=body)])
    result = downloader.fetch_package(session, _spec(body), tmp_path / "test.zip")
    assert (tmp_path / "test.zip").read_bytes() == body
    assert result["md5"] == _spec(body)["md5"]
    assert result["bytes"] == len(body)
    assert "PRIVATE_TOKEN" not in str(result) + capsys.readouterr().out
    assert not (tmp_path / "test.zip.part").exists()


def test_failed_download_preserves_partial_and_redacts_exception_token(tmp_path, capsys):
    session = FakeSession([FakeResponse({"token": "PRIVATE_TOKEN"}),
                           FakeResponse(body=b"partial", failure=requests.ConnectionError("bad PRIVATE_TOKEN"))])
    with pytest.raises(RuntimeError, match="incomplete data preserved") as failure:
        downloader.fetch_package(session, _spec(b"expected"), tmp_path / "test.zip", attempts=1)
    assert (tmp_path / "test.zip.part").read_bytes() == b"partial"
    assert not (tmp_path / "test.zip").exists()
    assert "PRIVATE_TOKEN" not in str(failure.value) + capsys.readouterr().out


def test_restart_preserves_previous_partial_instead_of_appending(tmp_path):
    (tmp_path / "test.zip.part").write_bytes(b"old partial")
    body = b"whole archive"
    session = FakeSession([FakeResponse({"token": "PRIVATE_TOKEN"}), FakeResponse(body=body)])
    downloader.fetch_package(session, _spec(body), tmp_path / "test.zip")
    preserved = list(tmp_path.glob("test.zip.part.failed-*"))
    assert len(preserved) == 1 and preserved[0].read_bytes() == b"old partial"
    assert (tmp_path / "test.zip").read_bytes() == body
    assert all("Range" not in kwargs.get("headers", {}) for _, kwargs in session.requests)


@pytest.mark.parametrize("body,headers", [(b"bad", {}), (b"good", {"Content-Length": "9999"})])
def test_checksum_or_fixed_budget_failure_never_publishes_archive(tmp_path, body, headers):
    session = FakeSession([FakeResponse({"token": "PRIVATE_TOKEN"}), FakeResponse(body=body, headers=headers)])
    with pytest.raises(RuntimeError, match="Download failed"):
        downloader.fetch_package(session, _spec(b"good"), tmp_path / "test.zip", attempts=1)
    assert not (tmp_path / "test.zip").exists()


@pytest.mark.parametrize("change", [{"personalData": True}, {"md5": "different"}, {"state": 1}])
def test_inventory_rejects_access_or_identity_changes(change):
    values = [{**spec, "personalData": False, "state": 0} for spec in downloader.PACKAGES.values()]
    values[0].update(change)
    with pytest.raises(ValueError, match="identity/access changed"):
        downloader._public_packages(FakeSession([FakeResponse({"packages": values})]))


@pytest.mark.parametrize("names", [["../escape.txt"], ["A.txt", "a.txt"], ["folder", "folder/child"]])
def test_extraction_rejects_unsafe_paths_before_destination_creation(tmp_path, names):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        for name in names:
            zipped.writestr(name, b"data")
    with pytest.raises(ValueError):
        downloader.extract_archive(archive, tmp_path / "extracted", {"sha256": "checked"})
    assert not (tmp_path / "extracted").exists()


def test_extraction_checks_expanded_budget_and_records_crc(tmp_path):
    archive = tmp_path / "input.zip"
    with zipfile.ZipFile(archive, "w") as zipped:
        zipped.writestr("data/image.png", b"12345")
    with pytest.raises(ValueError, match="budget"):
        downloader.extract_archive(archive, tmp_path / "over-budget", {"sha256": "checked"}, max_expanded_bytes=4)
    assert not (tmp_path / "over-budget").exists()
    result = downloader.extract_archive(archive, tmp_path / "extracted", {"sha256": "checked"})
    assert result["zip_crc_checked"] and result["expanded_bytes"] == 5
    assert (tmp_path / "extracted/data/image.png").read_bytes() == b"12345"
    assert not (tmp_path / "extracted/INCOMPLETE").exists()


def test_acdc_mapping_uses_cityscapes_raw_ids_and_keeps_uncertain_final_labels():
    labels = np.array([[7, 8, 24, 0, 255, 9], [33, 26, 31, 6, 14, 29]], dtype=np.uint8)
    invalid = np.ones_like(labels)
    converted = adapter.remap_labels(labels, invalid)
    np.testing.assert_array_equal(converted, [[0, 1, 1, 255, 255, 255], [1, 1, 1, 255, 255, 255]])
    assert converted.shape == labels.shape


@pytest.mark.parametrize("labels,invalid", [
    (np.array([[7, 100]], dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8)),
    (np.array([[[7, 7, 7]]], dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)),
    (np.array([[7]], dtype=np.uint8), np.array([[255]], dtype=np.uint8)),
    (np.array([[7]], dtype=np.uint8), np.zeros((2, 2), dtype=np.uint8)),
    (np.array([[0, 255]], dtype=np.uint8), np.zeros((1, 2), dtype=np.uint8)),
])
def test_acdc_invalid_labels_uncertainty_and_void_only_are_rejected(labels, invalid):
    with pytest.raises(ValueError):
        adapter.remap_labels(labels, invalid)


def _small_acdc(tmp_path, monkeypatch):
    monkeypatch.setattr(adapter, "COUNTS", {"train": {"fog": 1}, "val": {"night": 1}})
    image_root, gt_root = tmp_path / "source/rgb_anon", tmp_path / "source/gt"
    images = []
    for index, (condition, split) in enumerate((("fog", "train"), ("night", "val"))):
        sequence, frame = "GOPR0101", f"{index:06d}"
        image = image_root / condition / split / sequence / f"{sequence}_frame_{frame}_rgb_anon.png"
        label = gt_root / condition / split / sequence / f"{sequence}_frame_{frame}_gt_labelIds.png"
        inv = label.with_name(f"{sequence}_frame_{frame}_gt_invIds.png")
        image.parent.mkdir(parents=True)
        label.parent.mkdir(parents=True)
        Image.fromarray(np.random.default_rng(index).integers(0, 256, (24, 32, 3), dtype=np.uint8)).save(image)
        labels = np.full((24, 32), 8, dtype=np.uint8)
        labels[:, :16] = 7
        labels[0, 0] = 255
        Image.fromarray(labels).save(label)
        Image.fromarray(np.ones((24, 32), dtype=np.uint8)).save(inv)
        images.append(image)
    test = image_root / "fog/test/GOPR0999/GOPR0999_frame_000000_rgb_anon.png"
    test.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(test)
    return image_root, gt_root, images


def test_acdc_preparation_preserves_original_masks_official_splits_and_uncertainty(tmp_path, monkeypatch):
    image_root, gt_root, images = _small_acdc(tmp_path, monkeypatch)
    report = adapter.prepare_acdc(image_root, gt_root, tmp_path / "prepared")
    assert report["complete"] and report["total_samples"] == 2
    assert report["excluded_images"]["test"] == 1
    assert report["test_or_reference_used_for_supervised_learning"] is False
    assert report["source_sequence_names_present_in_both_geographic_splits"] == ["GOPR0101"]
    rows = {split: json.loads((tmp_path / "prepared" / f"{split}.jsonl").read_text(encoding="utf-8"))
            for split in ("train", "val")}
    assert rows["train"]["group_id"] != rows["val"]["group_id"]
    assert rows["train"]["source_sequence"] == rows["val"]["source_sequence"] == "GOPR0101"
    for split, row in rows.items():
        assert Path(row["image"]) in images
        assert Path(row["uncertainty_mask"]).is_file()
        assert row["target_sha256"] == adapter._sha256(tmp_path / "prepared" / row["target"])
        with Image.open(tmp_path / "prepared" / row["target"]) as target:
            assert target.size == (32, 24)
            assert np.asarray(target)[0, 0] == 255
            assert np.asarray(target)[10, 10] == 0  # invIds=1 does not erase valid road label.
        assert report["manifests"][split]["uncertain_semantically_labelled_pixels"] == 24 * 32 - 1
    assert not (tmp_path / "prepared/.incomplete").exists()
    with pytest.raises(FileExistsError):
        adapter.prepare_acdc(image_root, gt_root, tmp_path / "prepared")


def test_acdc_missing_pair_rejected_before_output_creation(tmp_path, monkeypatch):
    image_root, gt_root, images = _small_acdc(tmp_path, monkeypatch)
    images[0].unlink()
    with pytest.raises(ValueError, match="Missing"):
        adapter.prepare_acdc(image_root, gt_root, tmp_path / "prepared")
    assert not (tmp_path / "prepared").exists()


def test_acdc_content_duplicate_across_official_splits_leaves_incomplete_marker(tmp_path, monkeypatch):
    image_root, gt_root, images = _small_acdc(tmp_path, monkeypatch)
    images[1].write_bytes(images[0].read_bytes())
    with pytest.raises(ValueError, match="RGB train/val leakage"):
        adapter.prepare_acdc(image_root, gt_root, tmp_path / "prepared")
    assert (tmp_path / "prepared/.incomplete").exists()
    assert not (tmp_path / "prepared/train.jsonl").exists()
    report = json.loads((tmp_path / "prepared/report.json").read_text(encoding="utf-8"))
    assert report["complete"] is False


def test_acdc_source_mask_alignment_mismatch_is_rejected(tmp_path, monkeypatch):
    image_root, gt_root, images = _small_acdc(tmp_path, monkeypatch)
    target = next(gt_root.glob("*/*/*/*_gt_invIds.png"))
    Image.new("L", (10, 10)).save(target)
    with pytest.raises(ValueError, match="size or format mismatch"):
        adapter.prepare_acdc(image_root, gt_root, tmp_path / "prepared")
    assert (tmp_path / "prepared/.incomplete").exists()
