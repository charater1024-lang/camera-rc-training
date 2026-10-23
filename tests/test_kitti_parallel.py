import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import threading
import zipfile

import pytest
import requests


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
spec = importlib.util.spec_from_file_location("download_kitti_parallel", PROJECT / "scripts/download_kitti_parallel.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


class Response:
    def __init__(self, body, begin, end, *, headers=None, status=206, error=None):
        self.status_code, self.body, self.error = status, body, error
        self.url = downloader.URL
        self.headers = {"Content-Length": str(end - begin + 1),
                        "Content-Range": f"bytes {begin}-{end}/{downloader.SIZE}", "ETag": downloader.ETAG}
        if headers:
            self.headers.update(headers)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        for offset in range(0, len(self.body), chunk_size):
            yield self.body[offset:offset + chunk_size]
        if self.error:
            raise self.error


class Factory:
    def __init__(self, payload, first_fault=None, forbidden=False):
        self.payload, self.first_fault, self.forbidden = payload, first_fault, forbidden
        self.calls = []
        self.lock = threading.Lock()

    def __call__(self):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def get(self, url, **kwargs):
        with self.lock:
            if self.forbidden:
                raise AssertionError("Unexpected network call")
            self.calls.append((url, kwargs))
            begin, end = map(int, kwargs["headers"]["Range"].removeprefix("bytes=").split("-"))
            if self.first_fault is not None:
                fault, self.first_fault = self.first_fault, None
                return fault(begin, end)
            return Response(self.payload[begin:end + 1], begin, end)


@pytest.fixture
def payload(monkeypatch):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as zipped:
        zipped.writestr("training/image_2/um_000000.png", b"road-data" * 100)
        zipped.writestr("training/calib/um_000000.txt", b"calibration")
    value = stream.getvalue()
    monkeypatch.setattr(downloader, "SIZE", len(value))
    monkeypatch.setattr(downloader, "CHUNK_BYTES", 128)
    return value


def test_parallel_fixed_headers_zip_integrity_and_old_partial_preserved(tmp_path, payload):
    archives = tmp_path / "archives"
    archives.mkdir()
    old_partial = archives / "data_road.zip.part"
    old_partial.write_bytes(b"older full GET")
    factory = Factory(payload)
    result = downloader.download_kitti_parallel(tmp_path, workers=4, session_factory=factory, sleep=lambda _: None)
    assert Path(result["path"]).read_bytes() == payload
    assert result["sha256"] == hashlib.sha256(payload).hexdigest()
    assert result["zip_crc_checked"] and not result["publisher_hash_available"]
    assert result["etag_verified_during_this_download"]
    assert old_partial.read_bytes() == b"older full GET"
    assert not (archives / "data_road.zip.parallel.part").exists()
    for url, kwargs in factory.calls:
        assert url == downloader.URL
        assert kwargs["headers"]["If-Match"] == downloader.ETAG
        assert kwargs["headers"]["Accept-Encoding"] == "identity"
        assert kwargs["allow_redirects"] is False
    again = downloader.download_kitti_parallel(tmp_path, session_factory=Factory(payload, forbidden=True))
    assert again["sha256"] == result["sha256"]


def test_completed_chunks_reused_only_after_hash_validation(tmp_path, payload):
    result = downloader.download_kitti_parallel(tmp_path, workers=2, session_factory=Factory(payload))
    Path(result["path"]).unlink()
    again = downloader.download_kitti_parallel(tmp_path, workers=2, session_factory=Factory(payload, forbidden=True))
    assert again["sha256"] == result["sha256"]
    Path(result["path"]).unlink()
    chunk = next((tmp_path / "archives/data_road.zip.parallel.chunks").glob("*.bin"))
    raw = bytearray(chunk.read_bytes())
    raw[0] ^= 1
    chunk.write_bytes(raw)
    with pytest.raises(ValueError, match="SHA256"):
        downloader.download_kitti_parallel(tmp_path, workers=1, session_factory=Factory(payload, forbidden=True))


def test_interrupted_chunk_retries_verified_exact_offset(tmp_path, payload):
    stop = threading.Event()
    factory = Factory(payload, first_fault=lambda begin, end: Response(payload[:20], begin, end, error=requests.ConnectionError("broken")))
    waits = []
    path = downloader._download_chunk(tmp_path, 0, 127, 5, factory, waits.append, stop)
    assert path.read_bytes() == payload[:128]
    assert factory.calls[1][1]["headers"]["Range"] == "bytes=20-127"
    assert waits == [2]
    checkpoint = json.loads(path.with_suffix(".json").read_text())
    assert checkpoint["bytes"] == 128 and checkpoint["sha256"] == hashlib.sha256(payload[:128]).hexdigest()


def test_resume_across_runs_discards_only_uncheckpointed_tail(tmp_path, payload):
    factory = Factory(payload, first_fault=lambda begin, end: Response(payload[:20], begin, end, error=requests.Timeout("timeout")))
    with pytest.raises(RuntimeError, match="after 1 attempts"):
        downloader._download_chunk(tmp_path, 0, 127, 1, factory, lambda _: None, threading.Event())
    partial = next(tmp_path.glob("*.part"))
    with partial.open("ab") as stream:
        stream.write(b"uncheckpointed")
    resume = Factory(payload)
    path = downloader._download_chunk(tmp_path, 0, 127, 5, resume, lambda _: None, threading.Event())
    assert resume.calls[0][1]["headers"]["Range"] == "bytes=20-127"
    assert path.read_bytes() == payload[:128]


@pytest.mark.parametrize("header", [
    {"ETag": '"changed"'}, {"Content-Length": "127"}, {"Content-Length": ""},
    {"Content-Range": "bytes 1-128/1000"}, {"Content-Encoding": "gzip"},
])
def test_wrong_response_identity_or_boundaries_never_written(tmp_path, payload, header):
    factory = Factory(payload, first_fault=lambda begin, end: Response(payload[:128], begin, end, headers=header))
    with pytest.raises(ValueError):
        downloader._download_chunk(tmp_path, 0, 127, 5, factory, lambda _: None, threading.Event())
    assert not list(tmp_path.glob("*.bin"))
    assert next(tmp_path.glob("*.part")).stat().st_size == 0


@pytest.mark.parametrize("status", [200, 302, 412])
def test_non_206_is_fatal(tmp_path, payload, status):
    factory = Factory(payload, first_fault=lambda begin, end: Response(payload[:128], begin, end, status=status))
    with pytest.raises((ValueError, requests.HTTPError)):
        downloader._download_chunk(tmp_path, 0, 127, 5, factory, lambda _: None, threading.Event())
    assert len(factory.calls) == 1


def test_retry_503_then_success(tmp_path, payload):
    factory = Factory(payload, first_fault=lambda begin, end: Response(b"", begin, end, status=503, headers={"Retry-After": "1"}))
    waits = []
    downloader._download_chunk(tmp_path, 0, 127, 5, factory, waits.append, threading.Event())
    assert waits == [1.0]


def test_crc_failure_prevents_final_publication(tmp_path, payload):
    damaged = bytearray(payload)
    damaged[payload.index(b"road-data")] ^= 1
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        downloader.download_kitti_parallel(tmp_path, workers=2, session_factory=Factory(bytes(damaged)))
    assert not (tmp_path / "archives/data_road.zip").exists()
    assert (tmp_path / "archives/data_road.zip.parallel.part").exists()


def test_resume_metadata_mismatch_and_concurrency_limit(tmp_path, payload):
    with pytest.raises(ValueError, match="workers"):
        downloader.download_kitti_parallel(tmp_path, workers=33)
    result = downloader.download_kitti_parallel(tmp_path, workers=2, session_factory=Factory(payload))
    Path(result["path"]).unlink()
    metadata = tmp_path / "archives/data_road.zip.parallel.chunks/metadata.json"
    data = json.loads(metadata.read_text())
    data["etag"] = '"different"'
    metadata.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="identity"):
        downloader.download_kitti_parallel(tmp_path, session_factory=Factory(payload, forbidden=True))


def test_existing_untracked_archive_is_preserved_not_adopted(tmp_path, payload):
    (tmp_path / "archives").mkdir()
    archive = tmp_path / "archives/data_road.zip"
    archive.write_bytes(payload)
    with pytest.raises(FileExistsError, match="provenance"):
        downloader.download_kitti_parallel(tmp_path, session_factory=Factory(payload, forbidden=True))
    assert archive.read_bytes() == payload


def test_recover_rename_before_report_using_verified_chunks(tmp_path, payload):
    original = downloader.download_kitti_parallel(tmp_path, workers=2, session_factory=Factory(payload))
    report = tmp_path / "archives/data_road.zip.parallel.json"
    report.unlink()
    recovered = downloader.download_kitti_parallel(tmp_path, session_factory=Factory(payload, forbidden=True))
    assert recovered["recovered_after_rename"] and recovered["sha256"] == original["sha256"]
    assert report.exists()
