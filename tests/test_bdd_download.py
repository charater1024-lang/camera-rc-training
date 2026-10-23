import hashlib
import importlib.util
from pathlib import Path
import stat
import zipfile

import pytest
import requests


PROJECT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("download_bdd_archives", PROJECT / "scripts/download_bdd_archives.py")
downloader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(downloader)


def asset(payload, filename="archive.zip"):
    return {"images": {"filename": filename, "size": len(payload),
                       "md5": hashlib.md5(payload, usedforsecurity=False).hexdigest()}}


class Response:
    def __init__(self, status, body=b"", headers=None, error=None):
        self.status_code, self.body = status, body
        self.headers = headers or {}
        self.url = "https://archive.org/download/bdd100k/archive.zip"
        self.error = error

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))

    def iter_content(self, chunk_size):
        if self.body:
            yield self.body
        if self.error:
            raise self.error


class Session:
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return self.responses.pop(0)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_interrupted_download_resumes_exact_offset_and_verifies(tmp_path):
    payload = b"abcdef"
    session = Session(
        Response(200, b"abc", {"Content-Length": "6"}, requests.ConnectionError("broken")),
        Response(206, b"def", {"Content-Range": "bytes 3-5/6", "Content-Length": "3"}),
    )
    waits = []
    result = downloader.download_asset("images", tmp_path, specs=asset(payload), session=session, sleep=waits.append)
    assert result["verified"] and Path(result["path"]).read_bytes() == payload
    assert session.calls[1][1]["headers"]["Range"] == "bytes=3-"
    assert waits == [2.0]
    assert not (tmp_path / "archive.zip.part").exists()


@pytest.mark.parametrize("response", [
    Response(200, b"abcdef", {"Content-Length": "6"}),
    Response(206, b"def", {"Content-Range": "bytes 2-4/6", "Content-Length": "3"}),
    Response(206, b"def", {"Content-Range": "bytes 3-5/7", "Content-Length": "3"}),
    Response(206, b"def", {"Content-Range": "bytes 3-5/6", "Content-Length": "4"}),
])
def test_bad_resume_response_does_not_modify_partial(tmp_path, response):
    partial = tmp_path / "archive.zip.part"
    partial.write_bytes(b"abc")
    with pytest.raises(ValueError):
        downloader.download_asset("images", tmp_path, specs=asset(b"abcdef"), session=Session(response))
    assert partial.read_bytes() == b"abc"


def test_bad_digest_preserves_downloaded_partial(tmp_path):
    with pytest.raises(ValueError, match="MD5"):
        downloader.download_asset("images", tmp_path, specs=asset(b"abcdef"),
                                  session=Session(Response(200, b"ghijkl")))
    assert (tmp_path / "archive.zip.part").read_bytes() == b"ghijkl"
    assert not (tmp_path / "archive.zip").exists()


def test_complete_partial_is_verified_without_http(tmp_path):
    (tmp_path / "archive.zip.part").write_bytes(b"abc")
    session = Session()
    result = downloader.download_asset("images", tmp_path, specs=asset(b"abc"), session=session)
    assert result["verified"] and not session.calls


def test_429_retry_is_bounded_and_wait_is_capped(tmp_path):
    waits = []
    session = Session(Response(429, headers={"Retry-After": "120"}),
                      Response(429, headers={"Retry-After": "120"}))
    with pytest.raises(RuntimeError, match="2 attempts"):
        downloader.download_asset("images", tmp_path, specs=asset(b"abc"), session=session,
                                  attempts=2, sleep=waits.append)
    assert waits == [60.0]
    assert len(session.calls) == 2


def test_existing_archive_is_verified_not_overwritten(tmp_path):
    archive = tmp_path / "archive.zip"
    archive.write_bytes(b"bad")
    session = Session()
    with pytest.raises(ValueError, match="MD5"):
        downloader.download_asset("images", tmp_path, specs=asset(b"abc"), session=session)
    assert archive.read_bytes() == b"bad" and not session.calls


def make_zip(tmp_path, members):
    archive = tmp_path / "archive.zip"
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_STORED) as zipped:
        for name, contents in members:
            if isinstance(name, str) and "\\" in name:
                # ZipInfo(name) normalizes Windows separators when running on
                # Windows; preserve the malicious raw member name for this test.
                entry = zipfile.ZipInfo()
                entry.filename = name
                name = entry
            zipped.writestr(name, contents)
    return archive, asset(archive.read_bytes())


def test_extract_selects_only_train_val_and_preserves_paths(tmp_path):
    archive, specs = make_zip(tmp_path, [
        ("bdd100k/images/100k/train/one.jpg", b"image1"),
        ("bdd100k/images/100k/val/two.jpg", b"image2"),
        ("bdd100k/images/100k/test/three.jpg", b"test"),
        ("bdd100k/images/100k/train/execute.py", b"never execute"),
    ])
    output = tmp_path / "extracted"
    result = downloader.extract_assets({"images": archive}, output, specs=specs)
    assert result["complete"] and result["counts"] == {"images": 2}
    assert (output / "bdd100k/images/100k/train/one.jpg").read_bytes() == b"image1"
    assert not (output / "bdd100k/images/100k/test").exists()
    assert not (output / "bdd100k/images/100k/train/execute.py").exists()
    assert not (output / ".incomplete").exists()
    with pytest.raises(FileExistsError):
        downloader.extract_assets({"images": archive}, output, specs=specs)


@pytest.mark.parametrize("name", [
    "../escape.jpg", "/absolute.jpg", "C:/absolute.jpg",
    "train\\escape.jpg", "train/../escape.jpg", "train/CON.jpg",
    "train/name:stream.jpg", "train/trailing./image.jpg",
])
def test_rejects_unsafe_zip_paths_before_output_created(tmp_path, name):
    archive, specs = make_zip(tmp_path, [(name, b"bad"), ("train/good.jpg", b"ok")])
    output = tmp_path / "extracted"
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        downloader.extract_assets({"images": archive}, output, specs=specs)
    assert not output.exists()


def test_rejects_symlinks_duplicates_and_file_parent(tmp_path):
    link = zipfile.ZipInfo("train/link.jpg")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    for index, members in enumerate([
        [(link, "../outside")],
        [("train/a.jpg", b"a"), ("train/A.jpg", b"b")],
        [("train", b"file"), ("train/a.jpg", b"a")],
    ]):
        folder = tmp_path / str(index)
        folder.mkdir()
        archive, specs = make_zip(folder, members)
        with pytest.raises(ValueError):
            downloader.extract_assets({"images": archive}, folder / "extracted", specs=specs)
        assert not (folder / "extracted").exists()


def test_expansion_limit_is_checked_before_creating_output(tmp_path):
    archive, specs = make_zip(tmp_path, [("train/a.jpg", b"12345")])
    with pytest.raises(ValueError, match="expansion"):
        downloader.extract_assets({"images": archive}, tmp_path / "extracted", specs=specs, max_expanded=4)
    assert not (tmp_path / "extracted").exists()


def test_selected_member_crc_failure_retains_incomplete_marker(tmp_path):
    archive, _ = make_zip(tmp_path, [("train/a.jpg", b"unique image payload")])
    payload = archive.read_bytes().replace(b"unique image payload", b"Unique image payload")
    archive.write_bytes(payload)
    specs = asset(payload)
    output = tmp_path / "extracted"
    with pytest.raises(zipfile.BadZipFile, match="CRC"):
        downloader.extract_assets({"images": archive}, output, specs=specs)
    assert (output / ".incomplete").exists()
    assert '"complete": false' in (output / "report.json").read_text()


def test_masks_select_id_labels_not_color_visualization(tmp_path):
    archive, specs = make_zip(tmp_path, [
        ("bdd100k/drivable_maps/labels/train/a_drivable_id.png", b"ids"),
        ("bdd100k/drivable_maps/color/train/a_drivable_color.png", b"colors"),
    ])
    specs = {"masks": specs["images"]}
    result = downloader.extract_assets({"masks": archive}, tmp_path / "extracted", specs=specs)
    assert result["counts"] == {"masks": 1}


def test_parallel_bounded_chunks_assemble_and_preserve_older_stream_partial(tmp_path):
    payload = b"abcdefghijkl"
    calls = []

    class RangeSession(Session):
        def get(self, url, **kwargs):
            byte_range = kwargs["headers"]["Range"]
            calls.append(byte_range)
            start, end = map(int, byte_range.removeprefix("bytes=").split("-"))
            return Response(206, payload[start:end + 1], {
                "Content-Range": f"bytes {start}-{end}/{len(payload)}",
                "Content-Length": str(end - start + 1),
            })

    old = tmp_path / "archive.zip.part"
    old.write_bytes(b"older partial")
    result = downloader.download_chunked("images", tmp_path, specs=asset(payload),
                                         workers=2, chunk_bytes=4, session_factory=RangeSession)
    assert Path(result["path"]).read_bytes() == payload
    assert set(calls) == {"bytes=0-3", "bytes=4-7", "bytes=8-11"}
    assert old.read_bytes() == b"older partial"


def test_chunk_resumes_within_range_and_checks_requested_end(tmp_path):
    folder = tmp_path / "chunks"
    folder.mkdir()
    (folder / "000000000004-000000000007.part").write_bytes(b"ef")
    session = Session(Response(206, b"gh", {"Content-Range": "bytes 6-7/12"}))
    chunk = downloader._download_chunk("https://archive.org/item", 4, 7, 12, folder,
                                      2, lambda: session, lambda value: None)
    assert chunk.read_bytes() == b"efgh"
    assert session.calls[0][1]["headers"]["Range"] == "bytes=6-7"
    another = tmp_path / "bad"
    another.mkdir()
    bad = Session(Response(206, b"abcd", {"Content-Range": "bytes 0-3/12"}))
    with pytest.raises(ValueError, match="exact requested"):
        downloader._download_chunk("https://archive.org/item", 0, 7, 12, another,
                                   1, lambda: bad, lambda value: None)
