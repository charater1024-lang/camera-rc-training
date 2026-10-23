"""Download the fixed official KITTI Road ZIP in verified, resumable byte ranges.

Usage: python scripts/download_kitti_parallel.py --root <kitti_road> --workers 32
Creates archives/data_road.zip, leaving the older data_road.zip.part untouched.
The S3 multipart ETag pins the remote object; it is NOT an MD5 or SHA256.
Chunk and assembled SHA256 values detect local corruption, not publisher-signed
authenticity. No publisher cryptographic checksum is available. ZIP CRC is also
checked before publication. Extraction is intentionally a separate step.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import threading
import time
import zipfile

import requests

from download_additional_datasets import hashes
from download_bdd_archives import emit, retry_delay


URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_road.zip"
FILENAME = "data_road.zip"
SIZE = 470992343
ETAG = '"ef2cedcbbf5f4122c9ead516342d0b71-57"'
CHUNK_BYTES = 4 * 1024**2
READ_BYTES = 64 * 1024


def _ordinary(path, directory=False):
    if path.is_symlink() or getattr(path, "is_junction", lambda: False)():
        raise ValueError(f"Refusing linked download path: {path}")
    if path.exists() and not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f"Unexpected download path type: {path}")


def _save(path, value):
    temporary = path.with_name(path.name + ".tmp")
    _ordinary(path)
    _ordinary(temporary)
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@contextmanager
def _lock(folder):
    """OS locks are released even after process termination; no stale PID locks."""
    path = folder / ".lock"
    _ordinary(path)
    with path.open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError("Another parallel KITTI downloader holds this directory") from error
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise RuntimeError("Another parallel KITTI downloader holds this directory") from error
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _state_base(start, end):
    return {"schema": 1, "url": URL, "etag": ETAG, "total": SIZE, "start": start, "end": end}


def _read_state(path, start, end):
    _ordinary(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if any(value.get(key) != expected for key, expected in _state_base(start, end).items()):
        raise ValueError(f"Chunk provenance mismatch: {path}")
    length = value.get("bytes")
    if isinstance(length, bool) or not isinstance(length, int) or not 0 <= length <= end - start + 1:
        raise ValueError(f"Invalid chunk checkpoint size: {path}")
    if not isinstance(value.get("sha256"), str) or len(value["sha256"]) != 64:
        raise ValueError(f"Missing chunk checkpoint hash: {path}")
    return value


def _prefix_hash(path, length):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        remaining = length
        while remaining:
            block = stream.read(min(1024**2, remaining))
            if not block:
                raise ValueError(f"Chunk shorter than its checkpoint: {path}")
            digest.update(block)
            remaining -= len(block)
    return digest


def _verify_complete(folder, start, end):
    chunk = folder / f"{start:012d}-{end:012d}.bin"
    state_path = chunk.with_suffix(".json")
    _ordinary(chunk)
    state = _read_state(state_path, start, end)
    expected = end - start + 1
    if state["bytes"] != expected or chunk.stat().st_size != expected:
        raise ValueError(f"Completed chunk size mismatch: {chunk}")
    if _prefix_hash(chunk, expected).hexdigest() != state["sha256"]:
        raise ValueError(f"Completed chunk SHA256 mismatch; preserved: {chunk}")
    return chunk


def _check_response(response, begin, end):
    if response.status_code != 206:
        response.raise_for_status()
        raise ValueError(f"Exact bounded range requires HTTP 206, received {response.status_code}")
    if response.url != URL:
        raise ValueError("Unexpected range response URL")
    if response.headers.get("ETag") != ETAG:
        raise ValueError("Publisher ETag changed")
    if response.headers.get("Content-Range") != f"bytes {begin}-{end}/{SIZE}":
        raise ValueError("Content-Range differs from the exact requested range")
    length = end - begin + 1
    if response.headers.get("Content-Length") != str(length):
        raise ValueError("Content-Length missing or incorrect")
    if response.headers.get("Content-Encoding", "identity").lower() != "identity":
        raise ValueError("HTTP compression is incompatible with byte-range verification")
    return length


class _Retryable(RuntimeError):
    def __init__(self, message, delay=None):
        super().__init__(message)
        self.delay = delay


def _download_chunk(folder, start, end, attempts, session_factory, sleep, stop):
    complete = folder / f"{start:012d}-{end:012d}.bin"
    partial, state_path = complete.with_suffix(".part"), complete.with_suffix(".json")
    for path in (complete, partial, state_path):
        _ordinary(path)
    if complete.exists():
        return _verify_complete(folder, start, end)
    expected = end - start + 1
    if state_path.exists():
        state = _read_state(state_path, start, end)
        offset = state["bytes"]
        if not partial.exists():
            if offset:
                raise ValueError(f"Chunk checkpoint has no partial file: {partial}")
            partial.touch(exist_ok=False)
        if partial.stat().st_size > expected:
            raise ValueError(f"Partial chunk exceeds its range: {partial}")
        digest = _prefix_hash(partial, offset)
        if digest.hexdigest() != state["sha256"]:
            raise ValueError(f"Partial chunk SHA256 mismatch; preserved: {partial}")
        # A hard kill can leave bytes newer than the last atomic checkpoint.
        # Only this tool's unverified tail is discarded, after its prefix passes.
        if partial.stat().st_size != offset:
            with partial.open("r+b") as stream:
                stream.truncate(offset)
    else:
        if partial.exists() and partial.stat().st_size:
            raise ValueError(f"Untracked nonempty partial chunk; preserved: {partial}")
        offset, digest = 0, hashlib.sha256()
        _save(state_path, {**_state_base(start, end), "bytes": 0, "sha256": digest.hexdigest()})
        partial.touch(exist_ok=True)
    with session_factory() as session:
        for attempt in range(1, attempts + 1):
            if stop.is_set():
                raise RuntimeError("Download stopped; verified chunk checkpoints preserved")
            if offset == expected:
                break
            begin, received = start + offset, 0
            try:
                with session.get(URL, headers={"Accept-Encoding": "identity", "If-Match": ETAG,
                                              "Range": f"bytes={begin}-{end}"},
                                 stream=True, timeout=(15, 45), allow_redirects=False) as response:
                    if response.status_code == 429 or 500 <= response.status_code <= 599:
                        raise _Retryable(f"HTTP {response.status_code}", retry_delay(response.headers.get("Retry-After"), 2 ** attempt))
                    length = _check_response(response, begin, end)
                    checkpoint = offset
                    with partial.open("ab") as stream:
                        try:
                            for block in response.iter_content(READ_BYTES):
                                if stop.is_set():
                                    raise RuntimeError("Download stopped; chunk checkpoint preserved")
                                if not block:
                                    continue
                                if received + len(block) > length:
                                    raise ValueError("Range body exceeds advertised size")
                                stream.write(block)
                                digest.update(block)
                                offset += len(block)
                                received += len(block)
                                if offset - checkpoint >= 1024**2:
                                    stream.flush()
                                    _save(state_path, {**_state_base(start, end), "bytes": offset, "sha256": digest.hexdigest()})
                                    checkpoint = offset
                        finally:
                            stream.flush()
                            _save(state_path, {**_state_base(start, end), "bytes": offset, "sha256": digest.hexdigest()})
                    if received != length:
                        raise _Retryable("Range response ended early")
                break
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError, _Retryable) as error:
                if attempt == attempts:
                    raise RuntimeError(f"Chunk failed after {attempts} attempts; checkpoint preserved: {partial}") from error
                delay = getattr(error, "delay", None)
                delay = min(2 ** attempt, 30) if delay is None else delay
                emit("kitti_chunk_retry", start=start, bytes=offset, attempt=attempt, delay=delay, reason=str(error))
                sleep(delay)
    if offset != expected or partial.stat().st_size != expected:
        raise ValueError(f"Incomplete chunk: {partial}")
    if complete.exists():
        raise FileExistsError(complete)
    partial.rename(complete)
    return _verify_complete(folder, start, end)


def _verify_zip(path):
    _ordinary(path)
    if path.stat().st_size != SIZE:
        raise ValueError("Assembled ZIP size mismatch")
    result = hashes(path)
    with zipfile.ZipFile(path) as zipped:
        entries = zipped.infolist()
        if not entries or sum(entry.file_size for entry in entries) > 4 * 1024**3:
            raise ValueError("Unexpected ZIP expansion or empty archive")
        failed = zipped.testzip()
        if failed is not None:
            raise ValueError(f"ZIP CRC failure: {failed}")
    result.update(url=URL, publisher_etag=ETAG, publisher_hash_available=False,
                  sha256_purpose="Local integrity record, not a publisher checksum", zip_crc_checked=True,
                  path=str(path.resolve()))
    return result


def download_kitti_parallel(root, *, workers=32, attempts=5, session_factory=requests.Session, sleep=time.sleep):
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 32:
        raise ValueError("workers must be an integer in 1..32")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 5:
        raise ValueError("attempts must be an integer in 1..5")
    root = Path(root).expanduser()
    _ordinary(root, directory=True)
    root = root.resolve()
    archives = root / "archives"
    _ordinary(archives, directory=True)
    archives.mkdir(parents=True, exist_ok=True)
    archive = archives / FILENAME
    folder = archives / (FILENAME + ".parallel.chunks")
    assembled = archives / (FILENAME + ".parallel.part")
    report_path = archives / (FILENAME + ".parallel.json")
    for path in (archive, assembled, report_path):
        _ordinary(path)
    _ordinary(folder, directory=True)
    folder.mkdir(exist_ok=True)
    with _lock(folder):
        metadata = {"schema": 1, "url": URL, "size": SIZE, "etag": ETAG, "chunk_bytes": CHUNK_BYTES}
        metadata_path = folder / "metadata.json"
        _ordinary(metadata_path)
        if metadata_path.exists():
            if json.loads(metadata_path.read_text(encoding="utf-8")) != metadata:
                raise ValueError("Chunk directory identity or chunk geometry mismatch")
        else:
            if any(path.name != ".lock" for path in folder.iterdir()):
                raise ValueError("Untracked chunk directory is not empty")
            _save(metadata_path, metadata)
        ranges = [(start, min(start + CHUNK_BYTES, SIZE) - 1) for start in range(0, SIZE, CHUNK_BYTES)]
        if archive.exists():
            result = _verify_zip(archive)
            if report_path.exists():
                previous = json.loads(report_path.read_text(encoding="utf-8"))
                if (previous.get("sha256") != result["sha256"] or previous.get("publisher_etag") != ETAG
                        or previous.get("url") != URL or previous.get("bytes") != SIZE
                        or previous.get("etag_verified_during_this_download") is not True):
                    raise ValueError("Existing ZIP differs from its saved local integrity record")
            else:
                # Recover a crash between final rename and report publication,
                # but never adopt an unrelated same-size ZIP on CRC alone.
                expected_digest = hashlib.sha256()
                for start, end in ranges:
                    if not (folder / f"{start:012d}-{end:012d}.bin").exists():
                        raise FileExistsError("Existing ZIP lacks verified download provenance; preserved")
                    chunk = _verify_complete(folder, start, end)
                    with chunk.open("rb") as source:
                        for block in iter(lambda: source.read(1024**2), b""):
                            expected_digest.update(block)
                if result["sha256"] != expected_digest.hexdigest():
                    raise ValueError("Existing ZIP differs from verified chunk provenance")
                result.update(etag_verified_during_this_download=True, recovered_after_rename=True,
                              workers=workers, chunk_bytes=CHUNK_BYTES, chunks=len(ranges))
                _save(report_path, result)
            result.update(etag_verified_in_saved_download_record=True, reused_existing=True)
            emit("kitti_parallel_existing_verified", **result)
            return result
        stop = threading.Event()
        done_count, completed_bytes, started = 0, 0, time.monotonic()
        emit("kitti_parallel_start", workers=workers, chunks=len(ranges), chunk_bytes=CHUNK_BYTES, bytes=SIZE)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_download_chunk, folder, start, end, attempts, session_factory, sleep, stop): (start, end)
                       for start, end in ranges}
            pending = set(futures)
            try:
                while pending:
                    done, pending = wait(pending, timeout=15, return_when=FIRST_COMPLETED)
                    for future in done:
                        future.result()
                        done_count += 1
                        start, end = futures[future]
                        completed_bytes += end - start + 1
                    emit("kitti_parallel_progress", chunks_complete=done_count, chunks_total=len(ranges),
                         verified_bytes=completed_bytes, total_bytes=SIZE, elapsed_seconds=round(time.monotonic() - started, 1))
            except BaseException:
                stop.set()
                for future in pending:
                    future.cancel()
                raise
        # Existing assembly is our own disposable output. Older full-GET .part
        # is never opened. Re-verify chunks while producing deterministic order.
        expected_digest = hashlib.sha256()
        with assembled.open("wb") as target:
            for start, end in ranges:
                chunk = _verify_complete(folder, start, end)
                with chunk.open("rb") as source:
                    for block in iter(lambda: source.read(1024**2), b""):
                        target.write(block)
                        expected_digest.update(block)
            target.flush()
            os.fsync(target.fileno())
        result = _verify_zip(assembled)
        if result["sha256"] != expected_digest.hexdigest():
            raise ValueError("Assembled file differs from verified chunk stream")
        result.update(path=str(archive), etag_verified_during_this_download=True, workers=workers,
                      chunk_bytes=CHUNK_BYTES, chunks=len(ranges))
        if archive.exists():
            raise FileExistsError("Archive appeared during parallel download; neither file overwritten")
        # Rename on the same filesystem publishes only the fully verified ZIP.
        assembled.rename(archive)
        _save(report_path, result)
        emit("kitti_parallel_complete", **result)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Dataset root, with archives/ beneath it")
    parser.add_argument("--workers", type=int, default=32, help="Concurrent range requests, 1..32")
    parser.add_argument("--attempts", type=int, default=5, help="Maximum attempts per chunk, 1..5")
    args = parser.parse_args()
    download_kitti_parallel(**vars(args))


if __name__ == "__main__":
    main()
