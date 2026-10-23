"""Download fixed BDD100K archives, verify bytes, and optionally extract train/val.

Archive.org is a mirror, not the dataset publisher. These checks establish the
selected mirror-file identity, not license approval or driving-model quality.
No archive content is executed. Failed downloads/extractions remain inspectable.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import time
import zipfile

import requests


BASE_URL = "https://archive.org/download/bdd100k/"
ASSETS = {
    "images": {
        "filename": "bdd100k_images.zip",
        "size": 6941520264,
        "md5": "b538e3731a132e28dae37f18c442c51e",
    },
    "masks": {
        "filename": "bdd100k_drivable_maps.zip",
        "size": 693038872,
        "md5": "1bd70019468a81572c70d374751eb9e2",
    },
}
MAX_EXPANDED = 20 * 1024**3
CHUNK_SIZE = 1024 * 1024


class RetryableDownloadError(RuntimeError):
    def __init__(self, message, delay=None):
        super().__init__(message)
        self.delay = delay


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def verify_file(path, spec):
    """Hash an existing archive; a mismatch never removes or renames it."""
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Expected an ordinary file: {path}")
    size = path.stat().st_size
    if size != spec["size"]:
        raise ValueError(f"Size mismatch for {path}: {size} != {spec['size']}")
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
            md5.update(chunk)
            sha256.update(chunk)
    if md5.hexdigest() != spec["md5"]:
        raise ValueError(f"MD5 mismatch for {path}; preserved for inspection")
    return {"path": str(path.resolve()), "bytes": size, "md5": md5.hexdigest(),
            "sha256": sha256.hexdigest(), "verified": True}


def retry_delay(value, fallback):
    delay = fallback
    if value:
        try:
            delay = float(value)
        except (ValueError, TypeError):
            try:
                then = parsedate_to_datetime(value)
                if then.tzinfo is None:
                    then = then.replace(tzinfo=timezone.utc)
                delay = (then - datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError, OverflowError):
                pass
    return min(60.0, max(0.0, delay))


def _check_response(response, offset, expected):
    encoding = response.headers.get("Content-Encoding", "identity").lower()
    if encoding != "identity":
        raise ValueError("Compressed HTTP transfer encoding is incompatible with byte-range verification")
    if response.status_code == 206:
        match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
        if not match:
            raise ValueError("Missing or invalid Content-Range")
        start, end, total = map(int, match.groups())
        if start != offset or total != expected or not start <= end < total:
            raise ValueError("Content-Range does not match the local partial file and expected archive size")
        length = end - start + 1
    elif response.status_code == 200 and offset == 0:
        length = expected
    elif response.status_code == 200:
        raise ValueError("Server ignored Range; partial file preserved, no bytes appended")
    else:
        response.raise_for_status()
        raise ValueError(f"Unexpected HTTP status {response.status_code}")
    if "Content-Length" in response.headers and int(response.headers["Content-Length"]) != length:
        raise ValueError("Content-Length does not match the expected byte range")
    return length


def download_asset(key, output, *, attempts=5, session=None, sleep=time.sleep,
                   clock=time.monotonic, specs=None):
    """Resume using validated HTTP byte ranges; retries are bounded."""
    specs = ASSETS if specs is None else specs
    spec = specs[key]
    if not isinstance(attempts, int) or isinstance(attempts, bool) or not 1 <= attempts <= 10:
        raise ValueError("attempts must be an integer from 1 to 10")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / spec["filename"]
    partial = archive.with_suffix(archive.suffix + ".part")
    if archive.exists() or archive.is_symlink():
        emit("verify_existing", asset=key, path=str(archive))
        return verify_file(archive, spec)
    if partial.is_symlink() or partial.exists() and not partial.is_file():
        raise ValueError(f"Expected ordinary partial file: {partial}")
    if partial.exists() and partial.stat().st_size > spec["size"]:
        raise ValueError("Partial archive exceeds expected size; preserved for inspection")
    own_session = session is None
    session = requests.Session() if own_session else session
    try:
        last_progress = clock()
        for attempt in range(1, attempts + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == spec["size"]:
                break
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            emit("download_attempt", asset=key, attempt=attempt, resumed_bytes=offset,
                 expected_bytes=spec["size"])
            try:
                with session.get(BASE_URL + spec["filename"], headers=headers, stream=True,
                                 timeout=(15, 60), allow_redirects=True) as response:
                    if not response.url.lower().startswith("https://"):
                        raise ValueError("Archive redirect did not preserve HTTPS")
                    if response.status_code == 429 or 500 <= response.status_code <= 599:
                        delay = retry_delay(response.headers.get("Retry-After"), 2**attempt)
                        raise RetryableDownloadError(f"HTTP {response.status_code}", delay)
                    response_length = _check_response(response, offset, spec["size"])
                    received = 0
                    with partial.open("ab" if partial.exists() else "xb") as stream:
                        for chunk in response.iter_content(CHUNK_SIZE):
                            if not chunk:
                                continue
                            if received + len(chunk) > response_length:
                                raise ValueError("Response body exceeds its advertised byte range")
                            stream.write(chunk)
                            received += len(chunk)
                            now = clock()
                            if now - last_progress >= 30:
                                stream.flush()
                                emit("download_progress", asset=key, bytes=offset + received,
                                     expected_bytes=spec["size"],
                                     percent=round(100 * (offset + received) / spec["size"], 2))
                                last_progress = now
                    if received != response_length:
                        raise RetryableDownloadError("Response ended before its advertised byte range")
                    if offset + received < spec["size"]:
                        raise RetryableDownloadError("Server returned only part of the remaining archive", 0)
                break
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                    RetryableDownloadError) as error:
                if attempt == attempts:
                    raise RuntimeError(f"Download failed after {attempts} attempts; partial file preserved: {partial}") from error
                delay = getattr(error, "delay", None)
                delay = retry_delay(None, 2**attempt) if delay is None else delay
                emit("download_retry", asset=key, reason=str(error), wait_seconds=delay)
                sleep(delay)
        emit("verify_download", asset=key, path=str(partial))
        result = verify_file(partial, spec)
        if archive.exists() or archive.is_symlink():
            raise FileExistsError(f"Destination appeared during download: {archive}")
        partial.rename(archive)
        result["path"] = str(archive)
        emit("download_complete", asset=key, **result)
        return result
    finally:
        if own_session:
            session.close()


def _download_chunk(url, start, end, total, folder, attempts, session_factory, sleep):
    complete = folder / f"{start:012d}-{end:012d}.bin"
    partial = complete.with_suffix(".part")
    expected = end - start + 1
    for path in (complete, partial):
        if path.is_symlink() or path.exists() and not path.is_file():
            raise ValueError(f"Expected ordinary chunk file: {path}")
    if complete.exists():
        if complete.stat().st_size != expected:
            raise ValueError(f"Completed chunk has incorrect size: {complete}")
        return complete
    with session_factory() as session:
        for attempt in range(1, attempts + 1):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > expected:
                raise ValueError(f"Chunk exceeds expected size: {partial}")
            if offset == expected:
                break
            begin = start + offset
            try:
                with session.get(url, headers={"Accept-Encoding": "identity",
                                              "Range": f"bytes={begin}-{end}"},
                                 stream=True, timeout=(15, 60), allow_redirects=True) as response:
                    if not response.url.lower().startswith("https://"):
                        raise ValueError("Archive redirect did not preserve HTTPS")
                    if response.status_code == 429 or 500 <= response.status_code <= 599:
                        raise RetryableDownloadError(
                            f"HTTP {response.status_code}",
                            retry_delay(response.headers.get("Retry-After"), 2**attempt))
                    if response.status_code != 206:
                        response.raise_for_status()
                        raise ValueError("Bounded-range download requires HTTP 206")
                    length = _check_response(response, begin, total)
                    if length != end - begin + 1:
                        raise ValueError("Server did not return the exact requested bounded range")
                    received = 0
                    with partial.open("ab" if partial.exists() else "xb") as stream:
                        for chunk in response.iter_content(CHUNK_SIZE):
                            if not chunk:
                                continue
                            if received + len(chunk) > length:
                                raise ValueError("Chunk body exceeds its advertised range")
                            stream.write(chunk)
                            received += len(chunk)
                    if received != length:
                        raise RetryableDownloadError("Truncated chunk")
                break
            except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError,
                    RetryableDownloadError) as error:
                if attempt == attempts:
                    raise RuntimeError(f"Chunk download failed after {attempts} attempts: {partial}") from error
                delay = getattr(error, "delay", None)
                sleep(retry_delay(None, 2**attempt) if delay is None else delay)
    if partial.stat().st_size != expected:
        raise ValueError(f"Chunk remains incomplete: {partial}")
    if complete.exists():
        raise FileExistsError(complete)
    partial.rename(complete)
    return complete


def download_chunked(key, output, *, workers=4, chunk_bytes=16 * 1024**2, attempts=5,
                     session_factory=requests.Session, sleep=time.sleep, clock=time.monotonic,
                     specs=None):
    """Download bounded ranges concurrently, then validate the entire archive.

    The older streaming .part is retained. Chunks and an interrupted assembly
    remain available; subsequent runs reuse complete chunks after size checks.
    """
    specs = ASSETS if specs is None else specs
    spec = specs[key]
    if (isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8
            or isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 1
            or isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 10):
        raise ValueError("Require workers=1..8, attempts=1..10, and positive chunk_bytes")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    archive = output / spec["filename"]
    if archive.exists() or archive.is_symlink():
        return verify_file(archive, spec)
    folder = output / (spec["filename"] + ".chunks")
    if folder.is_symlink():
        raise ValueError("Chunk directory must not be a symlink")
    folder.mkdir(exist_ok=True)
    ranges = [(start, min(start + chunk_bytes, spec["size"]) - 1)
              for start in range(0, spec["size"], chunk_bytes)]
    url = spec.get("url", BASE_URL + spec["filename"])
    completed, last_progress = 0, clock()
    emit("chunk_download_start", asset=key, chunks=len(ranges), workers=workers,
         chunk_bytes=chunk_bytes, expected_bytes=spec["size"])
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(_download_chunk, url, start, end, spec["size"], folder,
                            attempts, session_factory, sleep): (start, end)
            for start, end in ranges
        }
        pending_futures = set(futures)
        try:
            while pending_futures:
                done, pending_futures = wait(pending_futures, timeout=30, return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    completed += 1
                now = clock()
                if now - last_progress >= 30 or completed == len(ranges):
                    emit("chunk_download_progress", asset=key, chunks_complete=completed,
                         chunks_total=len(ranges),
                         bytes_complete=sum((end - start + 1) for item, (start, end) in futures.items()
                                            if item.done() and not item.cancelled() and item.exception() is None),
                         expected_bytes=spec["size"])
                    last_progress = now
        except BaseException:
            for future in futures:
                future.cancel()
            raise
    assembled = archive.with_suffix(archive.suffix + ".assembled.part")
    if assembled.is_symlink() or assembled.exists() and not assembled.is_file():
        raise ValueError("Expected ordinary assembly file")
    # Rewrite only this tool's assembly output, never the existing curl .part.
    with assembled.open("wb") as target:
        for start, end in ranges:
            with (folder / f"{start:012d}-{end:012d}.bin").open("rb") as source:
                for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                    target.write(chunk)
    result = verify_file(assembled, spec)
    if archive.exists():
        raise FileExistsError(archive)
    assembled.rename(archive)
    result["path"] = str(archive)
    emit("download_complete", asset=key, **result)
    return result


def _member_path(info):
    # ZipInfo normalizes backslashes on Windows, so validate the raw name.
    name = info.orig_filename
    if not name or "\\" in name or name.startswith("/") or ":" in name:
        raise ValueError(f"Unsafe ZIP path: {name!r}")
    parts = name.rstrip("/").split("/")
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
    if any(not part or part in {".", ".."} or part.endswith((" ", "."))
           or reserved.match(part) or any(ord(c) < 32 for c in part)
           or any(c in '<>"|?*' for c in part) for part in parts):
        raise ValueError(f"Unsafe ZIP path component: {name!r}")
    kind = stat.S_IFMT(info.external_attr >> 16)
    if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ValueError(f"ZIP symlink or special file rejected: {name}")
    if info.flag_bits & 1:
        raise ValueError(f"Encrypted ZIP member rejected: {name}")
    return PurePosixPath(*parts)


def _selected(key, path):
    if not {"train", "val"}.intersection(path.parts[:-1]):
        return False
    return path.name.lower().endswith(".jpg" if key == "images" else "_drivable_id.png")


def extract_assets(archives, output, *, specs=None, max_expanded=MAX_EXPANDED):
    """Extract selected members into a fresh directory, checking path and CRC."""
    specs = ASSETS if specs is None else specs
    if not isinstance(max_expanded, int) or isinstance(max_expanded, bool) or max_expanded < 1:
        raise ValueError("max_expanded must be a positive integer")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Extraction output already exists: {output}")
    # Validate all archive bytes before creating extraction output.
    verified = {key: verify_file(path, specs[key]) for key, path in archives.items()}
    selected, seen, kinds, expanded = [], set(), {}, 0
    for key, archive in archives.items():
        count = 0
        with zipfile.ZipFile(archive) as zipped:
            for info in zipped.infolist():
                path = _member_path(info)
                canonical = str(path).casefold()
                # Shared directory entries across the two official ZIPs are OK.
                if canonical in seen and not (info.is_dir() and kinds[canonical] == "directory"):
                    raise ValueError(f"Duplicate or case-colliding ZIP path: {path}")
                # Duplicate directory entries within one ZIP are also harmless;
                # only files can be extracted and overwritten.
                seen.add(canonical)
                kind = "directory" if info.is_dir() else "file"
                if canonical in kinds and kinds[canonical] != kind:
                    raise ValueError(f"ZIP file/directory collision: {path}")
                kinds[canonical] = kind
                if not info.is_dir() and _selected(key, path):
                    if info.file_size < 0:
                        raise ValueError(f"Invalid ZIP member size: {path}")
                    expanded += info.file_size
                    if expanded > max_expanded:
                        raise ValueError("Selected ZIP expansion exceeds the configured byte limit")
                    selected.append((key, Path(archive), info, path))
                    count += 1
        if not count:
            raise ValueError(f"No train/val members selected for {key}")
    for canonical in kinds:
        for parent in PurePosixPath(canonical).parents:
            if str(parent) != "." and kinds.get(str(parent)) == "file":
                raise ValueError(f"ZIP file is used as a parent directory: {canonical}")
    output.mkdir(parents=True, exist_ok=False)
    pending = output / ".incomplete"
    pending.write_text("Extraction is incomplete until report.json is written.\n", encoding="utf-8")
    report = {"complete": False, "archives": verified, "selected_bytes": expanded,
              "selection": "train/val JPG images and *_drivable_id.png masks only",
              "original_archives_modified": False, "counts": {}, "extracted_bytes": 0}
    try:
        for key, archive in archives.items():
            members = [entry for entry in selected if entry[0] == key]
            with zipfile.ZipFile(archive) as zipped:
                for index, (_, _, info, relative) in enumerate(members, 1):
                    destination = output.joinpath(*relative.parts)
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if not destination.resolve().is_relative_to(output):
                        raise ValueError(f"ZIP destination escapes output: {relative}")
                    read = 0
                    with zipped.open(info) as source, destination.open("xb") as target:
                        for chunk in iter(lambda: source.read(CHUNK_SIZE), b""):
                            read += len(chunk)
                            if read > info.file_size or report["extracted_bytes"] + len(chunk) > max_expanded:
                                raise ValueError("Actual ZIP expansion exceeds the verified limits")
                            target.write(chunk)
                            report["extracted_bytes"] += len(chunk)
                    if read != info.file_size:
                        raise ValueError(f"Truncated ZIP member: {relative}")
                    if index % 5000 == 0:
                        emit("extraction_progress", asset=key, files=index, total_files=len(members))
            report["counts"][key] = len(members)
        report["complete"] = True
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if report["complete"]:
            pending.unlink()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Archive and resumable .part directory")
    parser.add_argument("--assets", choices=tuple(ASSETS), nargs="+", default=list(ASSETS))
    parser.add_argument("--extract", type=Path, help="Optional NEW directory for verified train/val extraction")
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4, help="Bounded-range workers, 1..8")
    parser.add_argument("--chunk-mib", type=int, default=16)
    parser.add_argument("--stream", action="store_true", help="Use one streaming request instead of bounded chunks")
    args = parser.parse_args()
    assets = list(dict.fromkeys(args.assets))
    if args.chunk_mib < 1:
        parser.error("--chunk-mib must be positive")
    def download(key):
        if args.stream:
            return download_asset(key, args.output, attempts=args.attempts)
        return download_chunked(key, args.output, attempts=args.attempts, workers=args.workers,
                                chunk_bytes=args.chunk_mib * 1024**2)
    result = {"archives": {key: download(key) for key in assets},
              "source": BASE_URL, "license_acceptance_performed": False}
    if args.extract is not None:
        result["extraction"] = extract_assets({key: item["path"] for key, item in result["archives"].items()}, args.extract)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
