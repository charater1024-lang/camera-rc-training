"""Download anonymous public ACDC packages with publisher MD5 verification.

The official endpoint does not support byte-range resume. Interrupted partial
files are preserved under unique names before a bounded restart from byte zero.
Temporary download tokens stay in memory and never enter logs or provenance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import quote, urlparse
import zipfile

import requests

from download_bdd_archives import _member_path


BASE = "https://acdc.vision.ee.ethz.ch"
PACKAGES = {
    "gt": {"packageId": "6436eeae9880d97633275dc9", "name": "gt_trainval.zip",
           "md5": "54cf06e3f2d8a8c5d297dc55aa107cee", "max_bytes": 512 * 1024**2},
    "rgb": {"packageId": "6436f2259880d97633275dfc", "name": "rgb_anon_trainvaltest.zip",
            "md5": "3350587a08502b4dfee47750bfd2a052", "max_bytes": 20 * 1024**3},
}
MAX_EXPANDED_BYTES = 40 * 1024**3


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def identity(path):
    md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            md5.update(block)
            sha.update(block)
    return {"bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def _public_packages(session):
    with session.get(BASE + "/api/packages", timeout=(15, 45)) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Public package inventory HTTP {response.status_code}")
        inventory = response.json()
    packages = inventory.get("packages", [])
    result = {}
    for key, spec in PACKAGES.items():
        matches = [item for item in packages if item.get("packageId") == spec["packageId"]]
        if len(matches) != 1:
            raise ValueError(f"Pinned public package missing or duplicated: {spec['name']}")
        item = matches[0]
        if (item.get("personalData") is not False or item.get("state") != 0
                or item.get("name") != spec["name"] or item.get("md5") != spec["md5"]):
            raise ValueError(f"Public package identity/access changed: {spec['name']}")
        result[key] = item
    return result


def _token_url(session, spec):
    with session.get(BASE + "/api/getPackageUri/" + spec["packageId"], timeout=(15, 45)) as response:
        if response.status_code != 200:
            raise RuntimeError(f"Anonymous package request HTTP {response.status_code}")
        token = response.json().get("token")
    if not isinstance(token, str) or not token or len(token) > 8192 or not re.fullmatch(r"[A-Za-z0-9_.~-]+", token):
        raise ValueError("Anonymous package endpoint did not return a valid temporary token")
    return BASE + "/api/downloadPackage/" + quote(token, safe="") + "/" + spec["name"]


def fetch_package(session, spec, target, *, attempts=2):
    target = Path(target)
    if isinstance(attempts, bool) or not isinstance(attempts, int) or not 1 <= attempts <= 3:
        raise ValueError("attempts must be an integer from 1 to 3")
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if target.is_symlink() or partial.is_symlink():
        raise ValueError("Refusing symbolic-link download target")
    if target.exists():
        verified = identity(target)
        if verified["md5"] != spec["md5"] or verified["bytes"] > spec["max_bytes"]:
            raise ValueError(f"Existing archive identity mismatch: {target.name}")
        return {**verified, "name": spec["name"], "packageId": spec["packageId"],
                "publisher_md5": spec["md5"], "source": BASE + "/download", "path": str(target.resolve())}
    for attempt in range(1, attempts + 1):
        if partial.exists():
            preserved = partial.with_name(partial.name + f".failed-{time.time_ns()}")
            partial.rename(preserved)
            emit("partial_preserved", file=preserved.name, bytes=preserved.stat().st_size)
        try:
            url = _token_url(session, spec)
            with session.get(url, stream=True, headers={"Accept-Encoding": "identity"}, timeout=(15, 90)) as response:
                if urlparse(response.url).scheme != "https":
                    raise ValueError("Insecure package redirect")
                if response.status_code != 200:
                    raise RuntimeError(f"Package transfer HTTP {response.status_code}")
                advertised = response.headers.get("Content-Length")
                expected = int(advertised) if advertised is not None else None
                if expected is not None and not 0 < expected <= spec["max_bytes"]:
                    raise ValueError("Package Content-Length exceeds fixed download budget")
                received, last_report = 0, time.monotonic()
                md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
                with partial.open("xb") as stream:
                    for block in response.iter_content(4 * 1024**2):
                        if not block:
                            continue
                        if received + len(block) > spec["max_bytes"]:
                            raise ValueError("Package exceeds fixed download budget")
                        stream.write(block)
                        received += len(block)
                        md5.update(block)
                        sha.update(block)
                        if time.monotonic() - last_report >= 25:
                            emit("download_progress", file=target.name, bytes=received,
                                 total=expected, attempt=attempt)
                            last_report = time.monotonic()
                if expected is not None and expected != received:
                    raise ValueError("Truncated package Content-Length")
                if md5.hexdigest() != spec["md5"]:
                    raise ValueError("Publisher MD5 mismatch; partial preserved")
            partial.rename(target)
            result = {"name": spec["name"], "packageId": spec["packageId"], "bytes": received,
                      "md5": md5.hexdigest(), "sha256": sha.hexdigest(), "publisher_md5": spec["md5"],
                      "source": BASE + "/download", "path": str(target.resolve()),
                      "byte_range_resume": False}
            emit("download_verified", **result)
            return result
        except (requests.RequestException, ValueError, RuntimeError) as error:
            # Request exceptions may embed the token URL: never print error text
            # or chain the original exception into a traceback.
            emit("download_attempt_failed", file=target.name, attempt=attempt, error_type=type(error).__name__)
            if attempt == attempts:
                raise RuntimeError(f"Download failed for {target.name}; incomplete data preserved, no complete marker written") from None
            time.sleep(min(2 ** attempt, 8))


def extract_archive(archive, destination, verified, *, max_expanded_bytes=MAX_EXPANDED_BYTES):
    archive, destination = Path(archive), Path(destination)
    marker = destination / "EXTRACTION_COMPLETE.json"
    if marker.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior.get("archive_sha256") != verified["sha256"]:
            raise ValueError("Extraction provenance mismatch")
        return prior
    if destination.exists():
        raise FileExistsError(f"Incomplete or unknown extraction exists: {destination}")
    with zipfile.ZipFile(archive) as zipped:
        members, seen, files, total = [], set(), set(), 0
        for info in zipped.infolist():
            rel = _member_path(info)
            key = str(rel).casefold()
            if key in seen:
                raise ValueError("Case-insensitive duplicate ZIP member")
            seen.add(key)
            if not info.is_dir():
                files.add(key)
            total += info.file_size
            if total > max_expanded_bytes or len(seen) > 100000:
                raise ValueError("Expanded archive exceeds fixed budget")
            members.append((info, rel))
        for _, rel in members:
            if any(str(parent).casefold() in files for parent in rel.parents):
                raise ValueError("ZIP file/directory path collision")
        destination.mkdir(parents=True)
        (destination / "INCOMPLETE").touch()
        for index, (info, rel) in enumerate(members, 1):
            path = destination.joinpath(*rel.parts)
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target, length=4 * 1024**2)
            if index % 500 == 0:
                emit("extract_progress", archive=archive.name, members=index, total_members=len(members))
        result = {"archive_sha256": verified["sha256"], "files": len(files),
                  "expanded_bytes": total, "zip_crc_checked": True}
        write_json(marker, result)
        (destination / "INCOMPLETE").unlink()
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--only", choices=["both", "gt", "rgb"], default="both")
    parser.add_argument("--attempts", type=int, default=2)
    parser.add_argument("--skip-extract", action="store_true")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    with requests.Session() as session:
        packages = _public_packages(session)
        write_json(root / "source_inventory.json", {"source": BASE + "/api/packages", "packages": packages,
            "license": "Non-commercial research terms; preserve original packaged license; do not redistribute source or derived data."})
        results = []
        for key in (["gt", "rgb"] if args.only == "both" else [args.only]):
            spec = PACKAGES[key]
            archive = root / "archives" / spec["name"]
            result = fetch_package(session, spec, archive, attempts=args.attempts)
            if not args.skip_extract:
                result["extraction"] = extract_archive(archive, root / archive.stem, result)
            results.append(result)
        write_json(root / ("DOWNLOAD_COMPLETE.json" if args.only == "both" else f"DOWNLOAD_{args.only.upper()}_COMPLETE.json"),
                   {"dataset": "ACDC", "archives": results, "temporary_tokens_logged": False})


if __name__ == "__main__":
    main()
