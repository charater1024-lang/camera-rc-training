"""Download only official A2D2 semantic RGB/labels, with a frozen S3 inventory.

No AWS credentials, archive, or LiDAR download is needed. Existing files and
resumed ranges are checked against publisher single-part MD5 ETags. This does
not construct synchronized multi-camera samples: the label frames are sparse.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time
import xml.etree.ElementTree as ET

import requests

BASE = "https://audi-autonomous-driving-dataset.s3.eu-central-1.amazonaws.com/"
PREFIX = "camera_lidar_semantic/"
EXPECTED_PAIRS = 41277
EXPECTED_BYTES = 152021120762
EXPECTED_SEQUENCES = 23
CAMERAS = frozenset({"cam_front_center", "cam_front_left", "cam_front_right",
                     "cam_side_left", "cam_side_right", "cam_rear_center"})
METADATA = (PREFIX + "class_list.json", PREFIX + "LICENSE", PREFIX + "README-SemSeg.txt",
            "cams_lidars.json", "LICENSE.txt", "README.txt")
NS = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
_local = threading.local()


def emit(event, **values):
    print(json.dumps({"event": event, **values}, ensure_ascii=False), flush=True)


def save_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    if temp.is_symlink() or path.is_symlink():
        raise ValueError("Refusing symlink metadata")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers.update({"Accept-Encoding": "identity"})
    return _local.session


def object_record(element):
    key = element.findtext("s:Key", namespaces=NS)
    size = int(element.findtext("s:Size", namespaces=NS))
    etag = element.findtext("s:ETag", namespaces=NS).strip('"')
    if size < 1 or not re.fullmatch(r"[a-f0-9]{32}", etag):
        raise ValueError(f"Expected single-part MD5 ETag: {key}")
    return {"key": key, "bytes": size, "md5": etag,
            "last_modified": element.findtext("s:LastModified", namespaces=NS)}


def list_page(params):
    for attempt in range(5):
        try:
            with session().get(BASE, params=params, timeout=(20, 60)) as reply:
                reply.raise_for_status()
                if reply.url.split("?", 1)[0] != BASE:
                    raise ValueError("Unexpected inventory redirect")
                return ET.fromstring(reply.content)
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 10))


def sequence_inventory(prefix):
    params = {"list-type": "2", "prefix": prefix}
    entries = []
    while True:
        tree = list_page(params)
        for obj in tree.findall("s:Contents", NS):
            key = obj.findtext("s:Key", namespaces=NS)
            parts = key.split("/")
            if len(parts) == 5 and parts[2] in {"camera", "label"} and key.endswith(".png"):
                if parts[3] not in CAMERAS:
                    raise ValueError(f"Unexpected camera: {key}")
                entries.append(object_record(obj))
        token = tree.findtext("s:NextContinuationToken", namespaces=NS)
        if not token:
            break
        params["continuation-token"] = token
    return entries


def check_inventory(inventory, strict=True):
    if inventory.get("source") != BASE:
        raise ValueError("Unknown inventory origin")
    entries = inventory["objects"]
    keys, pairs, seqs, total = set(), {}, set(), 0
    for obj in entries:
        key = obj["key"]
        if key.casefold() in keys:
            raise ValueError("Duplicate inventory object")
        keys.add(key.casefold())
        safe_relative(key)
        if not isinstance(obj["bytes"], int) or obj["bytes"] < 1 or not re.fullmatch(r"[0-9a-f]{32}", obj["md5"]):
            raise ValueError("Invalid inventory identity")
        if key in METADATA:
            continue
        parts = key.split("/")
        if len(parts) != 5 or parts[0] != "camera_lidar_semantic" or parts[2] not in {"camera", "label"} or parts[3] not in CAMERAS:
            raise ValueError(f"Unexpected data object: {key}")
        match = re.fullmatch(r"(\d{14})_(camera|label)_([a-z]+)_(\d{9})\.png", parts[4])
        if not match or match.group(2) != parts[2]:
            raise ValueError(f"Unexpected frame name: {key}")
        if match.group(1) != parts[1].replace("_", "") or "cam_" + match.group(3).replace("center", "_center").replace("left", "_left").replace("right", "_right") != parts[3]:
            raise ValueError("Frame camera/recording mismatch")
        pair = (parts[1], parts[3], match.group(4))
        pairs.setdefault(pair, set()).add(parts[2])
        seqs.add(parts[1])
        total += obj["bytes"]
    if any(kinds != {"camera", "label"} for kinds in pairs.values()):
        raise ValueError("Unpaired RGB/label inventory")
    if strict and (len(pairs) != EXPECTED_PAIRS or total != EXPECTED_BYTES or len(seqs) != EXPECTED_SEQUENCES):
        raise ValueError(f"Official inventory changed: pairs={len(pairs)}, bytes={total}, sequences={len(seqs)}")
    if strict and not set(METADATA) <= {obj["key"] for obj in entries}:
        raise ValueError("Missing publisher metadata")
    return {"pairs": len(pairs), "data_bytes": total, "sequences": len(seqs)}


def safe_relative(key):
    if not key or "\\" in key or ":" in key or key.startswith("/"):
        raise ValueError("Unsafe object path")
    raw_parts = key.split("/")
    if any(p in {"", ".", ".."} or p.endswith((".", " ")) or any(ord(c) < 32 or c in '<>"|?*' for c in p) for p in raw_parts):
        raise ValueError("Unsafe object component")
    for p in raw_parts:
        if re.fullmatch(r"(?:CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\..*)?", p, re.I):
            raise ValueError("Reserved Windows object name")
    return PurePosixPath(key)


def target_path(root, key):
    path = root.joinpath(*safe_relative(key).parts)
    # Resolve the nearest existing ancestor strictly. On Windows non-strict
    # resolution of a not-yet-created parent can race with concurrent mkdirs.
    # Every remaining component was already checked to be a simple name.
    canonical_root = root.resolve(strict=True)
    ancestor = path
    while not ancestor.exists() and not ancestor.is_symlink():
        ancestor = ancestor.parent
    resolved_ancestor = ancestor.resolve(strict=True)
    if not resolved_ancestor.is_relative_to(canonical_root) or ancestor.is_symlink():
        raise ValueError(f"Object target escapes download root: key={key!r}, ancestor={resolved_ancestor!s}, root={canonical_root!s}")
    return path


def file_hashes(path):
    md5, sha = hashlib.md5(usedforsecurity=False), hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            md5.update(block)
            sha.update(block)
    return {"bytes": path.stat().st_size, "md5": md5.hexdigest(), "sha256": sha.hexdigest()}


def download_one(root, obj):
    path = target_path(root, obj["key"])
    partial = path.with_suffix(path.suffix + ".part")
    if partial.is_symlink():
        raise ValueError("Refusing symlink partial file")
    if path.exists():
        result = file_hashes(path)
        if result["bytes"] != obj["bytes"] or result["md5"] != obj["md5"]:
            raise ValueError(f"Existing object identity mismatch: {obj['key']}")
        return {"key": obj["key"], **result, "existing": True}
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.parent.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("Object parent changed")
    for attempt in range(5):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > obj["bytes"]:
            raise ValueError("Partial object exceeds expected size")
        if offset == obj["bytes"]:
            break
        headers = {"If-Match": '"' + obj["md5"] + '"'}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with session().get(BASE + obj["key"], headers=headers, stream=True, timeout=(20, 60)) as reply:
                reply.raise_for_status()
                if reply.url != BASE + obj["key"] or reply.headers.get("ETag", "").strip('"') != obj["md5"]:
                    raise ValueError("Publisher object identity changed")
                expected = obj["bytes"] - offset
                if reply.status_code != (206 if offset else 200):
                    raise ValueError("Server did not honor requested range")
                if offset and reply.headers.get("Content-Range") != f"bytes {offset}-{obj['bytes'] - 1}/{obj['bytes']}":
                    raise ValueError("Unexpected Content-Range")
                if int(reply.headers.get("Content-Length", -1)) != expected:
                    raise ValueError("Unexpected response size")
                received = 0
                with partial.open("ab" if offset else "wb") as target:
                    for block in reply.iter_content(1024 * 1024):
                        received += len(block)
                        if received > expected:
                            raise ValueError("Response exceeds pinned object size")
                        target.write(block)
                if received != expected:
                    raise requests.RequestException("Incomplete object response")
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 10))
    result = file_hashes(partial)
    if result["bytes"] != obj["bytes"] or result["md5"] != obj["md5"]:
        raise ValueError(f"Publisher MD5 mismatch: {obj['key']}; partial preserved")
    partial.rename(path)
    return {"key": obj["key"], **result, "existing": False}


def build_inventory(root, workers):
    tree = list_page({"list-type": "2", "prefix": PREFIX, "delimiter": "/"})
    sequences = sorted(node.findtext("s:Prefix", namespaces=NS) for node in tree.findall("s:CommonPrefixes", NS))
    if len(sequences) != EXPECTED_SEQUENCES:
        raise ValueError("Unexpected source recording count")
    entries = []
    with ThreadPoolExecutor(max_workers=min(workers, 8)) as pool:
        for sequence, objects in zip(sequences, pool.map(sequence_inventory, sequences)):
            entries.extend(objects)
            emit("inventory_progress", sequence=sequence, files=len(entries))
    for key in METADATA:
        tree = list_page({"list-type": "2", "prefix": key})
        exact = [obj for obj in tree.findall("s:Contents", NS) if obj.findtext("s:Key", namespaces=NS) == key]
        if len(exact) != 1:
            raise ValueError(f"Publisher metadata missing: {key}")
        entries.append(object_record(exact[0]))
    inventory = {"source": BASE, "license": "CC BY-ND 4.0", "objects": sorted(entries, key=lambda x: x["key"]),
                 "lidar_downloaded": False, "synchronized_multiview_samples": False,
                 "publisher": "https://a2d2-dataset.github.io/", "created_unix": time.time()}
    inventory["summary"] = check_inventory(inventory)
    save_json(root / "source_inventory.json", inventory)
    return inventory


def download(root, workers=16, inventory_only=False):
    if not 1 <= workers <= 32:
        raise ValueError("workers must be between 1 and 32")
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    inventory_path = root / "source_inventory.json"
    inventory = json.loads(inventory_path.read_text(encoding="utf-8")) if inventory_path.exists() else build_inventory(root, workers)
    summary = check_inventory(inventory)
    inventory_sha = hashlib.sha256(inventory_path.read_bytes()).hexdigest()
    emit("inventory_verified", **summary, inventory_sha256=inventory_sha)
    if inventory_only:
        return
    # Download labels first, then imagery. Retain original files for independent
    # label audits and future higher-resolution training.
    objects = sorted(inventory["objects"], key=lambda obj: (0 if obj["key"] in METADATA else 1 if "/label/" in obj["key"] else 2, obj["key"]))
    verified, checked_bytes, started, last_log = [], 0, time.monotonic(), time.monotonic()
    todo = iter(objects)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(download_one, root, obj) for obj in [next(todo) for _ in range(min(len(objects), workers * 3))]}
        while pending:
            done, pending = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
            for future in done:
                result = future.result()
                verified.append(result)
                checked_bytes += result["bytes"]
                obj = next(todo, None)
                if obj is not None:
                    pending.add(pool.submit(download_one, root, obj))
            if time.monotonic() - last_log >= 30:
                emit("download_progress", verified_files=len(verified), total_files=len(objects), verified_bytes=checked_bytes,
                     total_bytes=sum(obj["bytes"] for obj in objects), elapsed_seconds=round(time.monotonic() - started, 1))
                last_log = time.monotonic()
    save_json(root / "DOWNLOAD_COMPLETE.json", {"complete": True, "source": BASE, "inventory_sha256": inventory_sha,
              "verified_files": sorted(verified, key=lambda obj: obj["key"]), "summary": summary,
              "license": "CC BY-ND 4.0", "original_files_modified": False,
              "synchronized_multiview_samples": False})
    emit("download_complete", files=len(verified), verified_bytes=checked_bytes, pairs=summary["pairs"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    download(args.root, args.workers, args.inventory_only)


if __name__ == "__main__":
    main()
