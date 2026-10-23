"""Fetch fixed public research datasets; verify identities and never execute assets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile

import requests

import download_bdd_archives as bdd

COMMIT = "bb68b64e72cb004b3ed927fee0f5624dd8097498"
REPO = "alexgkendall/SegNet-Tutorial"
KITTI = "https://s3.eu-central-1.amazonaws.com/avg-kitti/"
ASSETS = {
    "kitti": [(KITTI + "data_road.zip", 470992343, None, '"ef2cedcbbf5f4122c9ead516342d0b71-57"'),
              (KITTI + "devkit_road_mapping.zip", 1480, None, None),
              (KITTI + "devkit_road.zip", 22283, None, None)],
    "obstacle": [("https://zenodo.org/records/5281633/files/dataset_ObstacleTrack.zip", 207582233,
                  "895fb36d18765482cc291f69e63d6da6", None)],
    "anomaly": [("https://zenodo.org/records/5270237/files/dataset_AnomalyTrack.zip", 52625353,
                 "231bf79ed58924bcd33d9cbe22e61076", None)],
}


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def hashes(path):
    sha, md5 = hashlib.sha256(), hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(block)
            md5.update(block)
    return {"sha256": sha.hexdigest(), "md5": md5.hexdigest(), "bytes": path.stat().st_size}


def fetch(url, target, size, md5=None, etag=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    if target.is_symlink() or partial.is_symlink():
        raise ValueError("Refusing symbolic link download targets")
    last_error = None
    if not target.exists():
        for attempt in range(5):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == size:
                break
            if offset > size:
                raise ValueError("Partial file exceeds expected size")
            headers = {"Accept-Encoding": "identity"}
            if offset:
                headers["Range"] = f"bytes={offset}-"
            if etag:
                headers["If-Match"] = etag
            try:
                with requests.get(url, stream=True, headers=headers, timeout=(15, 45)) as response:
                    if not response.url.startswith("https://"):
                        raise ValueError("Insecure redirect")
                    expected = bdd._check_response(response, offset, size)
                    if etag and response.headers.get("ETag") != etag:
                        raise ValueError("Publisher ETag changed")
                    received, started = 0, time.monotonic()
                    with partial.open("ab" if offset else "wb") as stream:
                        for block in response.iter_content(1024 * 1024):
                            if received + len(block) > expected:
                                raise ValueError("Response exceeds advertised range")
                            stream.write(block)
                            received += len(block)
                            if time.monotonic() - started > 25:
                                bdd.emit("download_progress", file=target.name, bytes=offset + received, total=size)
                                started = time.monotonic()
                    if received != expected or offset + received != size:
                        raise requests.RequestException("Incomplete response; resume next attempt")
                break
            except requests.RequestException as error:
                last_error = error
                bdd.emit("download_retry", file=target.name, attempt=attempt + 1, error=str(error))
                time.sleep(min(2 ** attempt, 15))
        if not partial.exists() or partial.stat().st_size != size:
            raise RuntimeError(f"Download incomplete: {target}") from last_error
        result = hashes(partial)
        if md5 and result["md5"] != md5:
            raise ValueError("Publisher checksum mismatch; partial preserved")
        partial.rename(target)
    result = hashes(target)
    if result["bytes"] != size or md5 and result["md5"] != md5:
        raise ValueError(f"Existing file identity mismatch: {target}")
    result.update(url=url, path=str(target.resolve()), publisher_md5=md5,
                  publisher_etag=etag, publisher_hash_available=md5 is not None)
    bdd.emit("download_verified", **result)
    return result


def extract(archive, destination, identity):
    marker = destination / "EXTRACTION_COMPLETE.json"
    if marker.exists():
        prior = json.loads(marker.read_text(encoding="utf-8"))
        if prior["archive_sha256"] != identity["sha256"]:
            raise ValueError("Extraction provenance mismatch")
        return
    if destination.exists():
        raise FileExistsError(f"Incomplete or unknown extraction exists: {destination}")
    with zipfile.ZipFile(archive) as zipped:
        members, seen, total = [], set(), 0
        for info in zipped.infolist():
            rel = bdd._member_path(info)
            key = str(rel).casefold()
            if key in seen:
                raise ValueError("Case-insensitive duplicate ZIP member")
            seen.add(key)
            total += info.file_size
            if total > 4 * 1024**3:
                raise ValueError("Expanded archive exceeds fixed budget")
            members.append((info, rel))
        destination.mkdir(parents=True)
        (destination / "INCOMPLETE").touch()
        for info, rel in members:
            path = destination.joinpath(*rel.parts)
            if info.is_dir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            with zipped.open(info) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)  # ZipFile checks each member CRC.
        save_json(marker, {"archive_sha256": identity["sha256"], "files": len(members),
                           "expanded_bytes": total, "zip_crc_checked": True})
        (destination / "INCOMPLETE").unlink()


def camvid(destination):
    destination.mkdir(parents=True, exist_ok=True)
    url = f"https://api.github.com/repos/{REPO}/git/trees/{COMMIT}?recursive=1"
    response = requests.get(url, timeout=(15, 45))
    response.raise_for_status()
    tree = response.json()
    if tree.get("truncated"):
        raise ValueError("Truncated source inventory")
    folders = {"train", "val", "test", "trainannot", "valannot", "testannot"}
    assets = [item for item in tree["tree"] if item["type"] == "blob" and
              len(item["path"].split("/")) == 3 and item["path"].split("/")[0] == "CamVid" and
              item["path"].split("/")[1] in folders and item["path"].endswith(".png")]
    if len(assets) != 1402 or sum(item["size"] for item in assets) != 186792534:
        raise ValueError("Pinned CamVid inventory mismatch")
    assets += [item for item in tree["tree"] if item["type"] == "blob" and item["path"] == "README.md"]
    save_json(destination / "source_inventory.json", {"url": url, "commit": COMMIT, "assets": assets})

    def one(item):
        path = destination / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError("Symlink source target")
        raw_url = f"https://raw.githubusercontent.com/{REPO}/{COMMIT}/{item['path']}"
        if path.exists():
            content = path.read_bytes()
        else:
            for attempt in range(4):
                try:
                    reply = requests.get(raw_url, timeout=(15, 45))
                    reply.raise_for_status()
                    content = reply.content
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        blob = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        if len(content) != item["size"] or blob != item["sha"]:
            raise ValueError(f"Git blob verification failed: {item['path']}")
        if not path.exists():
            with path.open("xb") as stream:
                stream.write(content)
        return {"path": item["path"], "git_blob_sha1": blob, "sha256": hashlib.sha256(content).hexdigest()}

    verified = []
    with ThreadPoolExecutor(max_workers=8) as pool:
        pending = [pool.submit(one, item) for item in assets]
        for result in as_completed(pending):
            verified.append(result.result())
            if len(verified) % 100 == 0:
                bdd.emit("camvid_progress", verified=len(verified), total=len(assets))
    save_json(destination / "DOWNLOAD_COMPLETE.json", {"commit": COMMIT, "source": url,
              "verified_files": sorted(verified, key=lambda row: row["path"]),
              "original_publisher": "https://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/",
              "license_note": "Preserve original dataset attribution; repository software license does not relicense original imagery."})
    bdd.emit("camvid_complete", files=len(verified), image_pairs=701)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", choices=["camvid", *ASSETS])
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.dataset == "camvid":
        camvid(root)
    else:
        results = []
        for url, size, md5, etag in ASSETS[args.dataset]:
            path = root / "archives" / url.rsplit("/", 1)[1]
            result = fetch(url, path, size, md5, etag)
            extract(path, root / path.stem, result)
            results.append(result)
        save_json(root / "DOWNLOAD_COMPLETE.json", {"dataset": args.dataset, "archives": results})


if __name__ == "__main__":
    main()
