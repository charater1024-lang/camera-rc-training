"""Download the fixed official VKITTI2 RGB, semantic and calibration archives."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tarfile

from download_bdd_archives import download_chunked, emit, verify_file

BASE = "https://download.europe.naverlabs.com/virtual_kitti_2.0.3/"
SPECS = {
    "rgb": {"filename": "vkitti_2.0.3_rgb.tar", "size": 7532472320, "md5": "1e00a143a397c2c53aa9720a868fe34a"},
    "semantic": {"filename": "vkitti_2.0.3_classSegmentation.tar", "size": 1015961600, "md5": "e8658ab49250e61f1caa174625563263"},
    "metadata": {"filename": "vkitti_2.0.3_textgt.tar.gz", "size": 24451078, "md5": "855f7f37746e5508f094e522c9cf41f4"},
}
for spec in SPECS.values():
    spec["url"] = BASE + spec["filename"]


def safe_path(member):
    name = member.name
    while name.startswith("./"):
        name = name[2:]
    if member.isdir() and name in ("", "."):
        return None
    if name.startswith("/") or "\\" in name or ":" in name:
        raise ValueError("Unsafe TAR path")
    parts = name.rstrip("/").split("/")
    reserved = re.compile(r"^(CON|PRN|AUX|NUL|COM[1-9]|LPT[1-9])(?:\.|$)", re.I)
    if any(not p or p in (".", "..") or p.endswith((" ", ".")) or reserved.match(p)
           or any(ord(c) < 32 or c in '<>"|?*' for c in p) for p in parts):
        raise ValueError("Unsafe TAR component")
    if not (member.isdir() or member.isfile()):
        raise ValueError("Linked/special TAR entries are forbidden")
    if member.size > 128 * 1024**2:
        raise ValueError("Unexpectedly large TAR member")
    return PurePosixPath(*parts)


def extract(key, archive, output, identity):
    complete = output / "EXTRACTION_COMPLETE.json"
    if complete.exists():
        prior = json.loads(complete.read_text(encoding="utf-8"))
        if prior["archive_sha256"] != identity["sha256"]:
            raise ValueError("Extraction identity mismatch")
        return prior
    if output.exists():
        raise FileExistsError(f"Unknown/incomplete extraction preserved: {output}")
    output.mkdir(parents=True)
    (output / "INCOMPLETE").touch()
    seen, count, total = set(), 0, 0
    with tarfile.open(archive, "r|*") as tar:
        for member in tar:
            relative = safe_path(member)
            if relative is None:
                continue
            canonical = str(relative).casefold()
            if canonical in seen:
                raise ValueError("Duplicate TAR member")
            seen.add(canonical)
            path = output.joinpath(*relative.parts)
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            total += member.size
            if total > 50 * 1024**3:
                raise ValueError("Expansion budget exceeded")
            path.parent.mkdir(parents=True, exist_ok=True)
            with tar.extractfile(member) as source, path.open("xb") as target:
                shutil.copyfileobj(source, target)
            if path.stat().st_size != member.size:
                raise ValueError("Extracted member size mismatch")
            count += 1
            if count % 10000 == 0:
                emit("tar_extract_progress", asset=key, files=count)
    result = {"asset": key, "archive_sha256": identity["sha256"], "files": count, "expanded_bytes": total}
    complete.write_text(json.dumps(result, indent=2), encoding="utf-8")
    (output / "INCOMPLETE").unlink()
    emit("tar_extract_complete", **result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--assets", nargs="+", choices=list(SPECS), default=list(SPECS))
    args = parser.parse_args()
    root = args.root.resolve()
    results = {}
    # Each asset has bounded concurrency; sequential assets cap total at workers.
    for key in args.assets:
        identity = download_chunked(key, root / "archives", workers=args.workers,
                                    chunk_bytes=32 * 1024**2, specs=SPECS)
        result = extract(key, Path(identity["path"]), root / key, identity)
        results[key] = {"download": identity, "extraction": result}
        (root / (key + "_complete.json")).write_text(json.dumps(results[key], ensure_ascii=False, indent=2), encoding="utf-8")
    (root / "requested_assets_complete.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
