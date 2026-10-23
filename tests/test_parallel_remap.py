import json
from pathlib import Path
import threading

import numpy as np
from PIL import Image
import pytest

from camera_rc.data import read_manifest, remap_masks


def sources(root, count=7):
    root.mkdir()
    rows = []
    for index in range(count):
        image, target = root / f"image-{index}.jpg", root / f"target-{index}.png"
        Image.new("RGB", (32, 24), (index * 20, 30, 40)).save(image)
        labels = np.tile(np.array([0, 1, 2, 3, 255, 0, 1, 2], np.uint8), (24, 4))
        labels[index, :] = index % 4
        Image.fromarray(labels).save(target)
        rows.append({"image": str(image), "target": str(target), "group_id": f"scene-{index}",
                     "dataset_id": "test", "sample_id": str(index)})
    manifest = root / "source.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    mapping = root / "mapping.json"
    mapping.write_text(json.dumps({"label_space_id": "road_surface_v1", "num_classes": 2,
                                   "mapping": {"0": 0, "1": 0, "2": 1, "255": 255}}), encoding="utf-8")
    return manifest, mapping, rows


def test_parallel_conversion_matches_sequential_bytes_counts_and_order(tmp_path):
    manifest, mapping, originals = sources(tmp_path / "original")
    before = {row[key]: Path(row[key]).read_bytes() for row in originals for key in ("image", "target")}
    sequential = remap_masks(manifest, mapping, tmp_path / "sequential")
    progress = []
    caller = threading.get_ident()
    parallel = remap_masks(manifest, mapping, tmp_path / "parallel", workers=2,
                           progress=lambda done, total: progress.append((done, total, threading.get_ident())))
    for key in ("samples", "source_pixel_counts", "target_pixel_counts",
                "undeclared_ids_mapped_to_ignore", "mapping", "complete"):
        assert parallel[key] == sequential[key]
    assert progress == [(index, len(originals), caller) for index in range(1, len(originals) + 1)]
    sequential_rows, parallel_rows = read_manifest(sequential["manifest"]), read_manifest(parallel["manifest"])
    assert [row["sample_id"] for row in parallel_rows] == [str(i) for i in range(len(originals))]
    for left, right in zip(sequential_rows, parallel_rows):
        assert Path(left["target"]).read_bytes() == Path(right["target"]).read_bytes()
        assert Path(left["target"]).name == Path(right["target"]).name
        assert {key: value for key, value in left.items() if key != "target"} == {
            key: value for key, value in right.items() if key != "target"}
    assert all(Path(path).read_bytes() == contents for path, contents in before.items())
    assert not (tmp_path / "parallel/.incomplete").exists()


def test_parallel_failure_finishes_workers_and_preserves_incomplete_output(tmp_path):
    manifest, mapping, rows = sources(tmp_path / "original")
    # The file still exists, so manifest validation succeeds; decoding fails in
    # a conversion worker after output creation.
    Path(rows[2]["target"]).write_bytes(b"not an image")
    output = tmp_path / "failed"
    threads_before = set(threading.enumerate())
    with pytest.raises(Exception):
        remap_masks(manifest, mapping, output, workers=2)
    assert (output / ".incomplete").exists()
    assert not (output / "manifest.jsonl").exists()
    assert not (output / "report.json").exists()
    assert not [thread for thread in threading.enumerate()
                if thread not in threads_before and thread.name.startswith("ThreadPoolExecutor")]
    assert Path(rows[2]["target"]).read_bytes() == b"not an image"


@pytest.mark.parametrize("workers", [-1, 9, True, False, 1.5, "2", None])
def test_invalid_workers_rejected_before_output_creation(tmp_path, workers):
    with pytest.raises(ValueError, match="workers"):
        remap_masks(tmp_path / "missing.jsonl", tmp_path / "missing.json",
                    tmp_path / "output", workers=workers)
    assert not (tmp_path / "output").exists()


def test_invalid_progress_rejected_before_output_creation(tmp_path):
    with pytest.raises(ValueError, match="progress"):
        remap_masks(tmp_path / "missing.jsonl", tmp_path / "missing.json",
                    tmp_path / "output", progress=42)
    assert not (tmp_path / "output").exists()


def test_parallel_inflight_jobs_are_bounded(tmp_path, monkeypatch):
    from concurrent import futures

    manifest, mapping, rows = sources(tmp_path / "original", count=20)
    real_executor = futures.ThreadPoolExecutor
    submitted, collected, peak = 0, 0, 0

    class TrackingExecutor:
        def __init__(self, max_workers):
            self.inner = real_executor(max_workers=max_workers)

        def submit(self, fn, *args):
            nonlocal submitted, peak
            submitted += 1
            peak = max(peak, submitted - collected)
            return self.inner.submit(fn, *args)

        def shutdown(self, **kwargs):
            return self.inner.shutdown(**kwargs)

    def progress(done, total):
        nonlocal collected
        collected = done

    monkeypatch.setattr(futures, "ThreadPoolExecutor", TrackingExecutor)
    result = remap_masks(manifest, mapping, tmp_path / "bounded", workers=2, progress=progress)
    assert result["samples"] == len(rows)
    assert peak <= 4 and submitted == collected == len(rows)
