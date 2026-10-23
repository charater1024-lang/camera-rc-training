import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading

import numpy as np
from PIL import Image
import pytest


@pytest.fixture
def exporter():
    path = Path(__file__).parents[1] / "scripts/prepare_yolo_semantic.py"
    spec = importlib.util.spec_from_file_location("prepare_yolo_semantic_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def fixture(tmp_path):
    root = tmp_path / "한글 source data"
    root.mkdir()
    rows = {}
    for split, count in (("train", 4), ("val", 2)):
        rows[split] = []
        for index in range(count):
            stem = f"{split}_{index}"
            image, mask = root / f"{stem}.png", root / f"{stem}_mask.png"
            Image.new("RGB", (32, 24), (30 + 25 * index, 40 if split == "train" else 90, 100)).save(image)
            values = np.ones((24, 32), np.uint8)
            values[:, :16] = 0
            values[0, :] = 255
            Image.fromarray(values).save(mask)
            rows[split].append({
                "image": image.name, "target": mask.name, "target_original": mask.name,
                "task": "segmentation", "label_space_id": "road_surface_v1",
                "dataset_id": f"source_{index % 2}", "group_id": stem,
                "origin_id": f"original_drive:{stem}", "sample_id": stem,
                "source_scene": f"scene_{index}", "source_split": split,
            })
    config = {
        "task": "segmentation", "model": {"num_classes": 2, "width": 16},
        "data": {"label_space_id": "road_surface_v1", "image_size": [16, 16],
                 "train_manifest": "train.jsonl", "val_manifest": "val.jsonl"},
        "training": {"device": "cuda"},
    }

    def save():
        for split in rows:
            (root / f"{split}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows[split]), encoding="utf-8")
        (root / "config.json").write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    save()
    return root, rows, config, save


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_full_export_native_pixels_metadata_and_completion_binding(exporter, fixture, tmp_path):
    root, originals, _, _ = fixture
    before = {str(path): exporter.sha256(path) for path in root.iterdir() if path.is_file()}
    output = tmp_path / "exported"
    report = exporter.prepare(root / "config.json", output)
    assert report["complete"] is True and report["diagnostic_only"] is False
    assert report["export_counts"] == {"train": {"source_0": 2, "source_1": 2},
                                        "val": {"source_0": 1, "source_1": 1}}
    assert report["input_split_identity_leakage_checked"] is True
    assert not (output / ".incomplete").exists()
    marker = json.loads((output / ".complete").read_text())
    assert marker["report_sha256"] == exporter.sha256(output / "report.json")
    assert marker["dataset_yaml_sha256"] == exporter.sha256(output / "dataset.yaml")
    data = json.loads((output / "dataset.yaml").read_text())
    assert data["names"] == {"0": "road_surface", "1": "not_road_surface"}
    assert data["masks_dir"] == "masks" and data["nc"] == 2
    assert Path(data["train"]).is_absolute() and Path(data["val"]).is_absolute()
    rows = read_rows(output / "train.jsonl")
    for index, row in enumerate(rows):
        assert row["origin_id"] == originals["train"][index]["origin_id"]
        assert row["source_scene"] == originals["train"][index]["source_scene"]
        assert row["source_class_ids"] == [0, 1, 255]
        assert row["width"] == 32 and row["height"] == 24  # NOT config's 16x16
        assert row["input_target_sha256"] == exporter.sha256(row["input_target"])
        assert row["image_sha256"] == exporter.sha256(row["input_image"])
        assert row["target_sha256"] == exporter.sha256(row["target"])
        with Image.open(row["target"]) as target, Image.open(row["input_target"]) as source:
            assert target.mode == "L" and target.size == (32, 24)
            np.testing.assert_array_equal(np.asarray(target), np.asarray(source))
    assert {str(path): exporter.sha256(path) for path in root.iterdir() if path.is_file()} == before
    assert report["manifest_outputs"]["train"]["sha256"] == exporter.sha256(output / "train.jsonl")


def test_limits_apply_to_each_dataset_inside_merged_manifest(exporter, fixture, tmp_path):
    root, _, _, _ = fixture
    report = exporter.prepare(root / "config.json", tmp_path / "limited",
                              limit_train_per_source=1, limit_val_per_source=1)
    assert report["diagnostic_only"] is True
    assert report["full_input_counts"]["train"] == {"source_0": 2, "source_1": 2}
    assert report["export_counts"]["train"] == {"source_0": 1, "source_1": 1}
    assert {row["dataset_id"] for row in read_rows(tmp_path / "limited/train.jsonl")} == {"source_0", "source_1"}


def test_deterministic_ids_do_not_depend_on_output_folder(exporter, fixture, tmp_path):
    root, _, _, _ = fixture
    for name in ("first", "second"):
        exporter.prepare(root / "config.json", tmp_path / name, limit_train_per_source=1)
    first = read_rows(tmp_path / "first/train.jsonl")
    second = read_rows(tmp_path / "second/train.jsonl")
    assert [row["export_id"] for row in first] == [row["export_id"] for row in second]


def test_existing_output_is_not_modified(exporter, fixture, tmp_path):
    root, _, _, _ = fixture
    output = tmp_path / "existing"
    output.mkdir()
    note = output / "keep.txt"
    note.write_text("untouched")
    with pytest.raises(FileExistsError):
        exporter.prepare(root / "config.json", output)
    assert list(output.iterdir()) == [note]
    assert note.read_text() == "untouched"


def test_copy_fallback_does_not_modify_sources(exporter, fixture, tmp_path, monkeypatch):
    root, _, _, _ = fixture
    monkeypatch.setattr(exporter.os, "link", lambda *a: (_ for _ in ()).throw(OSError("cross volume")))
    report = exporter.prepare(root / "config.json", tmp_path / "copied")
    assert report["image_storage_counts"] == {"copy": 6}
    for row in read_rows(tmp_path / "copied/train.jsonl"):
        assert exporter.sha256(row["input_image"]) == exporter.sha256(row["image"])


@pytest.mark.parametrize("kind", ["unknown_id", "rgb_mask", "all_ignore", "wrong_dimensions"])
def test_bad_masks_fail_with_incomplete_state(exporter, fixture, tmp_path, kind):
    root, rows, _, _ = fixture
    path = root / rows["train"][0]["target"]
    if kind == "unknown_id":
        Image.new("L", (32, 24), 7).save(path)
    elif kind == "rgb_mask":
        Image.new("RGB", (32, 24), (0, 1, 0)).save(path)
    elif kind == "all_ignore":
        Image.new("L", (32, 24), 255).save(path)
    else:
        Image.new("L", (30, 20), 0).save(path)
    output = tmp_path / "bad"
    with pytest.raises(ValueError):
        exporter.prepare(root / "config.json", output)
    assert (output / ".incomplete").is_file()
    assert not (output / ".complete").exists()
    assert not (output / "dataset.yaml").exists()
    assert json.loads((output / "report.json").read_text())["complete"] is False


def test_full_origin_leakage_checked_before_diagnostic_limits(exporter, fixture, tmp_path):
    root, rows, _, save = fixture
    rows["val"][0]["origin_id"] = rows["train"][3]["origin_id"]
    save()
    with pytest.raises(ValueError, match="leakage"):
        exporter.prepare(root / "config.json", tmp_path / "leak", limit_train_per_source=1)


def test_reencoded_same_pixels_across_splits_are_rejected(exporter, fixture, tmp_path):
    root, rows, _, _ = fixture
    source = root / rows["train"][0]["image"]
    destination = root / rows["val"][0]["image"]
    with Image.open(source) as image:
        image.save(destination, compress_level=0)
    assert exporter.sha256(source) != exporter.sha256(destination)
    with pytest.raises(ValueError, match="rgb SHA256 leakage"):
        exporter.prepare(root / "config.json", tmp_path / "duplicate")


def test_unapplied_row_mapping_is_not_silently_reinterpreted(exporter, fixture, tmp_path):
    root, rows, _, save = fixture
    rows["train"][0]["label_map"] = {"0": 1, "1": 0}
    save()
    with pytest.raises(ValueError, match="does not apply row.label_map"):
        exporter.prepare(root / "config.json", tmp_path / "mapping")


def test_declared_source_checksum_mismatch_rejected(exporter, fixture, tmp_path):
    root, rows, _, save = fixture
    rows["train"][0]["image_sha256"] = "0" * 64
    save()
    with pytest.raises(ValueError, match="SHA256 differs"):
        exporter.prepare(root / "config.json", tmp_path / "changed")


def test_noncanonical_config_is_rejected(exporter, fixture, tmp_path):
    root, _, config, save = fixture
    config["data"]["label_space_id"] = "unknown"
    save()
    with pytest.raises(ValueError, match="label_space_id"):
        exporter.prepare(root / "config.json", tmp_path / "wrong_labels")


def test_img2label_paths_matches_installed_semantic_contract(exporter, fixture, tmp_path):
    if importlib.util.find_spec("ultralytics") is None:
        pytest.skip("Optional Ultralytics installation unavailable")
    root, _, _, _ = fixture
    output = tmp_path / "format"
    exporter.prepare(root / "config.json", output, limit_train_per_source=1)
    rows = read_rows(output / "train.jsonl")
    # Importing Ultralytics patches PIL.Image.open, torch.save and Windows cv2
    # process-wide. Keep that optional integration inside a disposable process;
    # otherwise later corrupt-PNG/JPEG tests can try HEIF package installation
    # and receive ModuleNotFoundError instead of Pillow's original OSError.
    original_open = Image.open
    environment = dict(os.environ, YOLO_CONFIG_DIR=str(tmp_path / "ultralytics_settings"),
                       YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false", PYTHONDONTWRITEBYTECODE="1")
    script = (
        "import json, sys\n"
        "from ultralytics.data.utils import img2label_paths\n"
        "images = json.load(sys.stdin)\n"
        "print('CONTRACT_JSON=' + json.dumps(img2label_paths(images, label_dir='masks', suffix='.png')))\n"
    )
    completed = subprocess.run([sys.executable, "-X", "utf8", "-B", "-c", script],
                               input=json.dumps([row["image"] for row in rows]), capture_output=True,
                               text=True, encoding="utf-8", env=environment, timeout=45, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    result = [line.removeprefix("CONTRACT_JSON=") for line in completed.stdout.splitlines()
              if line.startswith("CONTRACT_JSON=")]
    assert len(result) == 1
    paths = json.loads(result[0])
    assert [Path(path).resolve() for path in paths] == [Path(row["target"]).resolve() for row in rows]
    assert Image.open is original_open


def test_parallel_matches_serial_identity_pixels_and_source_order(exporter, fixture, tmp_path, monkeypatch):
    root, originals, _, _ = fixture
    before = {str(path): exporter.sha256(path) for path in root.iterdir() if path.is_file()}
    serial = tmp_path / "serial"
    first = exporter.prepare(root / "config.json", serial, workers=1)
    decode = exporter._decode_image_and_mask
    second_started = threading.Event()
    completion_order = []

    def delayed_first(image, target):
        # Deliberately finish row 1 before row 0; output must still match source order.
        if image.stem == "train_0":
            assert second_started.wait(timeout=5)
        result = decode(image, target)
        completion_order.append(image.stem)
        if image.stem == "train_1":
            second_started.set()
        return result

    monkeypatch.setattr(exporter, "_decode_image_and_mask", delayed_first)
    parallel = tmp_path / "parallel"
    second = exporter.prepare(root / "config.json", parallel, workers=4)
    assert completion_order.index("train_1") < completion_order.index("train_0")
    for split in ("train", "val"):
        a, b = read_rows(serial / f"{split}.jsonl"), read_rows(parallel / f"{split}.jsonl")
        assert [row["sample_id"] for row in b] == [row["sample_id"] for row in originals[split]]
        for left, right in zip(a, b):
            # Output-root differences are expected; all provenance/content must match.
            assert {k: v for k, v in left.items() if k not in {"image", "target"}} == {
                k: v for k, v in right.items() if k not in {"image", "target"}}
            assert exporter.sha256(left["image"]) == exporter.sha256(right["image"])
            assert exporter.sha256(left["target"]) == exporter.sha256(right["target"])
    for key in ("full_input_counts", "export_counts", "target_pixel_counts", "source_inputs_fingerprinted"):
        assert first[key] == second[key]
    assert second["preprocessing_max_pending"] == 8
    assert {str(path): exporter.sha256(path) for path in root.iterdir() if path.is_file()} == before


def test_preprocessing_bounds_prefetch_and_cancels_on_consumer_error(exporter, monkeypatch):
    state = {"submitted": 0, "outstanding": 0, "maximum": 0, "cancelled": 0, "shutdown": None}

    class Future:
        def __init__(self, index):
            self.index = index

        def result(self):
            state["outstanding"] -= 1
            return self.index

        def cancel(self):
            state["cancelled"] += 1

    class Executor:
        def __init__(self, **kwargs):
            assert kwargs["max_workers"] == 3

        def submit(self, function, image, target):
            index = state["submitted"]
            state["submitted"] += 1
            state["outstanding"] += 1
            state["maximum"] = max(state["maximum"], state["outstanding"])
            return Future(index)

        def shutdown(self, **kwargs):
            state["shutdown"] = kwargs

    monkeypatch.setattr(exporter, "ThreadPoolExecutor", Executor)
    items = [(None, index, {"image": f"{index}.jpg", "target": f"{index}.png"}) for index in range(100)]
    with pytest.raises(RuntimeError, match="consumer failed"):
        with exporter._ordered_preprocessed(items, 3, lambda item: item[1]) as rows:
            assert [next(rows)[2] for _ in range(3)] == [0, 1, 2]
            assert state["submitted"] == 8  # initial six, then one refill per consumed result
            raise RuntimeError("consumer failed")
    assert state["maximum"] == 6
    assert state["cancelled"] == 5
    assert state["shutdown"] == {"wait": True, "cancel_futures": True}


def test_worker_failure_is_incomplete_and_all_reader_threads_finish(exporter, fixture, tmp_path, monkeypatch):
    root, _, _, _ = fixture
    decode = exporter._decode_image_and_mask
    active = []

    def fail(image, target):
        active.append(threading.current_thread())
        if image.stem == "train_1":
            raise OSError("injected decoder failure")
        return decode(image, target)

    monkeypatch.setattr(exporter, "_decode_image_and_mask", fail)
    output = tmp_path / "worker_error"
    with pytest.raises(OSError, match="injected decoder failure"):
        exporter.prepare(root / "config.json", output, workers=4)
    assert active and not any(thread.is_alive() for thread in active)
    assert (output / ".incomplete").is_file()
    assert not (output / ".complete").exists()
    assert json.loads((output / "report.json").read_text())["complete"] is False


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_canonical_histogram_matches_counts_without_sorting(exporter, tmp_path, monkeypatch, dtype):
    values = np.array([[0, 0, 1, 255], [1, 1, 255, 0]], dtype=dtype)
    path = tmp_path / "ids.png"
    Image.fromarray(values).save(path)
    monkeypatch.setattr(exporter.np, "unique", lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("canonical masks must not require sorting")))
    result, counts = exporter._mask(path, (4, 2))
    np.testing.assert_array_equal(result, values.astype(np.uint8))
    assert counts == {"0": 3, "1": 3, "255": 2}


@pytest.mark.parametrize("invalid_id", [2, 254, 256, 65535])
def test_invalid_16bit_ids_cannot_wrap_to_canonical(exporter, tmp_path, invalid_id):
    path = tmp_path / "unknown.png"
    Image.fromarray(np.array([[0, 1], [255, invalid_id]], dtype=np.uint16)).save(path)
    with pytest.raises(ValueError, match="Unexpected canonical class IDs"):
        exporter._mask(path, (2, 2))


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_invalid_worker_count_leaves_no_output(exporter, fixture, tmp_path, workers):
    root, _, _, _ = fixture
    output = tmp_path / "invalid_workers"
    with pytest.raises(ValueError, match="workers"):
        exporter.prepare(root / "config.json", output, workers=workers)
    assert not output.exists()
