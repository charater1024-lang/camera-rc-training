"""Build a new full-data road-training stage without starting training.

Example:
  python scripts/prepare_large_data_stage.py --project PROJECT --name large_v1
      --source-root PREPARED_VKITTI --source-root PREPARED_ACDC
      --validation-root PREPARED_CAMVID --validation-root PREPARED_KITTI
      --init-from PROJECT/runs/bdd100k_2018_baseline_v1/best.pt

Each source-root contributes every training image plus validation and reserved
test origins. Validation-root contributes validation/test only. The entire
original BDD 70k training set is included once per epoch, without replacement.
All prepared roots must have complete reports and matching manifest hashes.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

from camera_rc.config import load_config
from camera_rc.data import read_manifest, validate_splits
from camera_rc.training import _read_checkpoint, _check_compatible, validate_config


BDD_TRAIN_COUNT = 70000
BDD_VAL_COUNT = 10000
BDD_DATASET_ID = "bdd100k_drivable"
LABEL_SPACE = "road_surface_v1"


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _complete_report(root):
    if not root.is_dir():
        raise FileNotFoundError(root)
    if (root / ".incomplete").exists():
        raise ValueError(f"Preparation is incomplete: {root}")
    path = root / "report.json"
    report = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict) or report.get("complete") is not True:
        raise ValueError(f"Preparation report must confirm complete=true: {path}")
    if report.get("label_space_id") != LABEL_SPACE:
        raise ValueError(f"Preparation report must declare {LABEL_SPACE}: {path}")
    return report


def _checked_manifest(root, report, key, expected_path=None, allow_legacy_bdd_origin=False):
    entry = report.get("manifests", {}).get(key)
    if not isinstance(entry, dict) or not isinstance(entry.get("manifest"), str):
        raise ValueError(f"Manifest {key} is absent from completion report: {root}")
    path = Path(entry["manifest"]).expanduser()
    path = (path if path.is_absolute() else root / path).resolve()
    if root not in path.parents:
        raise ValueError(f"Reported manifest escapes prepared root: {path}")
    if expected_path is not None and path != Path(expected_path).resolve():
        raise ValueError(f"Completion report/config manifest mismatch: {path}")
    parent = path.parent
    while True:
        if (parent / ".incomplete").exists():
            raise ValueError(f"Manifest is beneath an incomplete preparation: {path}")
        if parent == root:
            break
        parent = parent.parent
    digest = _sha256(path)
    if entry.get("sha256") != digest:
        raise ValueError(f"Manifest SHA256 differs from completion report: {path}")
    rows = read_manifest(path)
    count = entry.get("samples")
    if isinstance(count, bool) or not isinstance(count, int) or count != len(rows):
        raise ValueError(f"Manifest sample count differs from completion report: {path}")
    for index, row in enumerate(rows, 1):
        if row.get("label_space_id") != LABEL_SPACE or row.get("task") != "segmentation":
            raise ValueError(f"Manifest row must declare segmentation/{LABEL_SPACE}: {path}:{index}")
        if not isinstance(row.get("target"), str) or not Path(row["target"]).is_file():
            raise ValueError(f"Manifest target is missing: {path}:{index}")
        # The verified legacy BDD adapter stored the publisher clip ID only as
        # group_id. Add its canonical namespace in the new stage copy; never
        # rewrite the original manifest or invent origins for other datasets.
        if ("origin_id" not in row and allow_legacy_bdd_origin
                and row.get("dataset_id") == BDD_DATASET_ID):
            row["origin_id"] = "bdd100k:" + row["group_id"]
        if not isinstance(row.get("origin_id"), str) or not row["origin_id"]:
            raise ValueError(f"Canonical origin_id is required: {path}:{index}")
        if row.get("dataset_id") == "default":
            raise ValueError(f"Explicit dataset_id is required: {path}:{index}")
    return rows, {"manifest": str(path), "sha256": digest, "samples": len(rows), "report_key": key}


def _read_prepared_source(root, include_train):
    report = _complete_report(root)
    result, entries = {}, {}
    for split in (("train", "val", "test") if include_train else ("val", "test")):
        expected = root / f"{split}.jsonl"
        declared = split in report.get("manifests", {})
        if split == "test" and not declared and not expected.exists():
            continue  # KITTI Road does not expose labelled test targets.
        result[split], entries[split] = _checked_manifest(root, report, split, expected)
    return result, {"root": str(root), "role": "train_and_validation" if include_train else "validation_only",
                    "report_sha256": _sha256(root / "report.json"), "manifests": entries,
                    "heldout_test_unlabelled": report.get("heldout_test_unlabelled"),
                    "official_test_note": report.get("official_test")}


def _initial_checkpoint(path, base_config):
    checkpoint = _read_checkpoint(path)
    if checkpoint.get("diagnostic") is not False or checkpoint.get("epoch_complete") is not True:
        raise ValueError("Initialization requires a complete, non-diagnostic BDD checkpoint")
    if checkpoint.get("initialization") is not None:
        raise ValueError("Initialization must be the original BDD-only baseline, not a later finetuned checkpoint")
    stored = checkpoint.get("config", {})
    if stored.get("training", {}).get("init_from"):
        raise ValueError("Initialization must not have inherited another checkpoint")
    _check_compatible(base_config, stored)
    data = stored.get("data", {})
    if data.get("train_sources") or not isinstance(data.get("train_manifest"), str):
        raise ValueError("Initialization checkpoint must have one BDD-only training manifest")
    if Path(data["train_manifest"]).resolve() != Path(base_config["data"]["train_manifest"]).resolve():
        raise ValueError("Initialization checkpoint is not from the configured original BDD training set")
    stored_val = data.get("val_manifests", [])
    base_val = base_config["data"].get("val_manifests", [])
    if (len(stored_val) != 1 or not isinstance(stored_val[0], dict)
            or not isinstance(stored_val[0].get("manifest"), str)
            or Path(stored_val[0]["manifest"]).resolve() != Path(base_val[0]["manifest"]).resolve()):
        raise ValueError("Initialization checkpoint must use only the original BDD validation set")
    if any(stored.get("training", {}).get(key) is not None for key in ("max_train_steps", "max_val_steps")):
        raise ValueError("Initialization checkpoint was configured as a diagnostic run")
    validation = checkpoint.get("validation", {})
    if validation.get("partial_validation") is not False:
        raise ValueError("Initialization checkpoint must have completed full validation")
    return {"path": str(path), "sha256": _sha256(path), "epoch": checkpoint.get("epoch"),
            "optimizer_steps": checkpoint.get("optimizer_steps"), "weights_only": True,
            "original_bdd_only_config_checked": True}


def prepare_large_data_stage(project, name, source_root, init_from, validation_root=(), epochs=1,
                             segmentation_class_weights=(1.0, 2.0)):
    if not isinstance(name, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,79}", name):
        raise ValueError("Stage name must contain only lowercase letters, digits and underscores")
    if re.fullmatch(r"con|prn|aux|nul|com[1-9]|lpt[1-9]", name):
        raise ValueError("Stage name must not be a reserved Windows device name")
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError("Epochs must be a positive integer")
    if (not isinstance(segmentation_class_weights, (list, tuple)) or len(segmentation_class_weights) != 2
            or any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x <= 0 for x in segmentation_class_weights)):
        raise ValueError("Exactly two finite positive class weights are required")
    project = Path(project).expanduser().resolve()
    checkpoint_path = Path(init_from).expanduser().resolve()
    source_roots = [Path(x).expanduser().resolve() for x in source_root]
    validation_roots = [Path(x).expanduser().resolve() for x in validation_root]
    if not source_roots:
        raise ValueError("At least one new source-root is required")
    all_roots = source_roots + validation_roots
    if len({str(x).casefold() for x in all_roots}) != len(all_roots):
        raise ValueError("Prepared roots must be unique across source and validation arguments")
    output = project / "data" / name
    config_path = project / "configs/local" / f"{name}.json"
    run_path = project / "runs" / name
    if output.exists() or config_path.exists() or run_path.exists():
        raise FileExistsError("Stage name must be new; existing data, configurations and runs are immutable")
    base_config_path = project / "configs/local/bdd2018_baseline_v1.json"
    base_config = load_config(base_config_path)
    if (base_config.get("task") != "segmentation" or base_config.get("model") != {"num_classes": 2, "width": 16}
            or base_config.get("data", {}).get("image_size") != [192, 320]
            or base_config["data"].get("label_space_id") != LABEL_SPACE):
        raise ValueError("Expected the native width16, 192x320, road_surface_v1 BDD baseline")
    bdd_root = project / "data/public_v2"
    bdd_report = _complete_report(bdd_root)
    rows, bdd_train_entry = _checked_manifest(bdd_root, bdd_report, "bdd_train", base_config["data"]["train_manifest"], allow_legacy_bdd_origin=True)
    val_specs = base_config["data"].get("val_manifests", [])
    if len(val_specs) != 1 or not isinstance(val_specs[0], dict):
        raise ValueError("Original BDD baseline must have exactly one validation manifest")
    validation, bdd_val_entry = _checked_manifest(bdd_root, bdd_report, "bdd_val", val_specs[0]["manifest"], allow_legacy_bdd_origin=True)
    if len(rows) != BDD_TRAIN_COUNT or len(validation) != BDD_VAL_COUNT:
        raise ValueError("The full original BDD 70,000/10,000 split is required")
    if {row["dataset_id"] for row in rows + validation} != {BDD_DATASET_ID}:
        raise ValueError("Original BDD manifest contains a different dataset")
    initialization = _initial_checkpoint(checkpoint_path, base_config)
    sources = [{"root": str(bdd_root), "role": "full_bdd_training_and_validation",
                "report_sha256": _sha256(bdd_root / "report.json"),
                "manifests": {"train": bdd_train_entry, "val": bdd_val_entry}}]
    new_val_specs = [{"name": "bdd", "manifest": bdd_val_entry["manifest"]}]
    reserved_test = []
    for index, root in enumerate(all_roots):
        include_train = index < len(source_roots)
        parts, source_entry = _read_prepared_source(root, include_train)
        if include_train:
            rows.extend(parts["train"])
        validation.extend(parts["val"])
        reserved_test.extend(parts.get("test", []))
        new_val_specs.append({"name": f"source_{index:02d}_{root.name}", "manifest": source_entry["manifests"]["val"]["manifest"]})
        sources.append(source_entry)
    # Same original drive is permitted within validation (e.g. real KITTI +
    # synthetic VKITTI); it must never occur in train or reserved test as well.
    validate_splits(rows, validation + reserved_test)
    validate_splits(validation, reserved_test)
    # Use one manifest so the native loader shuffles without replacement. Do
    # not inherit sampling caps, diagnostic flags or extra options from a run.
    manifest_path = output / "train.jsonl"
    config = {
        "task": "segmentation", "model": {"num_classes": 2, "width": 16},
        "data": {"label_space_id": LABEL_SPACE, "image_size": [192, 320],
                 "train_manifest": str(manifest_path), "val_manifests": new_val_specs},
        "training": {"output_dir": str(run_path), "epochs": epochs, "device": "cuda", "batch_size": 16,
                     "grad_accum_steps": 1, "num_workers": 2, "lr": 0.0002, "seed": 42,
                     "amp": True, "amp_init_scale": 1024.0, "log_every_steps": 50,
                     "init_from": str(checkpoint_path),
                     "segmentation_class_weights": list(segmentation_class_weights)},
    }
    validate_config(config)
    output.mkdir(parents=True, exist_ok=False)
    marker = output / ".incomplete"
    marker.write_text("Stage preparation incomplete. Do not train this stage.\n", encoding="utf-8")
    report = {
        "complete": False, "stage": name, "label_space_id": LABEL_SPACE,
        "train_counts": dict(Counter(row["dataset_id"] for row in rows)),
        "validation_counts": dict(Counter(row["dataset_id"] for row in validation)),
        "reserved_test_counts": dict(Counter(row["dataset_id"] for row in reserved_test)),
        "source_manifests": sources, "base_config": str(base_config_path), "base_config_sha256": _sha256(base_config_path),
        "initial_checkpoint": initialization, "epochs": epochs,
        "full_bdd_training_samples": BDD_TRAIN_COUNT, "total_training_samples": len(rows),
        "legacy_bdd_origin_normalization": "Rows without origin_id receive bdd100k:<publisher group_id> in the stage copy; original manifests are untouched.",
        "all_stage_images_once_per_epoch": True, "sampling_with_replacement": False,
        "diagnostic": False, "augmentation": "none", "segmentation_class_weights": list(segmentation_class_weights),
        "validation_loss_and_metrics": "Unweighted; class weights apply only to the training loss.",
        "split_leakage_checked_against_validation_and_reserved_test": True,
        "validation_vs_reserved_test_leakage_checked": True,
        "test_used_for_training_or_validation": False, "config": str(config_path),
        "training_started": False, "deployed": False,
    }
    try:
        with manifest_path.open("x", encoding="utf-8") as stream:
            for row in sorted(rows, key=lambda item: (item["dataset_id"], item["image"], item["target"])):
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        report["train_manifest_sha256"] = _sha256(manifest_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        with config_path.open("x", encoding="utf-8") as stream:
            stream.write(json.dumps(config, indent=2, ensure_ascii=False) + "\n")
        report["config_sha256"] = _sha256(config_path)
        report["complete"] = True
        _write_json(output / "report.json", report)
        marker.unlink()
        return report
    except BaseException as error:
        report.update(complete=False, error=f"{type(error).__name__}: {error}")
        _write_json(output / "report.json", report)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-root", required=True, action="append", type=Path)
    parser.add_argument("--validation-root", action="append", default=[], type=Path)
    parser.add_argument("--init-from", required=True, type=Path)
    parser.add_argument("--epochs", default=1, type=int)
    parser.add_argument("--segmentation-class-weights", nargs=2, default=[1.0, 2.0], type=float, metavar=("ROAD", "NONROAD"))
    report = prepare_large_data_stage(**vars(parser.parse_args()))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
