"""Precommit ONE corrected-label candidate; optionally export data using CPU only.

This script never runs training, inference, a GPU probe, a download or a server.
The default creates a plan after checking immutable metadata and the exact
33,228/4,636 partition. --prepare-data additionally calls the existing CPU
exporter (which hashes/decodes every image and mask) and freezes the dev pixels.
Use a NEW output directory; interrupted or failed outputs are preserved.

The development images were already seen by parent models. They are suitable
for subsequent candidate selection, NOT an independent generalization test.
The ordinary frozen holdout evaluator intentionally rejects this exposure;
commands use the separate evaluate_selection_development.py protocol instead.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys


EXPECTED_COUNTS = {"train": 33228, "dev": 4636}
EXPECTED_SOURCES = {
    "train": {"aihub_13": 5551, "aihub_14": 5736, "aihub_15": 5669,
              "bdd100k_drivable": 9000, "vkitti2_0_3": 7272},
    "dev": {"aihub_13": 727, "aihub_14": 615, "aihub_15": 634,
            "bdd100k_drivable": 1000, "vkitti2_0_3": 1660},
}
EXPOSURE = "parent_exposed_selection_only"
FORMAT = "aihub_training_preprocessing_revision_v1"


def identity(path):
    path = Path(path).expanduser().resolve(strict=True)
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError(f"Input changed while hashing: {path}")
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": after.st_size}


def _json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Require lowercase SHA256 hex")
    return value


def _expect(path, digest):
    actual = identity(path)
    if actual["sha256"] != _sha(digest):
        raise ValueError(f"SHA256 mismatch: {path}")
    return actual


def _reported(entry):
    if not isinstance(entry, dict) or not entry.get("path"):
        raise ValueError("Missing recorded input identity")
    result = _expect(entry["path"], entry.get("sha256"))
    if "bytes" in entry and entry["bytes"] != result["bytes"]:
        raise ValueError("Recorded byte count mismatch")
    return result


def _rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        result = [json.loads(line) for line in stream if line.strip()]
    if not result or any(not isinstance(row, dict) for row in result):
        raise ValueError("Manifest must contain nonempty object rows")
    return result


def _tokens(row):
    result = set()
    for key in ("image", "source_image", "input_image"):
        if row.get(key):
            result.add(("path", str(Path(row[key]).resolve()).casefold()))
    for key in ("image_sha256", "image_rgb_sha256"):
        if row.get(key):
            result.add((key, _sha(row[key])))
    if row.get("origin_id"):
        result.add(("origin", row["origin_id"]))
    if row.get("group_id"):
        result.add(("group", row.get("dataset_id"), row["group_id"]))
    if row.get("development_group_id"):
        result.add(("development_group", row["development_group_id"]))
    return result


def _membership(row):
    # Preserve multiplicity and original identity; corrected masks may differ.
    return tuple(row.get(key) for key in (
        "dataset_id", "image_sha256", "image_rgb_sha256", "origin_id", "group_id", "sample_id"))


def _protected_tokens(contract):
    result = set()
    groups = [*contract.get("sources", []), *contract.get("public_tracks", {}).values()]
    if not contract.get("complete") or not groups:
        raise ValueError("Official contract is incomplete")
    for group in groups:
        for record in group.get("records", []):
            result.update(_tokens(record["row"]))
            result.add(("image_sha256", _sha(record["image"]["sha256"])))
            result.add(("image_rgb_sha256", _sha(record["image_rgb_sha256"])))
    return result


def validate_revision(report_path, expected_report_sha256):
    """Validate fixed membership/provenance, not re-decode 37,864 source images.

    The exporter performs full content validation before a usable dataset is
    published. No CLI option changes the reviewed split size or source counts.
    """
    report_path = Path(report_path).resolve(strict=True)
    evidence = {"report": _expect(report_path, expected_report_sha256)}
    report = _json(report_path)
    if report.get("complete") is not True or report.get("format") != FORMAT:
        raise ValueError("Require completed corrected-label preprocessing report")
    marker = _json(report_path.parent / ".complete")
    if marker.get("report_sha256") != evidence["report"]["sha256"]:
        raise ValueError("Preprocessing completion marker mismatch")
    evidence["marker"] = identity(report_path.parent / ".complete")
    for name, field in (("policy", "policy_sha256"), ("mapping", "mapping_sha256"),
                        ("changes", "changes_sha256"), ("split_report", "split_report_sha256")):
        evidence[name] = _expect(report_path.parent / f"{name}.json", report.get(field))
    split_report = _json(report_path.parent / "split_report.json")
    if (split_report.get("complete") is not True or
            split_report.get("counts") != {"input": sum(EXPECTED_COUNTS.values()), **EXPECTED_COUNTS} or
            report.get("pool_count") != sum(EXPECTED_COUNTS.values())):
        raise ValueError("Fixed train/dev partition count mismatch")
    required_checks = (
        "all_referenced_training_images_byte_verified",
        "all_referenced_training_masks_byte_verified_and_decoded",
        "only_ignore255_to_nonroad1_changed",
        "official_reporting_manifests_contract_reports_unchanged",
    )
    if any(report.get("validation", {}).get(key) is not True for key in required_checks):
        raise ValueError("Required preprocessing validation missing")
    if report.get("validation", {}).get("reporting_cross_split_byte_rgb_group_overlap") != 0:
        raise ValueError("Preprocessing reports protected-set overlap")

    inputs = report.get("inputs", {})
    for key in ("export_report", "export_marker", "train_manifest", "reporting_manifest", "official_contract"):
        evidence[key] = _reported(inputs.get(key))
    old_export = _json(evidence["export_report"]["path"])
    old_marker = _json(evidence["export_marker"]["path"])
    if old_export.get("complete") is not True or old_marker.get("report_sha256") != evidence["export_report"]["sha256"]:
        raise ValueError("Original training export is not complete")
    original = _rows(evidence["train_manifest"]["path"])
    reporting = _rows(evidence["reporting_manifest"]["path"])
    protected = _protected_tokens(_json(evidence["official_contract"]["path"]))
    for row in reporting:
        protected.update(_tokens(row))

    partitions, all_tokens = {}, {}
    for split, expected in EXPECTED_COUNTS.items():
        entry = report.get("manifests", {}).get(split)
        evidence[split] = _reported(entry)
        if Path(evidence[split]["path"]) != report_path.parent / f"{split}.jsonl":
            raise ValueError("Revision manifests must belong to the reviewed preparation folder")
        rows = _rows(evidence[split]["path"])
        if entry.get("samples") != expected or len(rows) != expected:
            raise ValueError(f"Fixed {split} count mismatch")
        if dict(Counter(row.get("dataset_id") for row in rows)) != EXPECTED_SOURCES[split]:
            raise ValueError(f"Fixed {split} source counts mismatch; public/official data forbidden")
        tokens, paths = set(), set()
        for row in rows:
            proof = row.get("preprocessing_provenance", {})
            if (row.get("task") != "segmentation" or row.get("label_space_id") != "road_surface_v1"
                    or row.get("split") != split or row.get("export_split") != split
                    or proof.get("previous_export_split") != "train"
                    or proof.get("development_role") != (EXPOSURE if split == "dev" else "training")):
                raise ValueError("Only the reviewed training-derived partition is allowed")
            if not row.get("development_group_id") or not row.get("group_id"):
                raise ValueError("Missing development grouping identity")
            for key in ("image_sha256", "image_rgb_sha256", "target_sha256"):
                _sha(row.get(key))
            for key in ("image", "target"):
                if not isinstance(row.get(key), str) or not Path(row[key]).is_file():
                    raise ValueError(f"Missing {key} file")
            image_path = str(Path(row["image"]).resolve()).casefold()
            if image_path in paths:
                raise ValueError("Duplicate image path inside partition")
            paths.add(image_path)
            row_tokens = _tokens(row)
            if row_tokens & protected:
                raise ValueError("Official/public reporting image or group entered selection data")
            tokens.update(row_tokens)
        partitions[split], all_tokens[split] = rows, tokens
    if all_tokens["train"] & all_tokens["dev"]:
        raise ValueError("Train/dev path, bytes, RGB, origin or group overlap")
    actual_membership = Counter(_membership(row) for rows in partitions.values() for row in rows)
    if actual_membership != Counter(_membership(row) for row in original):
        raise ValueError("Corrected partition does not exactly preserve the original training pool")
    # Recheck metadata after reading. Per-image content is rehashed by export.
    for item in evidence.values():
        if identity(item["path"]) != item:
            raise ValueError("Preparation evidence changed during validation")
    return {"report": report, "evidence": evidence, "partitions": partitions}


def _write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def _verify_initializer(report_path):
    # This provenance function uses file hashes only; no torch import/unpickle.
    from train_yolo_semantic import verify_initialization
    report = _json(report_path)
    best = report.get("checkpoints", {}).get("best", {}).get("path")
    if not best:
        raise ValueError("Initializer must identify its best checkpoint")
    checkpoint, lineage = verify_initialization(best, report_path)
    return report, checkpoint, lineage


def _materialize(config, spec, output, workers):
    from prepare_yolo_semantic import prepare as export
    from evaluation_contract import prepare as freeze
    from train_yolo_semantic import verify_export
    export(config, output / "export", workers=workers)
    dataset, _ = verify_export(output / "export", config)
    if dataset["counts"] != {"train": EXPECTED_COUNTS["train"], "val": EXPECTED_COUNTS["dev"]}:
        raise ValueError("Export did not preserve the exact reviewed partition")
    freeze(spec, output / "dev_contract")
    return {"export_report": identity(output / "export/report.json"),
            "dev_contract": identity(output / "dev_contract/contract.json")}


def prepare(report_path, expected_report_sha256, initial_training_report, output, *, prepare_data=False, workers=2):
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new experiment directory: {output}")
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    checked = validate_revision(report_path, expected_report_sha256)
    initial_training_report = Path(initial_training_report).resolve(strict=True)
    initial_report, checkpoint, lineage = _verify_initializer(initial_training_report)
    if initial_report.get("dataset", {}).get("provenance_sha256") != checked["evidence"]["export_report"]["sha256"]:
        raise ValueError("Initializer must be the completed AIHub parent trained on the reviewed original export")
    output.mkdir(parents=True, exist_ok=False)
    incomplete = output / ".incomplete"
    incomplete.write_text("CPU preparation in progress; do not train.\n", encoding="utf-8")
    try:
        evidence = checked["evidence"]
        config_path = output / "config.json"
        config = {"task": "segmentation", "model": {"num_classes": 2, "width": 16},
                  "data": {"label_space_id": "road_surface_v1", "image_size": [192, 320],
                           "train_manifest": evidence["train"]["path"], "val_manifest": evidence["dev"]["path"]},
                  "training": {"output_dir": str(output / "native_training_not_used"), "epochs": 3,
                               "device": "cuda", "batch_size": 8, "num_workers": 2, "lr": .0001,
                               "seed": 42, "amp": True, "segmentation_class_weights": [1., 2.]},
                  "development_role": EXPOSURE,
                  "preprocessing_report": evidence["report"]}
        _write_new(config_path, config)
        sources = []
        for name in sorted(EXPECTED_SOURCES["dev"]):
            path = output / "dev_manifests" / f"{name}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("x", encoding="utf-8", newline="\n") as stream:
                for row in checked["partitions"]["dev"]:
                    if row["dataset_id"] == name:
                        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            sources.append({"name": name, "manifest": str(path)})
        spec_path = output / "dev_contract_spec.json"
        _write_new(spec_path, {"schema_version": 1, "sources": sources, "runtime_threshold": .65,
                              "retain_source_duplicate_multiplicity": False,
                              "development_role": EXPOSURE, "independent_holdout": False})
        script_root = Path(__file__).resolve().parent
        common_eval = ["--prepared", str(Path(report_path).resolve().parent), "--input-size", "384", "640",
                       "--batch-size", "4", "--device", "cuda", "--gpu-memory-gib", "5"]
        evaluator = str(script_root / "evaluate_selection_development.py")
        commands = {
            "1_baseline_development": [sys.executable, evaluator, *common_eval, "--role", "baseline",
                "--training-report", str(initial_training_report), "--checkpoint", checkpoint["path"],
                "--output", str(output / "baseline_dev.json")],
            "2_single_candidate_training": [sys.executable, "scripts/train_yolo_semantic.py",
                "--dataset", str(output / "export"), "--config", str(config_path),
                "--weights", checkpoint["path"], "--init-training-report", str(initial_training_report),
                "--output", str(output / "training"), "--epochs", "3", "--image-size", "384",
                "--batch", "8", "--nominal-batch", "32", "--workers", "2", "--lr", "0.0001",
                "--gpu-memory-gib", "5", "--device", "0"],
            "3_candidate_development": [sys.executable, evaluator, *common_eval, "--role", "candidate",
                "--training-report", str(output / "training/training_report.json"),
                "--checkpoint", str(output / "training/fit/weights/best.pt"),
                "--output", str(output / "candidate_dev.json")],
        }
        hypothesis = {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "experiment": "corrected_label_single_candidate_v1", "precommitted_before_training": True,
            "hypothesis": "Correcting documented ignore-to-nonroad omissions, while replaying the fixed training cohort, may reduce nonroad-as-road errors without excessive road recall/IoU regression. Improvement is not assumed.",
            "max_new_training_runs": 1, "epochs": 3, "automatic_retry_or_expansion": False,
            "fixed_counts": EXPECTED_COUNTS, "fixed_sources": EXPECTED_SOURCES,
            "preparation_evidence": evidence, "source_config": identity(config_path),
            "dev_contract_spec": identity(spec_path),
            "dev_manifest_identities": [identity(item["manifest"]) for item in sources],
            "initializer": lineage, "initializer_training_report": identity(initial_training_report),
            "model_selection": "best.pt chosen by trainer's dev mIoU fitness among exactly 3 epochs; then evaluated against precommitted per-source eligibility criteria. Do not switch to last.pt or retune threshold after seeing results.",
            "loss": {"objective": "CE + Dice + auxiliary CE", "ce_class_weights": [1., 2.], "ignore_id": 255},
            "training_settings": {"batch": 8, "nominal_batch": 32, "workers": 2, "amp": True,
                                  "image_size": 384, "lr": .0001, "gpu_memory_gib": 5., "seed": 42},
            "evaluation": {"role": EXPOSURE, "input_size": [384, 640], "native_gt_size": [192, 320],
                           "runtime_threshold": .65, "images": EXPECTED_COUNTS["dev"],
                           "all_sources_required": sorted(EXPECTED_SOURCES["dev"]),
                           "standard_frozen_holdout_evaluator_compatible": False,
                           "evaluator": "evaluate_selection_development.py",
                           "official_contract_untouched": evidence["official_contract"],
                           "official_and_public_used_for_selection": False},
            "acceptance": {"unit": "fraction, not percent", "comparison": "candidate minus baseline, identical prepared dev/input/precision/threshold/denominators",
                           "scope": "every source individually; missing/undefined metrics fail eligibility",
                           "maximum_nonroad_as_road_fpr_increase": 0.0,
                           "maximum_road_recall_decrease": .01, "maximum_road_iou_decrease": .01,
                           "requires_at_least_one_source_nonroad_fpr_strict_decrease": True,
                           "no_automatic_deployment": True},
            "execution_prerequisites": ["CPU export and dev contract complete with matching hashes",
                "Baseline dev evaluation completes successfully BEFORE any candidate optimizer update",
                "One fixed candidate only; do not terminate other applications or restart automations",
                "A failure/OOM preserves logs and checkpoints; no automatic restart or batch change"],
            "limitations": ["Development images were exposed to parent training; this is selection, not blind generalization evidence.",
                "Candidate also selects best.pt on this dev set; post-selection results remain optimistic.",
                "The intervention includes extra optimization and revised train/dev membership; it is not an isolated causal test of label correction.",
                "Nonroad FP covers annotated nonroad pixels; it is not an obstacle-only or collision probability.",
                "Night rare-object coverage, partial ignore labels and source-task semantics remain limitations.",
                "Allocator cap excludes CUDA context and other apps; nominal batch is not physical batch equivalence.",
                "No real driving, 80 km/h or target MSI device performance claim is made."],
            "commands": commands, "commands_executed_by_this_helper": False,
            "execution_working_directory": str(Path(report_path).resolve().parents[2]),
            "trainer_location_note": "Run the relative training command in the original execution project containing models/pretrained/yolo26n.pt; the clean publication checkout intentionally excludes weights.",
        }
        _write_new(output / "plan/hypothesis.json", hypothesis)
        # Commit the plan before any materialization; never rewrite it afterwards.
        plan_identity = identity(output / "plan/hypothesis.json")
        materialized = _materialize(config_path, spec_path, output, workers) if prepare_data else None
        for item in evidence.values():
            if identity(item["path"]) != item:
                raise ValueError("Preparation evidence changed during export")
        result = {"complete": True, "mode": "cpu_materialized" if prepare_data else "plan_only",
                  "ready_for_training_after_baseline": bool(prepare_data),
                  "baseline_evaluation_still_required": True, "hypothesis": plan_identity,
                  "materialized": materialized, "training_started": False, "gpu_used": False,
                  "download_started": False, "source_files_modified": False}
        _write_new(output / "preparation.json", result)
        _write_new(output / ".complete", {"preparation_sha256": identity(output / "preparation.json")["sha256"],
                                           "hypothesis_sha256": plan_identity["sha256"]})
        incomplete.unlink()
        return result
    except BaseException as exc:
        _write_new(output / "failure.json", {"error": f"{type(exc).__name__}: {exc}",
                                             "training_started": False, "gpu_used": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preprocessing-report", type=Path, required=True)
    parser.add_argument("--expected-report-sha256", required=True)
    parser.add_argument("--initial-training-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare-data", action="store_true", help="CPU-only export and pixel contract; never starts the generated GPU commands")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    result = prepare(args.preprocessing_report, args.expected_report_sha256, args.initial_training_report,
                     args.output, prepare_data=args.prepare_data, workers=args.workers)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
