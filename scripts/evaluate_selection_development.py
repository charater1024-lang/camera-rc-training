"""Evaluate a parent-exposed selection set; NEVER an independent test score.

This deliberately separate protocol leaves the official evaluator's ancestor
leakage checks unchanged. Only the baseline role permits direct training
exposure. Candidate training must contain exactly the revised training split.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path

import torch

import evaluate_frozen_road as frozen
import evaluate_yolo_semantic as yolo
from verify_aihub_preprocessing import verify_revision, row_keys, disjoint

PROTOCOL = "parent_exposed_selection_only_v1"


def check_model_partition(role, direct_training_hashes, train, dev):
    """Check direct training only; ancestor exposure is separately disclosed."""
    if role not in ("baseline", "candidate"):
        raise ValueError("Explicit baseline or candidate role required")
    train_hashes = {r["image_sha256"] for r in train}
    dev_hashes = {r["image_sha256"] for r in dev}
    if train_hashes & dev_hashes:
        raise ValueError("Prepared train/dev byte overlap")
    if role == "candidate":
        if set(direct_training_hashes) != train_hashes:
            raise ValueError("Candidate must train on exactly the revised train split")
    elif not ((train_hashes | dev_hashes) <= set(direct_training_hashes)):
        raise ValueError("Baseline must cover the parent-exposed pool")
    return len(set(direct_training_hashes) & dev_hashes)


def check_candidate_export_partitions(exported, train, dev):
    """Bind image AND corrected target bytes, retaining sample multiplicity.

Image membership alone cannot distinguish old masks or an official validation
list used for checkpoint selection. Export input_target_sha256 identifies the
exact source mask before the exporter re-encodes its canonical PNG.
"""
    def pairs(rows, target_key):
        result = Counter()
        for row in rows:
            if (not isinstance(row, dict) or not isinstance(row.get("image_sha256"), str)
                    or not row["image_sha256"] or not isinstance(row.get(target_key), str)
                    or not row[target_key]):
                raise ValueError("Image and input-target identities required for candidate partition")
            result[(row["image_sha256"], row[target_key])] += 1
        return result

    for split, prepared in (("train", train), ("val", dev)):
        if split not in exported or pairs(exported[split], "input_target_sha256") != pairs(prepared, "target_sha256"):
            label = "corrected training" if split == "train" else "selection development"
            raise ValueError(f"Candidate export {split} must exactly match {label} image/target bytes and multiplicity")


def _confusion(source):
    if (not isinstance(source, dict) or type(source.get("images")) is not int or source["images"] < 1
            or not isinstance(source.get("runtime_threshold_metrics"), dict)):
        raise ValueError("Positive image count and threshold metrics required")
    matrix = source["runtime_threshold_metrics"].get("confusion_matrix")
    if (not isinstance(matrix, list) or len(matrix) != 2
            or any(not isinstance(row, list) or len(row) != 2
                   or any(type(value) is not int or value < 0 for value in row) for row in matrix)):
        raise ValueError("Require a 2x2 nonnegative integer confusion matrix")
    if any(sum(row) == 0 for row in matrix):
        raise ValueError("Finite defined development metrics require both road and nonroad GT denominators")
    return matrix


def _canonical_metrics(source, matrix, threshold):
    """Recompute ratios from counts and reject inconsistent derived JSON fields."""
    tp, fn = matrix[0]
    fp, tn = matrix[1]
    total = tp + fn + fp + tn
    metrics = {"road_iou": tp / (tp + fn + fp), "road_recall": tp / (tp + fn),
               "false_road_rate": fp / (fp + tn)}
    stored = source["runtime_threshold_metrics"]

    def check_ratio(key, actual, expected):
        if type(actual) not in (int, float) or not math.isfinite(actual) or not 0 <= actual <= 1:
            raise ValueError(f"Finite defined development metrics required: {key}")
        if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Stored {key} is inconsistent with the confusion matrix")

    for key, value in metrics.items():
        check_ratio(key, stored.get(key), value)
    # These redundant fields are emitted by the current evaluator. Validate
    # them when present, while allowing the compact older comparison schema.
    for obj in (source, stored):
        if "valid_pixels" in obj and (type(obj["valid_pixels"]) is not int or obj["valid_pixels"] != total):
            raise ValueError("Stored valid_pixels denominator differs from confusion matrix")
    for key, expected in (("road_tp_pixels", tp), ("road_fn_pixels", fn),
                          ("road_fp_pixels", fp), ("road_tn_pixels", tn)):
        if key in stored and (type(stored[key]) is not int or stored[key] != expected):
            raise ValueError(f"Stored {key} differs from confusion matrix")
    if "road_precision" in stored:
        if tp + fp:
            check_ratio("road_precision", stored["road_precision"], tp / (tp + fp))
        elif stored["road_precision"] is not None:
            raise ValueError("Road precision must be undefined with no predicted road pixels")
    if "class_iou" in stored:
        values = stored["class_iou"]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError("Two class IoU values required")
        check_ratio("class_iou[0]", values[0], metrics["road_iou"])
        check_ratio("class_iou[1]", values[1], tn / (tn + fn + fp))
    if "threshold" in stored and stored["threshold"] != threshold:
        raise ValueError("Source threshold differs from report condition")
    return metrics


def compare_reports(baseline, candidate):
    """Fixed development gate, never a deployment/safety approval."""
    for key in ("protocol_id", "prepared_report_sha256", "dev_manifest_sha256",
                "input_size", "native_gt_size", "runtime_threshold", "precision_condition"):
        if baseline.get(key) != candidate.get(key):
            raise ValueError(f"Unmatched development condition: {key}")
    if (baseline.get("protocol_id") != PROTOCOL or baseline.get("role") != "baseline"
            or candidate.get("role") != "candidate"
            or baseline.get("evaluation_complete") is not True or candidate.get("evaluation_complete") is not True):
        raise ValueError("Completed development reports with matching roles required")
    if (not isinstance(baseline.get("native_sources"), dict) or not baseline["native_sources"]
            or not isinstance(candidate.get("native_sources"), dict) or not candidate["native_sources"]):
        raise ValueError("Nonempty development sources required")
    if set(baseline["native_sources"]) != set(candidate["native_sources"]):
        raise ValueError("Source mismatch")
    rows, failures, improved = {}, [], []
    for source, old in baseline["native_sources"].items():
        new = candidate["native_sources"][source]
        old_matrix, new_matrix = _confusion(old), _confusion(new)
        if old["images"] != new["images"] or [sum(x) for x in old_matrix] != [sum(x) for x in new_matrix]:
            raise ValueError("Changed evaluation denominator")
        a = _canonical_metrics(old, old_matrix, baseline["runtime_threshold"])
        b = _canonical_metrics(new, new_matrix, candidate["runtime_threshold"])
        changes = {}
        for key in ("road_iou", "road_recall", "false_road_rate"):
            before, after = a[key], b[key]
            changes[key] = {"baseline": before, "candidate": after,
                            "difference_pp": (after - before) * 100}
        if b["false_road_rate"] > a["false_road_rate"] + 1e-12:
            failures.append(f"{source}: nonroad false-road increased")
        elif b["false_road_rate"] < a["false_road_rate"] - 1e-12:
            improved.append(source)
        for key in ("road_iou", "road_recall"):
            if b[key] < a[key] - .01 - 1e-12:
                failures.append(f"{source}: {key} decreased by more than 1 pp")
        rows[source] = changes
    if not improved:
        failures.append("No source has a strict nonroad false-road rate decrease")
    return {"protocol_id": PROTOCOL, "development_gate_passed": not failures,
            "criteria": "Each source: no nonroad FPR increase; road recall and IoU decrease <=1 pp; at least one source FPR strictly decreases (numerical tolerance 1e-12)",
            "sources_with_strict_false_road_decrease": improved,
            "derived_metrics_recomputed_from_confusion": True,
            "failures": failures, "sources": rows, "deployment_approved": False,
            "causal_label_correction_effect_established": False,
            "notice": "Parent-exposed selection data; not independent generalization or vehicle safety evidence."}


def evaluate(prepared, training_report, checkpoint, output, *, role, input_size=(384, 640),
             batch_size=4, device="cuda", gpu_memory_gib=5):
    output, prepared = Path(output).resolve(), Path(prepared).resolve()
    if output.exists():
        raise FileExistsError("Preserve existing results; select a fresh output")
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("Batch must be an integer in 1..8")
    if not math.isfinite(gpu_memory_gib) or not 0 < gpu_memory_gib <= 8:
        raise ValueError("Allocator cap must be finite in (0,8] GiB")
    audit = verify_revision(prepared)
    ledger = yolo.IdentityLedger()
    prepared_report, report_id = yolo._json(prepared / "report.json", ledger)
    rows = {}
    for split in ("train", "dev"):
        item = prepared_report["manifests"][split]
        ledger.expect(item["path"], item["sha256"])
        payload, _ = ledger.read(item["path"])
        rows[split] = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line]
    disjoint(row_keys(rows["train"]), row_keys(rows["dev"]), "Selection train/dev")
    report, _ = yolo._json(training_report, ledger)
    config_path = report["source_config"]["path"]
    completion = yolo.validate_training_report(training_report, checkpoint, config_path, ledger)
    config = frozen.load_config(config_path)
    training, validation = frozen.training_sources(config, ledger)
    direct = yolo.validate_export_snapshot(completion, training, validation, ledger)
    overlap = check_model_partition(role, direct["training_image_sha256"], rows["train"], rows["dev"])
    if role == "candidate":
        exported = {}
        for split in ("train", "val"):
            item = direct["manifests"][split]
            ledger.expect(item["path"], item["sha256"])
            payload, _ = ledger.read(item["path"])
            exported[split] = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
        check_candidate_export_partitions(exported, rows["train"], rows["dev"])
    _, _, ancestry, proofs, _ = frozen.yolo_provenance(config_path, checkpoint, training_report, ledger)
    grouped = defaultdict(list)
    for row in rows["dev"]:
        ledger.expect(row["image"], row["image_sha256"])
        ledger.expect(row["target"], row["target_sha256"])
        grouped[row["dataset_id"]].append(row)
    gpu = None
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no silent CPU fallback")
        properties = torch.cuda.get_device_properties(0)
        cap = int(gpu_memory_gib * 1024**3)
        if cap >= properties.total_memory:
            raise ValueError("Allocator cap must leave physical GPU headroom")
        torch.cuda.set_per_process_memory_fraction(cap / properties.total_memory, 0)
        gpu = {"name": properties.name, "uuid": str(getattr(properties, "uuid", "unavailable")),
               "physical_bytes": properties.total_memory, "allocator_cap_gib": gpu_memory_gib,
               "context_outside_allocator_cap": True}
    predictor = yolo.YoloSemanticPredictor(checkpoint, input_size=input_size, device=device, amp=True)
    native = {}
    for name, selected in sorted(grouped.items()):
        native[name] = yolo.evaluate_native_source({"name": name, "rows": selected}, predictor, ledger, batch_size, .65)
    ledger.verify_unchanged()
    value = {"evaluation_complete": True, "protocol_id": PROTOCOL, "role": role,
             "created_utc": datetime.now(timezone.utc).isoformat(),
             "prepared_report_sha256": report_id["sha256"],
             "dev_manifest_sha256": prepared_report["manifests"]["dev"]["sha256"],
             "input_size": list(input_size), "native_gt_size": [192, 320], "runtime_threshold": .65,
             "precision_condition": "autocast_fp16_logits_fp32_metrics" if predictor.amp else "fp32",
             "checkpoint": ledger.identity(checkpoint), "training_completion": proofs,
             "direct_training_overlap_dev_images": overlap,
             "candidate_corrected_training_and_development_input_targets_verified": role == "candidate",
             "ancestry_training_overlap_dev_images": sum(r["image_sha256"] in ancestry["training_image_sha256"] for r in rows["dev"]),
             "native_sources": native, "gpu": gpu, "batch_size": batch_size,
             "preprocessing_audit": audit, "official_or_public_images_evaluated": False,
             "independent_test": False, "deployment_approved": False,
             "notice": "Parent-exposed development data for one subsequent candidate selection. No blind-test claim.",
             "input_content_identities": list(ledger.identities.values())}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("prepared", "training-report", "checkpoint", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--role", choices=("baseline", "candidate"), required=True)
    parser.add_argument("--input-size", nargs=2, type=int, default=[384, 640])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--gpu-memory-gib", type=float, default=5)
    args = parser.parse_args()
    evaluate(args.prepared, args.training_report, args.checkpoint, args.output, role=args.role,
             input_size=args.input_size, batch_size=args.batch_size, device=args.device,
             gpu_memory_gib=args.gpu_memory_gib)


if __name__ == "__main__":
    main()
