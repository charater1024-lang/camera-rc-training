"""Evaluate TinyUNet or YOLO on one independently frozen cohort, without training."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from camera_rc.config import load_config
from camera_rc.data import load_data_sources
from camera_rc.models import build_model
from camera_rc.training import _check_compatible, _device, _read_checkpoint
import evaluation_contract as cohort
import evaluate_yolo_semantic as yolo
from evaluate_road_safety import INPUT_SIZE, LABEL_SPACE, NOTICE

OFFICIAL_SHA256 = "eae8ff0d95bf96863c59c40bade61037870c7a39f2c53a8c49fd1b5022ed331d"


def training_sources(config, ledger):
    data = config["data"]
    for item in [*data.get("train_sources", [{"manifest": data.get("train_manifest")}]),
                 *data.get("val_manifests", [data.get("val_manifest")])]:
        ledger.identity(item if isinstance(item, str) else item["manifest"])
    return load_data_sources(config)


def yolo_provenance(config_path, checkpoint, report_path, ledger, seen=None):
    seen = set() if seen is None else seen
    path = Path(report_path).resolve(strict=True)
    if str(path).casefold() in seen or len(seen) >= 16:
        raise ValueError("Cyclic or excessive local training ancestry")
    seen.add(str(path).casefold())
    completion = yolo.validate_training_report(path, checkpoint, config_path, ledger)
    config = load_config(config_path)
    training, validation = training_sources(config, ledger)
    snapshot = yolo.validate_export_snapshot(completion, training, validation, ledger)
    proofs = [{"report": completion["identity"], "config": completion["config"],
               "checkpoint": completion["checkpoint"], "export_manifests": snapshot["manifests"]}]
    initializer = completion["report"].get("initialization")
    if initializer is not None and (not isinstance(initializer, dict) or initializer.get("kind") not in
                                     ("official_pretrained", "completed_best_checkpoint")):
        raise ValueError("Invalid local initializer lineage")
    parent = (initializer or {}).get("training_report")
    if initializer and (initializer["kind"] == "completed_best_checkpoint") != bool(parent):
        raise ValueError("Local initializer requires its ancestor training report")
    parent_weights = completion["report"].get("pretrained", {})
    if not isinstance(parent_weights, dict) or not parent_weights.get("path"):
        raise ValueError("Missing initializer provenance")
    ledger.expect(parent_weights["path"], parent_weights.get("sha256"))
    ledger.identity(parent_weights["path"])
    if parent:
        ledger.expect(parent["path"], parent["sha256"])
        parent_report, _ = yolo._json(parent["path"], ledger)
        _, ancestors, ancestor_snapshot, ancestor_proofs, _ = yolo_provenance(
            parent_report["source_config"]["path"], parent_weights["path"], parent["path"], ledger, seen)
        training.extend(ancestors)
        snapshot["training_image_sha256"].update(ancestor_snapshot["training_image_sha256"])
        snapshot["training_rgb_sha256"].update(ancestor_snapshot["training_rgb_sha256"])
        proofs.extend(ancestor_proofs)
    elif parent_weights["sha256"] != OFFICIAL_SHA256:
        raise ValueError("Unreported local training ancestor; initializer is not the pinned official model")
    return config, training, snapshot, proofs, completion


def current_training_snapshot(training, ledger):
    """Fallback for historical native checkpoints whose training code did not hash inputs."""
    hashes, pixels = set(), set()
    seen = set()
    for source in training:
        for index, row in enumerate(source["rows"], 1):
            for key in ("image", "source_image"):
                if not row.get(key) or row[key].casefold() in seen:
                    continue
                seen.add(row[key].casefold())
                payload, identity = ledger.read(row[key])
                with Image.open(io.BytesIO(payload)) as image:
                    rgb = np.asarray(image.convert("RGB")).copy()
                hashes.add(identity["sha256"])
                pixels.add(cohort.rgb_sha256(rgb))
            if index % 1000 == 0:
                print(json.dumps({"event": "native_training_content_audit", "source": source["name"], "images": index}), flush=True)
    return {"training_image_sha256": hashes, "training_rgb_sha256": pixels,
            "historical_content_binding": False}


def native_provenance(config_path, checkpoint, summary_path, ledger, content_export=None):
    config_identity, checkpoint_identity = ledger.identity(config_path), ledger.identity(checkpoint)
    config = load_config(config_path)
    stored = _read_checkpoint(checkpoint)
    _check_compatible(config, stored["config"])
    # Equality of the entire data section binds the actual model's train sources,
    # rather than accepting an arbitrary same-architecture config.
    if config["data"] != stored["config"]["data"] or config["model"] != stored["config"]["model"]:
        raise ValueError("Native checkpoint training data/model differs from own source config")
    if (stored.get("diagnostic") is not False or stored.get("epoch_complete") is not True
            or type(stored.get("optimizer_steps")) is not int or stored["optimizer_steps"] < 1):
        raise ValueError("Native checkpoint is diagnostic, incomplete or untrained")
    summary, summary_identity = yolo._json(summary_path, ledger)
    if (summary.get("training_complete") is not True or summary.get("diagnostic") is not False
            or summary.get("completed_epochs") != summary.get("requested_epochs")
            or str(Path(checkpoint).resolve()) not in [summary.get("best_checkpoint"), summary.get("last_checkpoint")]):
        raise ValueError("Native own training summary does not identify a completed run/checkpoint")
    training, validation = training_sources(config, ledger)
    proofs = [{"report": summary_identity, "config": config_identity, "checkpoint": checkpoint_identity}]
    if content_export:
        provenance, identity = yolo._json(content_export, ledger)
        snapshot = yolo.validate_export_snapshot({"dataset_provenance": provenance,
            "dataset_provenance_identity": identity, "config": config_identity}, training, validation, ledger)
        proofs[0]["independent_content_export"] = identity
        proofs[0]["historical_content_binding"] = False
    else:
        snapshot = current_training_snapshot(training, ledger)
    # Local weight initializers can have learned data outside the final run.
    # Walk their embedded configs; skip hashing only when every ancestor image
    # and scene already appears in the checked descendant sources.
    seen, ancestor = {str(Path(checkpoint).resolve()).casefold()}, stored.get("initialization")
    covered = {str(Path(r["image"]).resolve()).casefold() for s in training for r in s["rows"]}
    while ancestor:
        if len(seen) >= 16 or ancestor["path"].casefold() in seen:
            raise ValueError("Cyclic or excessive native initializer ancestry")
        seen.add(ancestor["path"].casefold())
        ledger.expect(ancestor["path"], ancestor["sha256"])
        ancestor_identity = ledger.identity(ancestor["path"])
        previous = _read_checkpoint(ancestor["path"])
        prior_training, _ = training_sources(previous["config"], ledger)
        missing = [{**s, "rows": [r for r in s["rows"] if str(Path(r["image"]).resolve()).casefold() not in covered]}
                   for s in prior_training]
        missing = [s for s in missing if s["rows"]]
        if missing:
            audit = current_training_snapshot(missing, ledger)
            snapshot["training_image_sha256"].update(audit["training_image_sha256"])
            snapshot["training_rgb_sha256"].update(audit["training_rgb_sha256"])
        training.extend(prior_training)
        covered.update(str(Path(r["image"]).resolve()).casefold() for s in prior_training for r in s["rows"])
        proofs.append({"initializer_checkpoint": ancestor_identity, "embedded_training_config_checked": True})
        ancestor = previous.get("initialization")
    return config, training, snapshot, proofs, stored


class TinyPredictor:
    def __init__(self, config, stored, *, input_size, device, amp):
        self.input_size = yolo.validate_input_size(input_size)
        self.device = _device(device)
        self.amp = amp and self.device.type == "cuda"
        self.model = build_model(config).float().eval()
        self.model.load_state_dict(stored["model"], strict=True)
        if any(not torch.isfinite(v).all() for v in self.model.state_dict().values()):
            raise ValueError("Nonfinite native weights")
        self.model.to(self.device)

    @torch.inference_mode()
    def predict_logits(self, images):
        tensors = torch.stack([yolo.image_tensor(rgb, self.input_size) for rgb in images]).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            logits = self.model(tensors)
        return yolo.logits_at_input(logits, self.input_size, len(images))

    @torch.inference_mode()
    def __call__(self, rgb):
        return yolo.restore_road_probability(self.predict_logits([rgb]), self.input_size, tuple(rgb.shape[:2]))


def evaluate(evaluation_contract, model_family, config_path, checkpoint, training_report, output, *,
             input_size=INPUT_SIZE, device="cuda", batch_size=4, amp=True, training_content_export=None):
    if model_family not in ("tinyunet", "yolo26x-sem"):
        raise ValueError("Unsupported model family")
    input_size = yolo.validate_input_size(input_size)
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be in 1..64")
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new evaluation report: {output}")
    ledger = yolo.IdentityLedger()
    contract, identity, validation, public = cohort.load(evaluation_contract, ledger)
    if model_family == "yolo26x-sem":
        if training_content_export:
            raise ValueError("YOLO uses only its own recorded export provenance")
        config, training, snapshot, proofs, state = yolo_provenance(config_path, checkpoint, training_report, ledger)
    else:
        config, training, snapshot, proofs, state = native_provenance(
            config_path, checkpoint, training_report, ledger, training_content_export)
    if (config.get("task") != "segmentation" or config.get("model", {}).get("num_classes") != 2
            or config.get("data", {}).get("label_space_id") != LABEL_SPACE):
        raise ValueError("Model requires canonical two-class road semantics")
    # Ancestral sources may repeat; deduplicate same images before strict split checks.
    unique = {}
    for source in training:
        for row in source["rows"]:
            unique.setdefault(str(Path(row["image"]).resolve()).casefold(), row)
    cohort.assert_no_leakage(contract, [{"rows": list(unique.values())}], snapshot)
    predictor = (yolo.YoloSemanticPredictor(checkpoint, input_size=input_size, device=device, amp=amp)
                 if model_family == "yolo26x-sem" else TinyPredictor(config, state, input_size=input_size, device=device, amp=amp))
    threshold = contract["runtime_threshold"]
    native = {source["name"]: yolo.evaluate_native_source(source, predictor, ledger, batch_size, threshold) for source in validation}
    for value in native.values():
        value["runtime_threshold_metrics"]["comparison_limit"] = "Paired only against the same frozen evaluation contract, input size and precision condition."
    tracks = {name: yolo.evaluate_public_track(value, predictor, name, ledger, threshold, snapshot) for name, value in public.items()}
    denominators = {source["name"]: {str(label): sum(r["evaluated_gt_pixels"][str(label)] for r in source["records"])
                    for label in (0, 1, 255)} for source in contract["sources"]}
    for name, counts in denominators.items():
        if [sum(r) for r in native[name]["confusion_matrix"]] != [counts["0"], counts["1"]]:
            raise ValueError("Evaluation native pixel denominator differs from frozen contract")
    report = {"format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "notice": NOTICE,
              "evaluation_complete": True, "evaluation_only": True, "training_performed": False,
              "deployment_validated": False, "models_validated_for_control": False, "real_performance_validated": False,
              "evaluation_contract": identity, "protocol_id": cohort.PROTOCOL,
              "checkpoint": ledger.identity(checkpoint), "config": ledger.identity(config_path), "checkpoint_diagnostic": False,
              "training_completion": proofs, "training_provenance_preserved": True,
              "model_family": model_family, "task": "semantic" if model_family == "yolo26x-sem" else "segmentation",
              "class_names": yolo.CLASS_NAMES, "road_class_id": 0, "label_space_id": LABEL_SPACE,
              "input_size": list(input_size), "native_gt_size": list(INPUT_SIZE), "runtime_threshold": threshold,
              "precision_condition": "autocast_fp16_logits_fp32_metrics" if predictor.amp else "fp32",
              "device": str(predictor.device), "amp_effective": predictor.amp, "batch_size": batch_size,
              "validation_sources": [{"name": s["name"], "images": s["images"], "manifest": s["manifest"]} for s in contract["sources"]],
              "gt_denominators": denominators, "native_sources": native, "public_obstacle_anomaly_subsets": tracks,
              "retained_within_source_duplicates": contract["retained_within_source_duplicates"],
              "public_role": cohort.PUBLIC_ROLE, "local_training_ancestry_checked": True,
              "small_obstacle_instance_detection_evaluated": False,
              "small_obstacle_limit": "Binary semantic masks do not supply object-instance identities or physical distance/size; tiny original GT pixels remain in the denominator, but instance miss rate is not inferred.",
              "leakage_limit": "Checks known local training paths, scene/origin IDs, byte and RGB hashes, including local initializers. Unknown external pretraining cannot be ruled out. Native historical image bytes were not pinned at training time; independent export/current content checks cannot prove those past bytes.",
              "protocol": {"rgb": "PIL RGB bilinear direct resize; float32 [0,1]",
                           "native_gt": "nearest 192x320; ignore255 excluded; FP32 logits argmax and softmax road>=threshold",
                           "public_gt": "original resolution GT1 only; GT0 usual is not assumed road",
                           "public_probability": "FP32 softmax after input-size logits resize then bilinear probability to original GT"}}
    ledger.verify_unchanged()
    # The completion marker itself is in the ledger and is checked again above.
    report["input_content_identities"] = sorted(ledger.identities.values(), key=lambda item: item["path"].casefold())
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("evaluation-contract", "config", "checkpoint", "training-report", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--model-family", choices=("tinyunet", "yolo26x-sem"), required=True)
    parser.add_argument("--training-content-export", type=Path)
    parser.add_argument("--input-size", nargs=2, type=int, default=list(INPUT_SIZE))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    evaluate(args.evaluation_contract, args.model_family, args.config, args.checkpoint, args.training_report, args.output,
             input_size=args.input_size, device=args.device, batch_size=args.batch_size, amp=not args.no_amp,
             training_content_export=args.training_content_export)


if __name__ == "__main__":
    main()
