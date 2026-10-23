"""Evaluate a completed local YOLO26x semantic road model without training it.

The common condition directly resizes RGB to 192x320, matching TinyUNet's
preprocessing. Native GT always uses nearest-neighbour 192x320 labels. Larger
model inputs are separate accuracy conditions, not a matched-resolution gain.
Public obstacle/anomaly GT keeps every original pixel, including tiny objects.
These are reused development pixel diagnostics, never collision probabilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from camera_rc.config import load_config
from camera_rc.data import load_data_sources
from camera_rc.metrics import MetricAccumulator
from camera_rc.training import _device
from evaluate_road_safety import (
    INPUT_SIZE, LABEL_SPACE, NOTICE, TRACKS, TRACKS_SOURCE, _assert_not_training,
    _thresholds, aggregate_obstacle_metrics, discover_pairs, road_metrics,
    obstacle_pixel_metrics,
)


CLASS_NAMES = ["road_surface", "not_road_surface"]


class IdentityLedger:
    """Hash the exact bytes decoded, then re-hash every input before publishing."""
    def __init__(self):
        self.identities = {}
        self.expected = {}

    def expect(self, path, sha256):
        if not isinstance(sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", sha256):
            raise ValueError("Dataset snapshot requires lowercase SHA256 hex")
        key = str(Path(path).expanduser().resolve(strict=True)).casefold()
        if key in self.expected and self.expected[key] != sha256:
            raise ValueError("Conflicting expected hashes for one input")
        self.expected[key] = sha256
        if key in self.identities and self.identities[key]["sha256"] != sha256:
            raise ValueError("Current input differs from its training dataset snapshot")

    def read(self, path):
        path = Path(path).expanduser().resolve(strict=True)
        payload = path.read_bytes()
        identity = {"path": str(path), "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)}
        key = str(path).casefold()
        if key in self.expected and self.expected[key] != identity["sha256"]:
            raise ValueError(f"Input SHA256 differs from training dataset snapshot: {path}")
        if key in self.identities and self.identities[key] != identity:
            raise RuntimeError(f"Input changed during evaluation: {path}")
        self.identities[key] = identity
        return payload, identity

    def identity(self, path):
        return self.read(path)[1]

    def verify_unchanged(self):
        for identity in list(self.identities.values()):
            self.identity(identity["path"])


def _json(path, ledger):
    payload, identity = ledger.read(path)
    result = json.loads(payload.decode("utf-8-sig"))
    if not isinstance(result, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return result, identity


def _report_path(value, parent):
    if not isinstance(value, str) or not value:
        raise ValueError("Training report requires a file path")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve(strict=True)


def validate_training_report(path, checkpoint, config_path, ledger):
    path = Path(path).resolve(strict=True)
    report, report_identity = _json(path, ledger)
    if type(report.get("schema_version")) is not int or report["schema_version"] != 1:
        raise ValueError("Training report requires schema_version=1")
    if (report.get("model_family") != "yolo26x-sem" or report.get("task") != "semantic"
            or report.get("class_names") != CLASS_NAMES or type(report.get("road_class_id")) is not int
            or report["road_class_id"] != 0):
        raise ValueError("Require trained yolo26x-sem with canonical road=0/nonroad=1 names")
    if report.get("training_complete") is not True or report.get("diagnostic") is not False:
        raise ValueError("Incomplete, smoke or diagnostic training is not a completed candidate")
    if not isinstance(report.get("actual_args"), dict) or not isinstance(report.get("history"), list) or not report["history"]:
        raise ValueError("Training report must retain actual_args and nonempty history")
    source = report.get("source_config")
    if not isinstance(source, dict):
        raise ValueError("Missing source_config provenance")
    config_identity = ledger.identity(config_path)
    if (_report_path(source.get("path"), path.parent) != Path(config_path).resolve()
            or source.get("sha256") != config_identity["sha256"]):
        raise ValueError("Source config path/hash differs from the completed training report")
    checkpoints = report.get("checkpoints")
    if not isinstance(checkpoints, dict) or not all(isinstance(checkpoints.get(key), dict) for key in ("best", "last")):
        raise ValueError("Training report must identify both best and last checkpoints")
    selected = ledger.identity(checkpoint)
    matched = []
    for name in ("best", "last"):
        item = checkpoints[name]
        identity = ledger.identity(_report_path(item.get("path"), path.parent))
        if (identity["sha256"] != item.get("sha256") or type(item.get("bytes")) is not int
                or identity["bytes"] != item["bytes"]):
            raise ValueError(f"Training report {name} checkpoint identity changed")
        if identity == selected:
            matched.append(name)
    if not matched:
        raise ValueError("Selected checkpoint is not the completed report's best or last checkpoint")
    dataset = report.get("dataset")
    if not isinstance(dataset, dict):
        raise ValueError("Missing training dataset provenance")
    provenance, provenance_identity = _json(_report_path(dataset.get("provenance_path"), path.parent), ledger)
    if provenance_identity["sha256"] != dataset.get("provenance_sha256"):
        raise ValueError("Dataset provenance SHA256 differs from the training report")
    return {"report": report, "identity": report_identity, "checkpoint": selected,
            "checkpoint_roles": matched, "config": config_identity,
            "dataset_provenance": provenance, "dataset_provenance_identity": provenance_identity}


def validate_input_size(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 2
            or any(type(d) is not int or d < 32 or d > 4096 or d % 32 for d in value)):
        raise ValueError("input_size must be [height,width], each a multiple of 32 in 32..4096")
    return tuple(value)


def validate_export_snapshot(completion, training, validation, ledger):
    provenance = completion["dataset_provenance"]
    identity = completion["dataset_provenance_identity"]
    folder = Path(identity["path"]).parent
    if ((folder / ".incomplete").exists() or provenance.get("complete") is not True
            or provenance.get("diagnostic_only") is not False
            or provenance.get("label_space_id") != LABEL_SPACE
            or provenance.get("names") != {str(i): name for i, name in enumerate(CLASS_NAMES)}
            or provenance.get("source_class_id_to_export_class_id") != {"0": 0, "1": 1, "255": 255}):
        raise ValueError("Require a complete non-diagnostic canonical semantic export")
    marker, _ = _json(folder / ".complete", ledger)
    if (marker.get("report_sha256") != identity["sha256"] or marker.get("diagnostic_only") is not False
            or marker.get("dataset_yaml_sha256") != provenance.get("dataset_yaml_sha256")):
        raise ValueError("Export completion marker does not bind the dataset report")
    source = provenance.get("source_config", {})
    if (_report_path(source.get("path"), folder) != Path(completion["config"]["path"])
            or source.get("sha256") != completion["config"]["sha256"]):
        raise ValueError("Dataset export and training source config differ")
    yaml_path = _report_path(provenance.get("dataset_yaml"), folder)
    ledger.expect(yaml_path, provenance.get("dataset_yaml_sha256"))
    ledger.identity(yaml_path)
    exported, train_hashes, train_rgb_hashes = {}, set(), set()
    for split, sources in (("train", training), ("val", validation)):
        output = provenance.get("manifest_outputs", {}).get(split, {})
        manifest = _report_path(output.get("path"), folder)
        ledger.expect(manifest, output.get("sha256"))
        payload, _ = ledger.read(manifest)
        records = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
        reference = {}
        for source in sources:
            manifest_key = str(Path(source["manifest"]).resolve()).casefold()
            reference.update({(manifest_key, index): row for index, row in enumerate(source["rows"])})
        seen, source_path_cache, source_hashes = set(), {}, {}
        for record_index, row in enumerate(records, 1):
            if not isinstance(row, dict) or row.get("export_split") != split:
                raise ValueError("Invalid export manifest row or split")
            raw_source = row.get("source_manifest")
            if not isinstance(raw_source, str) or not raw_source:
                raise ValueError("Export row requires source manifest path")
            if raw_source not in source_path_cache:
                source_path_cache[raw_source] = _report_path(raw_source, manifest.parent)
            source_manifest = source_path_cache[raw_source]
            index = row.get("source_manifest_index")
            if type(index) is not int or index < 0:
                raise ValueError("Missing export source manifest index")
            key = (str(source_manifest).casefold(), index)
            if key in seen or key not in reference:
                raise ValueError("Export contains a duplicate or unexpected source row")
            seen.add(key)
            original = reference[key]
            source_digest = row.get("source_manifest_sha256")
            if source_manifest not in source_hashes:
                ledger.expect(source_manifest, source_digest)
                source_hashes[source_manifest] = source_digest
            elif source_hashes[source_manifest] != source_digest:
                raise ValueError("Conflicting source manifest hashes in exported rows")
            for exported_key, original_key in (("input_image", "image"), ("input_target", "target")):
                if _report_path(row.get(exported_key), manifest.parent) != Path(original[original_key]).resolve():
                    raise ValueError("Export input image/target differs from original source config")
            if any(row.get(field) != original.get(field) for field in ("dataset_id", "group_id", "origin_id")):
                raise ValueError("Export original scene identity differs from source config")
            if split == "val":
                ledger.expect(original["image"], row.get("image_sha256"))
                ledger.expect(original["target"], row.get("input_target_sha256"))
            else:
                for field in ("image_sha256", "source_image_sha256", "image_rgb_sha256"):
                    digest = row.get(field)
                    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
                        raise ValueError("Export training image identities are incomplete")
                train_hashes.update((row["image_sha256"], row["source_image_sha256"]))
                train_rgb_hashes.add(row["image_rgb_sha256"])
            if record_index % 10000 == 0:
                print(json.dumps({"event": "verify_training_export", "split": split,
                                  "records": record_index, "total": len(records)}), flush=True)
        if seen != set(reference):
            raise ValueError("Export manifest omits original source rows; partial datasets are not full evaluations")
        exported[split] = ledger.identity(manifest)
    return {"folder": folder, "manifests": exported, "training_image_sha256": train_hashes,
            "training_rgb_sha256": train_rgb_hashes}


def validate_semantic_model(wrapper, *, model_class=None, head_class=None):
    if model_class is None or head_class is None:
        from ultralytics.nn.tasks import SemanticSegmentationModel
        from ultralytics.nn.modules.head import SemanticSegment
        model_class, head_class = SemanticSegmentationModel, SemanticSegment
    model = wrapper.model
    if wrapper.task != "semantic" or not isinstance(model, model_class):
        raise ValueError("Expected a native PyTorch semantic segmentation model")
    head = model.model[-1]
    if (not isinstance(head, head_class) or head.nc != 2 or head.export or head.bake_argmax):
        raise ValueError("Require a two-class non-export SemanticSegment logits head")
    names = model.names
    if names != dict(enumerate(CLASS_NAMES)) and names != CLASS_NAMES:
        raise ValueError("Checkpoint class names must be road_surface=0 and not_road_surface=1")
    architecture = model.yaml
    if (architecture.get("scale") != "x"
            or Path(architecture.get("yaml_file", "")).stem not in {"yolo26-sem", "yolo26x-sem"}):
        raise ValueError("Checkpoint architecture must be YOLO26x-sem, scale x")
    return model


def logits_at_input(logits, input_size, batch_size):
    """Reject baked class maps and train-mode auxiliary tuples."""
    if (not isinstance(logits, torch.Tensor) or not torch.is_floating_point(logits)
            or logits.ndim != 4 or logits.shape[:2] != (batch_size, 2)
            or min(logits.shape[2:]) < 1 or not torch.isfinite(logits).all()):
        raise ValueError("Expected finite raw floating [B,2,H,W] semantic logits")
    return F.interpolate(logits.float(), size=input_size, mode="bilinear", align_corners=False)


def restore_road_probability(logits, input_size, original_size):
    full_logits = logits_at_input(logits, input_size, 1)
    road = full_logits.softmax(dim=1)[:, 0:1]
    return F.interpolate(road, size=original_size, mode="bilinear", align_corners=False)[0, 0].detach().cpu().numpy()


def image_tensor(rgb, input_size):
    image = Image.fromarray(rgb).resize((input_size[1], input_size[0]), Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1).copy()) / 255.0


class YoloSemanticPredictor:
    def __init__(self, checkpoint, *, input_size=INPUT_SIZE, device="cuda", amp=False):
        from ultralytics import YOLO
        self.input_size = validate_input_size(input_size)
        self.device = _device(device)
        self.amp = bool(amp) and self.device.type == "cuda"
        wrapper = YOLO(str(Path(checkpoint).resolve(strict=True)), task="semantic")
        self.model = validate_semantic_model(wrapper).float().eval()
        if any(not torch.isfinite(value).all() for value in self.model.state_dict().values()):
            raise ValueError("Checkpoint contains nonfinite weights")
        self.model = self.model.to(self.device)

    @torch.inference_mode()
    def predict_logits(self, images):
        tensor = torch.stack([image_tensor(rgb, self.input_size) for rgb in images]).to(self.device)
        with torch.autocast(device_type=self.device.type, enabled=self.amp):
            logits = self.model(tensor)
        return logits_at_input(logits, self.input_size, len(images))

    @torch.inference_mode()
    def __call__(self, rgb):
        logits = self.predict_logits([rgb])
        return restore_road_probability(logits, self.input_size, tuple(rgb.shape[:2]))


def read_labelled_pair(image_path, target_path, ledger):
    image_bytes, image_identity = ledger.read(image_path)
    target_bytes, target_identity = ledger.read(target_path)
    with Image.open(io.BytesIO(image_bytes)) as image:
        rgb = np.asarray(image.convert("RGB")).copy()
    with Image.open(io.BytesIO(target_bytes)) as target:
        labels = np.asarray(target).copy()
    if labels.ndim != 2 or labels.shape != rgb.shape[:2] or labels.dtype.kind not in "iu":
        raise ValueError("Image and single-channel integer label shapes must match")
    if not np.isin(labels, [0, 1, 255]).all():
        raise ValueError("Labels must already be canonical 0/1/255; no implicit remapping")
    return rgb, labels, image_identity, target_identity


def evaluate_native_source(source, predictor, ledger, batch_size=4, runtime_threshold=.65):
    _thresholds(runtime_threshold)
    metrics = MetricAccumulator("segmentation", 2)
    threshold_metrics = MetricAccumulator("segmentation", 2)
    batches, loss_sum, samples = 0, 0.0, 0
    for start in range(0, len(source["rows"]), batch_size):
        selected = source["rows"][start:start + batch_size]
        images, labels = [], []
        for row in selected:
            rgb, target, _, _ = read_labelled_pair(row["image"], row["target"], ledger)
            target = np.asarray(Image.fromarray(target).resize((INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.NEAREST)).copy()
            if not (target != 255).any():
                raise ValueError("No supervised native GT pixels after nearest resize")
            images.append(rgb)
            labels.append(torch.from_numpy(target.astype(np.int64)))
        prediction = predictor.predict_logits(images)
        # The larger-input condition still evaluates exactly the historical GT grid.
        prediction = F.interpolate(prediction.float(), size=INPUT_SIZE, mode="bilinear", align_corners=False)
        target = torch.stack(labels).to(prediction.device)
        loss = F.cross_entropy(prediction, target, ignore_index=255)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite native validation loss")
        metrics.update(prediction, target)
        # Keep the operational threshold's road coverage visible: predicting
        # nonroad everywhere must not masquerade as useful obstacle rejection.
        road = prediction.softmax(dim=1)[:, 0] >= runtime_threshold
        threshold_metrics.update(torch.stack((road, ~road), dim=1).float(), target)
        loss_sum += float(loss) * len(selected)
        samples += len(selected)
        batches += 1
        if batches % 100 == 0 or samples == len(source["rows"]):
            print(json.dumps({"event": "native_validation", "source": source["name"],
                              "images": samples, "total": len(source["rows"])}), flush=True)
    result = {"loss": loss_sum / samples, **metrics.compute(), "batches": batches,
              "images": samples, "partial_validation": False}
    result["road_safety_pixels"] = road_metrics(result["confusion_matrix"])
    threshold_result = threshold_metrics.compute()
    threshold_result.update(road_metrics(threshold_result["confusion_matrix"]))
    threshold_result.update(
        road_iou=threshold_result["class_iou"][0], threshold=float(runtime_threshold), comparison=">=",
        decision="FP32 softmax road=0 probability >= runtime threshold; GT ignore=255 excluded. This differs from argmax, whose equal-logit tie chooses road=0.",
        comparison_limit="Candidate operating-threshold coverage diagnostic. Historical TinyUNet reports do not contain this metric; no paired threshold improvement is claimed.",
    )
    result["runtime_threshold_metrics"] = threshold_result
    return result


def _assert_public_hash_not_training(rgb, identity, snapshot):
    digest = hashlib.sha256(f"RGB:{rgb.shape[1]}:{rgb.shape[0]}:".encode("ascii"))
    digest.update(rgb.tobytes())
    if identity["sha256"] in snapshot["training_image_sha256"] or digest.hexdigest() in snapshot["training_rgb_sha256"]:
        raise ValueError("Public evaluation image bytes/pixels occur in exported training data")


def evaluate_public_track(discovered, predictor, track, ledger, runtime_threshold, snapshot):
    rows = []
    for index, pair in enumerate(discovered["pairs"], 1):
        rgb, target, image_identity, target_identity = read_labelled_pair(pair["image"], pair["target"], ledger)
        _assert_public_hash_not_training(rgb, image_identity, snapshot)
        row = obstacle_pixel_metrics(predictor(rgb), target, runtime_threshold)
        row.update(frame_id=pair["frame_id"], original_image_size=list(target.shape),
                   image=image_identity, ground_truth=target_identity)
        rows.append(row)
        print(json.dumps({"event": "public_gt_image", "track": track, "image": index,
                          "total": len(discovered["pairs"]), "frame_id": pair["frame_id"]}), flush=True)
    return {"dataset": TRACKS[track]["name"], "role": "evaluation_only_public_labelled_subset",
            "root": discovered["root"], "available_images": discovered["available_images"],
            "unlabelled_images_not_evaluated": discovered["unlabelled_images_not_evaluated"],
            "expected_public_gt_images": discovered["expected_public_gt_images"],
            "labels": {"1": "labelled obstacle/anomaly; only evaluated denominator", "0": "usual; not assumed road", "255": "ignore"},
            "source_format_reference": TRACKS_SOURCE, "ground_truth_resizing": "none",
            "probability_resizing": "bilinear to original GT size, align_corners=False, before thresholding",
            "aggregate": aggregate_obstacle_metrics(rows, runtime_threshold), "per_image": rows}


def evaluate(config_path, checkpoint, training_report, output, *, input_size=INPUT_SIZE,
             obstacle_root=None, anomaly_root=None, runtime_threshold=.65, device="cuda", batch_size=4,
             evaluation_contract=None):
    if evaluation_contract is not None:
        if obstacle_root is not None or anomaly_root is not None or runtime_threshold != .65:
            raise ValueError("Public roots and threshold come only from the frozen evaluation contract")
        from evaluate_frozen_road import evaluate as evaluate_frozen
        return evaluate_frozen(evaluation_contract, "yolo26x-sem", config_path, checkpoint, training_report,
                               output, input_size=input_size, device=device, batch_size=batch_size)
    _thresholds(runtime_threshold)
    input_size = validate_input_size(input_size)
    if type(batch_size) is not int or not 1 <= batch_size <= 64:
        raise ValueError("batch_size must be an integer in 1..64")
    config_path, checkpoint, output = [Path(p).expanduser().resolve() for p in (config_path, checkpoint, output)]
    if output.exists():
        raise FileExistsError(f"Choose a new report path: {output}")
    ledger = IdentityLedger()
    completion = validate_training_report(training_report, checkpoint, config_path, ledger)
    config = load_config(config_path)
    if (config.get("task") != "segmentation" or config.get("model", {}).get("num_classes") != 2
            or config.get("data", {}).get("label_space_id") != LABEL_SPACE
            or tuple(config.get("data", {}).get("image_size", [])) != INPUT_SIZE):
        raise ValueError("Source config must use canonical two-class road_surface_v1 and GT image_size=[192,320]")
    # Hash manifests before parsing; any mutation during their read or evaluation fails publication.
    data = config["data"]
    train_specs = data.get("train_sources", [{"manifest": data.get("train_manifest")}])
    val_specs = data.get("val_manifests", [data.get("val_manifest")])
    for item in [*train_specs, *val_specs]:
        ledger.identity(item if isinstance(item, str) else item["manifest"])
    training, validation = load_data_sources(config)
    snapshot = validate_export_snapshot(completion, training, validation, ledger)
    discovered = {name: discover_pairs(root, name) for name, root in (("obstacle", obstacle_root), ("anomaly", anomaly_root)) if root is not None}
    for value in discovered.values():
        _assert_not_training(value, training)
        for pair in value["pairs"]:
            rgb, _, identity, _ = read_labelled_pair(pair["image"], pair["target"], ledger)
            _assert_public_hash_not_training(rgb, identity, snapshot)
    amp = bool(config.get("training", {}).get("amp", False))
    predictor = YoloSemanticPredictor(checkpoint, input_size=input_size, device=device, amp=amp)
    native = {source["name"]: evaluate_native_source(source, predictor, ledger, batch_size, runtime_threshold) for source in validation}
    tracks = {name: evaluate_public_track(value, predictor, name, ledger, runtime_threshold, snapshot) for name, value in discovered.items()}
    matched = input_size == INPUT_SIZE
    report = {
        "format_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "notice": NOTICE,
        "evaluation_only": True, "training_performed": False, "deployment_validated": False,
        "models_validated_for_control": False, "real_performance_validated": False,
        "checkpoint": completion["checkpoint"], "config": completion["config"], "checkpoint_diagnostic": False,
        "model_family": "yolo26x-sem", "task": "semantic", "class_names": CLASS_NAMES,
        "input_size": list(input_size), "native_gt_size": list(INPUT_SIZE), "label_space_id": LABEL_SPACE, "road_class_id": 0,
        "device": str(predictor.device), "amp_requested": amp, "amp_effective": predictor.amp,
        "batch_size": batch_size, "runtime_threshold": runtime_threshold,
        "accuracy_condition": "matched_native_input_192x320" if matched else "larger_input_separate_accuracy_condition",
        "matched_input_resolution_to_native_baseline": matched,
        "protocol": {"rgb": "PIL RGB bilinear direct resize; float32 [0,1]; no letterbox or normalization",
                     "native_gt": "nearest resize to 192x320; 0=road,1=nonroad,255=ignore",
                     "native_decision": "raw logits bilinear to model input then 192x320, argmax; ties choose road=0",
                     "road_probability": "raw logits bilinear to model input, softmax channel 0, then probability bilinear to original GT size",
                     "resolution_limit": "Larger model inputs change the accuracy condition and must not be described as a matched-resolution model gain."},
        "validation_sources": [{"name": source["name"], "images": len(source["rows"]),
                                "manifest": ledger.identity(source["manifest"])} for source in validation],
        "training_manifests_checked_for_exact_path_overlap": [ledger.identity(source["manifest"]) for source in training],
        "overlap_check_limit": "Image/source_image paths and original scene identities in source_config are checked; unknown pretraining or copied images cannot be ruled out.",
        "training_completion": {"report": completion["identity"], "checkpoint_roles": completion["checkpoint_roles"],
                                "dataset_provenance": completion["dataset_provenance_identity"],
                                "export_manifests": snapshot["manifests"]},
        "native_sources": native, "public_obstacle_anomaly_subsets": tracks,
        "native_resolution_note": "Native GT remains nearest 192x320; obstacle/anomaly GT retains original pixels. These protocols must not be pooled.",
        "content_identity_note": "All evaluated image/GT bytes, checkpoints, config, manifests and training provenance are hashed and rechecked before publication. Historical TinyUNet native image/GT bytes were not hashed.",
        "input_content_identities": sorted(ledger.identities.values(), key=lambda item: item["path"].casefold()),
    }
    # Metadata assembly above also reads manifest bytes. Recheck all identities once
    # more so changed evidence cannot silently enter an otherwise completed report.
    ledger.verify_unchanged()
    if (snapshot["folder"] / ".incomplete").exists():
        raise RuntimeError("Dataset export became incomplete during evaluation")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("config", "checkpoint", "training-report", "output"):
        parser.add_argument("--" + key, required=True, type=Path)
    parser.add_argument("--input-size", nargs=2, type=int, default=list(INPUT_SIZE), metavar=("H", "W"))
    parser.add_argument("--obstacle-root", type=Path)
    parser.add_argument("--anomaly-root", type=Path)
    parser.add_argument("--runtime-threshold", type=float, default=.65)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--evaluation-contract", type=Path, help="Independent frozen cohort; --config remains this model's original training config")
    args = parser.parse_args()
    report = evaluate(args.config, args.checkpoint, args.training_report, args.output, input_size=args.input_size,
                      obstacle_root=args.obstacle_root, anomaly_root=args.anomaly_root,
                      runtime_threshold=args.runtime_threshold, device=args.device, batch_size=args.batch_size,
                      evaluation_contract=args.evaluation_contract)
    print(json.dumps({"output": str(args.output.resolve()), "condition": report.get("accuracy_condition", report.get("protocol_id")),
                      "validation_images": sum(item["images"] for item in report["validation_sources"]),
                      "deployment_validated": False}), flush=True)


if __name__ == "__main__":
    main()
