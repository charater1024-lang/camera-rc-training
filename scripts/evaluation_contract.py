"""Independent, immutable evaluation cohorts. No model training config is rewritten."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from camera_rc.data import read_manifest, validate_splits
from evaluate_road_safety import INPUT_SIZE, LABEL_SPACE, discover_pairs, _thresholds
from evaluate_yolo_semantic import IdentityLedger, _json, read_labelled_pair


PROTOCOL = "frozen_road_pixels_v1"
PUBLIC_ROLE = "reused_development_regression_only_not_training_or_threshold_tuning"


def rgb_sha256(rgb):
    digest = hashlib.sha256(f"RGB:{rgb.shape[1]}:{rgb.shape[0]}:".encode("ascii"))
    digest.update(rgb.tobytes())
    return digest.hexdigest()


def resolve(value, parent):
    if not isinstance(value, str) or not value:
        raise ValueError("Nonempty path required")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else parent / path).resolve(strict=True)


def manifest_rows(path, ledger):
    ledger.identity(path)
    rows = read_manifest(path)
    ledger.identity(path)
    for row in rows:
        if (row.get("task", "segmentation") != "segmentation"
                or row.get("label_space_id", LABEL_SPACE) != LABEL_SPACE):
            raise ValueError("Evaluation requires canonical road_surface_v1 segmentation")
    return rows


def freeze_pair(row, ledger, *, public=False):
    rgb, target, image, gt = read_labelled_pair(row["image"], row["target"], ledger)
    evaluated = target if public else np.asarray(Image.fromarray(target).resize(
        (INPUT_SIZE[1], INPUT_SIZE[0]), Image.Resampling.NEAREST))
    counts = {str(label): int((evaluated == label).sum()) for label in (0, 1, 255)}
    if not public and counts["0"] + counts["1"] == 0:
        raise ValueError("No supervised pixels after native GT resizing")
    return {"row": row, "image": image, "ground_truth": gt,
            "image_rgb_sha256": rgb_sha256(rgb), "original_size": list(target.shape),
            "evaluated_gt_pixels": counts}


def prepare(spec_path, output):
    ledger = IdentityLedger()
    spec_path, output = Path(spec_path).resolve(strict=True), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new contract directory: {output}")
    spec, spec_identity = _json(spec_path, ledger)
    if spec.get("schema_version") != 1 or not isinstance(spec.get("sources"), list) or not spec["sources"]:
        raise ValueError("Spec requires schema_version=1 and nonempty sources")
    threshold = spec.get("runtime_threshold", .65)
    _thresholds(threshold)
    retain_duplicates = spec.get("retain_source_duplicate_multiplicity", False)
    if type(retain_duplicates) is not bool:
        raise ValueError("retain_source_duplicate_multiplicity must be a boolean")
    sources, all_rows, seen_names, hashes, pixels, duplicates = [], [], set(), {}, {}, []
    for item in spec["sources"]:
        name = item.get("name")
        if not isinstance(name, str) or not name.strip() or name in seen_names:
            raise ValueError("Unique nonempty evaluation source names required")
        seen_names.add(name)
        path = resolve(item.get("manifest"), spec_path.parent)
        rows = manifest_rows(path, ledger)
        # Duplicated frames inflate denominators, including copies under other paths.
        frozen = []
        source_duplicates = []
        for index, row in enumerate(rows):
            record = freeze_pair(row, ledger)
            previous = hashes.get(record["image"]["sha256"]) or pixels.get(record["image_rgb_sha256"])
            if previous:
                if not retain_duplicates or previous["source"] != name:
                    raise ValueError("Duplicate evaluation image bytes or RGB pixels")
                duplicate = {"source": name, "index": index, "image": record["image"],
                             "image_rgb_sha256": record["image_rgb_sha256"], "first_occurrence": previous}
                duplicates.append(duplicate)
                source_duplicates.append(duplicate)
            first = {"source": name, "index": index, "image": record["image"]}
            hashes.setdefault(record["image"]["sha256"], first)
            pixels.setdefault(record["image_rgb_sha256"], first)
            frozen.append(record)
        all_rows.extend(rows)
        sources.append({"name": name, "manifest": ledger.identity(path), "images": len(rows), "records": frozen,
                        "duplicate_frames_retained": len(source_duplicates)})
        print(json.dumps({"event": "freeze_evaluation_source", "source": name, "images": len(rows)}), flush=True)
    validate_splits([], all_rows)
    public = {}
    for track in ("obstacle", "anomaly"):
        if not spec.get(track + "_root"):
            continue
        discovered = discover_pairs(resolve(spec[track + "_root"], spec_path.parent), track)
        records = []
        for pair in discovered.pop("pairs"):
            pair = {key: str(value) if isinstance(value, Path) else value for key, value in pair.items()}
            record = freeze_pair(pair, ledger, public=True)
            if record["image"]["sha256"] in hashes or record["image_rgb_sha256"] in pixels:
                raise ValueError("Duplicate public/native evaluation image")
            hashes[record["image"]["sha256"]] = {"public_track": track}
            pixels[record["image_rgb_sha256"]] = {"public_track": track}
            records.append(record)
        public[track] = {**discovered, "records": records, "role": PUBLIC_ROLE}
    ledger.verify_unchanged()
    result = {"schema_version": 1, "complete": True, "protocol_id": PROTOCOL,
              "source_spec": spec_identity, "label_space_id": LABEL_SPACE,
              "native_gt_size": list(INPUT_SIZE), "runtime_threshold": threshold,
              "public_role": PUBLIC_ROLE, "sources": sources, "public_tracks": public,
              "retain_source_duplicate_multiplicity": retain_duplicates,
              "retained_within_source_duplicates": duplicates,
              "duplicate_note": "Official source sample multiplicity is preserved only when explicitly selected; repeated images have repeated weight in pixel metrics. These are not independent observations. Cross-source/public duplicates remain forbidden.",
              "training_authorized": False, "deployment_validated": False}
    # Nothing is published until every byte and mask has been checked.
    output.mkdir(parents=True)
    path = output / "contract.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    with (output / ".complete").open("x", encoding="utf-8") as stream:
        json.dump({"contract_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}, stream)
    return result


def load(path, ledger):
    path = Path(path).resolve(strict=True)
    contract, identity = _json(path, ledger)
    marker, _ = _json(path.parent / ".complete", ledger)
    if (marker.get("contract_sha256") != identity["sha256"] or contract.get("schema_version") != 1
            or contract.get("complete") is not True or contract.get("protocol_id") != PROTOCOL
            or contract.get("label_space_id") != LABEL_SPACE or contract.get("native_gt_size") != list(INPUT_SIZE)
            or contract.get("public_role") != PUBLIC_ROLE):
        raise ValueError("Incomplete or mismatched frozen evaluation contract")
    _thresholds(contract.get("runtime_threshold"))
    if not isinstance(contract.get("sources"), list) or not contract["sources"]:
        raise ValueError("Frozen cohort has no sources")
    validation, public = [], {}
    names, rows = set(), []
    for source in contract["sources"]:
        name = source["name"]
        if not isinstance(name, str) or not name or name in names:
            raise ValueError("Duplicate/invalid frozen source name")
        names.add(name)
        manifest = source["manifest"]
        ledger.expect(manifest["path"], manifest["sha256"])
        current = manifest_rows(manifest["path"], ledger)
        if source["images"] != len(current) or [r["row"] for r in source["records"]] != current:
            raise ValueError("Frozen manifest membership changed")
        validation.append({"name": name, "manifest": manifest["path"], "rows": current})
        rows.extend(current)
    validate_splits([], rows)
    for track, item in contract["public_tracks"].items():
        if track not in ("obstacle", "anomaly") or item.get("role") != PUBLIC_ROLE:
            raise ValueError("Invalid public development regression track")
        discovered = discover_pairs(item["root"], track)
        actual = [{key: str(value) if isinstance(value, Path) else value for key, value in pair.items()}
                  for pair in discovered["pairs"]]
        if [r["row"] for r in item["records"]] != actual:
            raise ValueError("Public frozen cohort membership changed")
        public[track] = discovered
    # Read+decode all inputs before any GPU allocation. Later ledger reads and
    # final rehashing also reject changes between preflight and report writing.
    for records, is_public in [(s["records"], False) for s in contract["sources"]] + [
            (s["records"], True) for s in contract["public_tracks"].values()]:
        for record in records:
            for key in ("image", "ground_truth"):
                ledger.expect(record[key]["path"], record[key]["sha256"])
            if freeze_pair(record["row"], ledger, public=is_public) != record:
                raise ValueError("Frozen evaluation content/denominator changed")
    return contract, identity, validation, public


def assert_no_leakage(contract, training_sources, snapshot):
    train = [row for source in training_sources for row in source["rows"]]
    validation = [r["row"] for source in contract["sources"] for r in source["records"]]
    validate_splits(train, validation)
    train_paths = {str(Path(row[key]).resolve()).casefold() for row in train
                   for key in ("image", "source_image") if row.get(key)}
    for source in [*contract["sources"], *contract["public_tracks"].values()]:
        for record in source["records"]:
            if (record["image"]["path"].casefold() in train_paths
                    or record["image"]["sha256"] in snapshot["training_image_sha256"]
                    or record["image_rgb_sha256"] in snapshot["training_rgb_sha256"]):
                raise ValueError("Frozen evaluation/train image bytes, RGB pixels or paths overlap")


def assert_comparable(left, right):
    """Compare evaluation identity, never require training configurations to match."""
    for report in (left, right):
        if (report.get("evaluation_complete") is not True or report.get("training_performed") is not False
                or report.get("deployment_validated") is not False):
            raise ValueError("Incomplete or invalid evaluation report")
    for key in ("evaluation_contract", "protocol_id", "input_size", "native_gt_size", "runtime_threshold",
                "label_space_id", "precision_condition", "validation_sources", "gt_denominators"):
        if left.get(key) is None or left.get(key) != right.get(key):
            raise ValueError(f"Evaluation comparison mismatch: {key}")
    for key in ("native_sources", "public_obstacle_anomaly_subsets"):
        if set(left[key]) != set(right[key]):
            raise ValueError(f"Evaluation source-set mismatch: {key}")
    for name, source in left["native_sources"].items():
        other = right["native_sources"][name]
        for key in ("confusion_matrix",):
            if [sum(r) for r in source[key]] != [sum(r) for r in other[key]]:
                raise ValueError("Native evaluation denominator mismatch")
    for track, source in left["public_obstacle_anomaly_subsets"].items():
        def identities(value):
            return [(r["frame_id"], r["image"], r["ground_truth"], r["labelled_obstacle_pixels"])
                    for r in value["per_image"]]
        if identities(source) != identities(right["public_obstacle_anomaly_subsets"][track]):
            raise ValueError("Public evaluation denominator/content mismatch")

