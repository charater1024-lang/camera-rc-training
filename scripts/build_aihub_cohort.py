"""Freeze AIHub13-15 plus deterministic BDD/VKITTI training replay.

Retains every official validation row and original scene/origin metadata. This
is metadata preparation only: run prepare_yolo_semantic.py afterwards for full
image/mask validation and pixel-content cross-split duplicate checks.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path

from camera_rc.config import load_config
from camera_rc.data import load_data_sources, read_manifest, validate_splits

BDD = "bdd100k_drivable"
VKITTI = "vkitti2_0_3"


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _rank(seed, value):
    return hashlib.sha256(_json([seed, value]).encode("utf-8")).hexdigest()


def row_identity(row):
    return (row["dataset_id"], row["group_id"],
            row.get("sample_id", Path(row["image"]).name),
            row.get("source_scene"), row.get("variant"), row.get("camera_id"), row.get("frame_id"))


def select_replay(rows, count, seed, stratified=False):
    """Stable identity ranking, no replacement; proportional VKITTI strata."""
    if type(count) is not int or not 0 < count <= len(rows):
        raise ValueError("Replay count must be positive and within available training rows")
    if type(seed) is not int:
        raise ValueError("Integer seed required")
    strata = defaultdict(list)
    identities = set()
    for row in rows:
        identity = row_identity(row)
        if identity in identities:
            raise ValueError("Duplicate canonical frame identity in replay source")
        identities.add(identity)
        if stratified:
            if not row.get("source_scene") or not row.get("variant") or "camera_id" not in row:
                raise ValueError("VKITTI replay requires scene, variant and camera metadata")
            key = (row["source_scene"], row["variant"], row["camera_id"])
        else:
            key = ("all",)
        strata[key].append(row)
    # Largest-remainder apportionment, using integer arithmetic.
    quotas = {k: count * len(v) // len(rows) for k, v in strata.items()}
    remaining = count - sum(quotas.values())
    priority = sorted(strata, key=lambda k: (-(count * len(strata[k]) % len(rows)), _rank(seed, k)))
    for k in priority[:remaining]:
        quotas[k] += 1
    selected, stats = [], []
    for k in sorted(strata, key=_json):
        ranked = sorted(strata[k], key=lambda r: (_rank(seed, row_identity(r)), _json(row_identity(r))))
        selected.extend(ranked[:quotas[k]])
        stats.append({"stratum": list(k), "available": len(ranked), "selected": quotas[k]})
    if len(selected) != count:
        raise RuntimeError("Replay apportionment count mismatch")
    return selected, stats


def _write(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _rows_with_provenance(rows, manifest, manifest_hash):
    result = []
    for index, original in enumerate(rows):
        row = dict(original)
        for field in ("image", "target", "source_image", "target_original"):
            if isinstance(row.get(field), str):
                p = Path(row[field])
                row[field] = str((p if p.is_absolute() else manifest.parent / p).resolve())
        row["cohort_input_manifest"] = str(manifest)
        row["cohort_input_manifest_sha256"] = manifest_hash
        row["cohort_input_manifest_index"] = index
        result.append(row)
    return result


def build(baseline_config, aihub_prepared, output, bdd_count=10000, vkitti_count=9177, seed=42):
    baseline_config, aihub_prepared = Path(baseline_config).resolve(strict=True), Path(aihub_prepared).resolve(strict=True)
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError("Use a fresh cohort output directory")
    source_hashes = {}

    def fingerprint(path):
        p = Path(path).resolve(strict=True)
        before = p.stat()
        digest = sha256(p)
        after = p.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Source changed while hashing: {p}")
        source_hashes[str(p)] = digest
        return digest

    fingerprint(baseline_config)
    cfg = load_config(baseline_config)
    if cfg.get("task") != "segmentation" or cfg.get("model", {}).get("num_classes") != 2 or cfg["data"].get("label_space_id") != "road_surface_v1":
        raise ValueError("Baseline requires road_surface_v1 two-class segmentation")
    # Bind manifests before loading so an in-flight edit cannot escape provenance.
    data = cfg["data"]
    input_specs = list(data.get("train_sources", [{"manifest": data.get("train_manifest")}]))
    input_specs += data.get("val_manifests", [data.get("val_manifest")])
    for spec in input_specs:
        path = spec if isinstance(spec, str) else spec["manifest"]
        fingerprint(path)
    old_train_sources, old_val_sources = load_data_sources(cfg)
    old_train, old_val = [], []
    for sources, destination in ((old_train_sources, old_train), (old_val_sources, old_val)):
        for source in sources:
            p = Path(source["manifest"])
            destination.extend(_rows_with_provenance(source["rows"], p, source_hashes[str(p)]))
    if set(r["dataset_id"] for r in old_train) != {BDD, VKITTI}:
        raise ValueError("Expected exactly previous BDD/VKITTI training sources")
    if (aihub_prepared / ".incomplete").exists() or (aihub_prepared / "INCOMPLETE").exists():
        raise ValueError("AIHub preparation is incomplete")
    marker_path, report_path = aihub_prepared / ".complete", aihub_prepared / "report.json"
    fingerprint(marker_path)
    report_hash = fingerprint(report_path)
    marker = json.loads(marker_path.read_text(encoding="utf-8-sig"))
    prepared = json.loads(report_path.read_text(encoding="utf-8-sig"))
    if (marker.get("report_sha256") != report_hash or prepared.get("complete") is not True
            or prepared.get("diagnostic") is not False or prepared.get("label_space_id") != "road_surface_v1"
            or prepared.get("format") != "camera_rc_aihub_road_v1"
            or prepared.get("cross_source_clip_leakage_checked") is not True
            or prepared.get("exact_and_decoded_rgb_duplicates_checked") is not True):
        raise ValueError("AIHub preparation lacks a bound completed identity/duplicate audit")
    aihub = {}
    for split in ("train", "val"):
        record = prepared["manifests"][split]
        p = Path(record["manifest"]).resolve(strict=True)
        if p != aihub_prepared / (split + ".jsonl") or fingerprint(p) != record["sha256"]:
            raise ValueError("AIHub manifest differs from preparation report")
        rows = read_manifest(p)
        if len(rows) != record["samples"] or record.get("all_samples_decoded") is not True:
            raise ValueError("AIHub manifest counts/completion mismatch")
        for row in rows:
            if row.get("source_dataset_number") not in (13, 14, 15) or row.get("split") != split:
                raise ValueError("Only supported AIHub13-15 official split rows may be included")
            if row.get("source_split") != ("Training" if split == "train" else "Validation"):
                raise ValueError("AIHub publisher split changed")
        aihub[split] = _rows_with_provenance(rows, p, record["sha256"])
    # Check *all* previously trained rows against all holdouts, before replay
    # sampling. A sampled subset must not hide a contaminated baseline.
    validate_splits(old_train + aihub["train"], old_val + aihub["val"])
    bdd, bdd_stats = select_replay([r for r in old_train if r["dataset_id"] == BDD], bdd_count, seed)
    vkitti, vkitti_stats = select_replay([r for r in old_train if r["dataset_id"] == VKITTI], vkitti_count, seed, True)
    train, val = aihub["train"] + bdd + vkitti, old_val + aihub["val"]
    validate_splits(train, val)
    output.mkdir(parents=True, exist_ok=False)
    (output / ".incomplete").write_text("Cohort incomplete; do not train.\n", encoding="utf-8")
    report = {"schema_version": 1, "format": "camera_rc_aihub_replay_cohort_v1", "complete": False,
              "seed": seed, "selection": "SHA256(seed, original dataset/group/sample/scene/variant/camera/frame identity); no replacement",
              "vkitti_apportionment": "Proportional original scene x variant x camera counts, largest remainder",
              "validation_policy": "All original holdouts plus all AIHub13-15 official Validation, with unchanged scene/origin identities",
              "full_original_split_identity_checked": True,
              "pixel_content_checks_required_in_semantic_export": True,
              "source_hashes": source_hashes, "bdd_replay": bdd_stats, "vkitti_replay": vkitti_stats,
              "all_prior_training_rows_checked": len(old_train),
              "training_started": False, "deployment_approved": False}
    for split, rows in (("train", train), ("val", val)):
        p = output / (split + ".jsonl")
        with p.open("x", encoding="utf-8", newline="\n") as f:
            for row in rows:
                f.write(_json(row) + "\n")
        report[split] = {"manifest": str(p), "sha256": sha256(p), "samples": len(rows),
                         "by_dataset": dict(Counter(r["dataset_id"] for r in rows)),
                         "groups_by_dataset": {ds: len({r["group_id"] for r in rows if r["dataset_id"] == ds})
                                               for ds in sorted({r["dataset_id"] for r in rows})}}
    new_cfg = copy.deepcopy(cfg)
    new_cfg["data"] = {"label_space_id": "road_surface_v1", "image_size": cfg["data"].get("image_size", [192, 320]),
                       "train_manifest": str(output / "train.jsonl"), "val_manifest": str(output / "val.jsonl")}
    new_cfg["training"].pop("init_from", None)
    new_cfg["training"].pop("samples_per_epoch", None)
    new_cfg["training"].update(epochs=3, seed=seed, batch_size=32, num_workers=4,
                               output_dir=str(output / "native_training_not_run"))
    new_cfg["cohort"] = {"report": str(output / "report.json"), "model_initializer": "Pass the verified YOLO26x-sem v2 best weights AND --init-training-report to train_yolo_semantic.py; this native config does not load an initializer."}
    _write(output / "config.json", new_cfg)
    for path, digest in source_hashes.items():
        if sha256(path) != digest:
            raise RuntimeError(f"Source provenance changed while constructing cohort: {path}")
    report.update(complete=True, config={"path": str(output / "config.json"), "sha256": sha256(output / "config.json")})
    _write(output / "report.json", report)
    (output / ".incomplete").unlink()
    _write(output / ".complete", {"report_sha256": sha256(output / "report.json")})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-config", required=True, type=Path)
    parser.add_argument("--aihub-prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--bdd-count", type=int, default=10000)
    parser.add_argument("--vkitti-count", type=int, default=9177)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    report = build(args.baseline_config, args.aihub_prepared, args.output, args.bdd_count, args.vkitti_count, args.seed)
    print(json.dumps({"complete": report["complete"], "train": report["train"], "val": report["val"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
