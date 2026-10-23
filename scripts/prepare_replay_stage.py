"""Create a reproducible new training stage with all new images and BDD replay."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

from camera_rc.config import load_config
from camera_rc.data import read_manifest, validate_splits


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--name", required=True)
    parser.add_argument("--source-root", required=True, type=Path, action="append")
    parser.add_argument("--init-from", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--replay-count", type=int, default=10000)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z0-9_]+", args.name) or args.epochs < 1 or args.replay_count < 1:
        raise ValueError("Invalid stage name or size")
    project = args.project.resolve()
    checkpoint = args.init_from.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    config = load_config(project / "configs/local/bdd2018_baseline_v1.json")
    base_manifest = Path(config["data"]["train_manifest"])
    bdd = read_manifest(base_manifest)
    if args.replay_count > len(bdd):
        raise ValueError("Replay count exceeds available BDD training images")
    # Stable selection independent of Python hash randomization and input order.
    bdd.sort(key=lambda row: hashlib.sha256(("seed42:" + row["sample_id"]).encode()).hexdigest())
    rows = bdd[:args.replay_count]
    held_out = read_manifest(config["data"]["val_manifests"][0]["manifest"])
    sources = [{"manifest": str(base_manifest), "sha256": sha(base_manifest),
                "available": len(bdd), "selected": len(rows)}]
    for source in args.source_root:
        source = source.resolve()
        train, val = source / "train.jsonl", source / "val.jsonl"
        rows.extend(read_manifest(train))
        held_out.extend(read_manifest(val))
        test = source / "test.jsonl"
        if test.is_file():
            held_out.extend(read_manifest(test))  # Split leakage guard only; not scored or tuned.
        config["data"]["val_manifests"].append({"name": source.name, "manifest": str(val)})
        sources.append({"manifest": str(train), "sha256": sha(train),
                        "validation_sha256": sha(val), "reserved_test_sha256": sha(test) if test.is_file() else None})
    validate_splits(rows, held_out)
    if any(row.get("label_space_id") != "road_surface_v1" for row in rows + held_out):
        raise ValueError("All stage inputs must use explicitly remapped road_surface_v1")
    output = project / "data" / args.name
    config_path = project / "configs/local" / (args.name + ".json")
    if output.exists() or config_path.exists() or (project / "runs" / args.name).exists():
        raise FileExistsError("Stage name must be new; existing results are immutable")
    output.mkdir(parents=True)
    manifest = output / "train.jsonl"
    with manifest.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    config["data"]["train_manifest"] = str(manifest)
    config["training"].update(output_dir=str(project / "runs" / args.name), epochs=args.epochs,
                               lr=0.0002, init_from=str(checkpoint))
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    report = {"stage": args.name, "train_counts": dict(Counter(row["dataset_id"] for row in rows)),
              "replay_selection": "lowest SHA256(seed42:sample_id), BDD training only",
              "source_manifests": sources, "train_manifest_sha256": sha(manifest),
              "initial_checkpoint": str(checkpoint), "initial_checkpoint_sha256": sha(checkpoint),
              "epochs": args.epochs, "all_stage_images_once_per_epoch": True,
              "augmentation": "none", "split_leakage_checked_against_validation_and_reserved_test": True,
              "config": str(config_path), "deployed": False}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
