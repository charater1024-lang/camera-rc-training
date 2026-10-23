"""Run a public-view balanced follow-up only after the large-data queue completes.

This one-shot queue never edits deployment models or opens a vehicle connection.
Public images remain independent images; no real eight-camera coverage is claimed.
The upstream queue and all prepared manifests must retain their recorded hashes.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import time

from run_large_data_queue import Queue, compare_reports, digest, read_json


UPSTREAM = "large_all_stage2_v1"
STAGE = "camera_role_balanced_v1"
LABEL_SPACE = "road_surface_v1"
SOURCE_ROOTS = {"public_v2": "bdd", "vkitti2_v1": "vkitti2",
                "a2d2_road_v1": "a2d2", "acdc_road_v1": "acdc"}
VALIDATION_ROOTS = {"camvid_road_v1", "kitti_road_v2"}
REAL_ROLES = {"front_main", "front_wide", "front_near", "left_forward", "right_forward",
              "left_rearward", "right_rearward", "rear"}


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _completed_report(root):
    if (root / ".incomplete").exists():
        raise ValueError(f"Incomplete preparation: {root}")
    report = read_json(root / "report.json")
    if report.get("complete") is not True or report.get("label_space_id") != LABEL_SPACE:
        raise ValueError(f"Expected complete {LABEL_SPACE} preparation: {root}")
    return report


def _manifest_path(root, value):
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest path is missing")
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if root not in path.parents:
        raise ValueError(f"Manifest escapes prepared root: {path}")
    if any((parent / ".incomplete").exists() for parent in path.parents):
        raise ValueError(f"Manifest preparation is incomplete: {path}")
    return path


def verify_upstream(project):
    """Bind completed training to its immutable recipe and original inputs."""
    project = Path(project).resolve()
    root = project / "data" / UPSTREAM
    report = _completed_report(root)
    config_path = project / "configs/local" / (UPSTREAM + ".json")
    if digest(config_path) != report.get("config_sha256"):
        raise ValueError("Upstream configuration SHA256 changed")
    config = read_json(config_path)
    train_manifest = (root / "train.jsonl").resolve()
    if (Path(config["data"]["train_manifest"]).resolve() != train_manifest
            or digest(train_manifest) != report.get("train_manifest_sha256")):
        raise ValueError("Upstream combined training manifest SHA256 changed")
    summary_path = project / "runs" / UPSTREAM / "summary.json"
    summary = read_json(summary_path)
    if summary.get("training_complete") is not True or summary.get("diagnostic") is not False:
        raise ValueError("Upstream training must be complete and non-diagnostic")
    checkpoint = (project / "runs" / UPSTREAM / "best.pt").resolve()
    if Path(summary.get("best_checkpoint", "")).resolve() != checkpoint or not checkpoint.is_file():
        raise ValueError("Upstream best checkpoint is missing or mismatched")
    checkpoint_hash = digest(checkpoint)
    comparison_path = project / "runs" / UPSTREAM / "evaluation/comparison.json"
    comparison = read_json(comparison_path)
    if (comparison.get("candidate_checkpoint", {}).get("sha256") != checkpoint_hash
            or comparison.get("deployment_approved") is not False):
        raise ValueError("Upstream checkpoint differs from its completed research evaluation")
    expected = {str((project / "data" / name).resolve()): name
                for name in set(SOURCE_ROOTS) | VALIDATION_ROOTS}
    sources, seen, all_val = {}, set(), set()
    for source in report.get("source_manifests", []):
        source_root = Path(source["root"]).resolve()
        key = str(source_root)
        if key not in expected or key in seen:
            raise ValueError("Unexpected or duplicate upstream source root")
        seen.add(key)
        source_report = _completed_report(source_root)
        if digest(source_root / "report.json") != source.get("report_sha256"):
            raise ValueError(f"Source completion report SHA256 changed: {source_root}")
        manifests = {}
        for split, entry in source.get("manifests", {}).items():
            if split not in {"train", "val", "test"}:
                raise ValueError("Unexpected source manifest split")
            path = _manifest_path(source_root, entry.get("manifest"))
            declared = source_report.get("manifests", {}).get(entry.get("report_key"))
            if not isinstance(declared, dict) or _manifest_path(source_root, declared.get("manifest")) != path:
                raise ValueError("Source completion report manifest does not match upstream")
            actual_hash = digest(path)
            if actual_hash != entry.get("sha256") or actual_hash != declared.get("sha256"):
                raise ValueError(f"Source manifest SHA256 changed: {path}")
            count = entry.get("samples")
            if isinstance(count, bool) or not isinstance(count, int) or count < 1 or count != declared.get("samples"):
                raise ValueError("Source manifest sample count is inconsistent")
            manifests[split] = str(path)
            if split == "val":
                all_val.add(str(path))
        name = expected[key]
        required = {"train", "val"} if name in SOURCE_ROOTS else {"val"}
        if not required <= manifests.keys():
            raise ValueError(f"Required source manifests are missing: {name}")
        if name in VALIDATION_ROOTS and "train" in manifests:
            raise ValueError("Regression validation roots must not supply training images")
        sources[name] = manifests
    if seen != set(expected):
        raise ValueError("Upstream source preparation reports are missing")
    val_specs = config["data"]["val_manifests"]
    val_paths = [str(Path(entry["manifest"]).resolve()) for entry in val_specs]
    if len(set(val_paths)) != len(val_paths) or set(val_paths) != all_val:
        raise ValueError("Upstream validation manifests differ from verified sources")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "summary": str(summary_path), "summary_sha256": digest(summary_path),
            "comparison": str(comparison_path), "comparison_sha256": digest(comparison_path),
            "recipe_report": str(root / "report.json"), "recipe_report_sha256": digest(root / "report.json"),
            "config": config, "sources": sources}


def validate_public_role_report(report):
    """Reject fabricated real-rig coverage, even when public views are numerous."""
    if (report.get("complete") is not True or report.get("image_masks_decoded_and_validated") is not True
            or report.get("models_validated_for_control") is not False
            or report.get("synchronized_multiview_samples_created") is not False
            or report.get("all_roles_have_real_rig_train_and_val_samples") is not False
            or not report.get("missing_real_rig_train_val")):
        raise ValueError("Public-view preparation cannot claim complete real-rig coverage or control validation")
    coverage = report.get("rig_role_coverage")
    if not isinstance(coverage, dict) or set(coverage) != REAL_ROLES:
        raise ValueError("Eight real-rig coverage gaps must be reported explicitly")
    if set(report["missing_real_rig_train_val"]) != {f"{split}/{role}" for split in ("train", "val") for role in REAL_ROLES}:
        raise ValueError("Every real-rig training and validation role remains missing")
    for splits in coverage.values():
        if not isinstance(splits, dict) or set(splits) != {"train", "val", "test"}:
            raise ValueError("Incomplete real-rig coverage report")
        for entry in splits.values():
            if entry.get("samples") != 0 or entry.get("conditions") != []:
                raise ValueError("Public sources must not count as real-rig camera samples")


class CameraRoleQueue(Queue):
    def wait_upstream(self):
        path = self.project / "runs/large_data_queue_v1/status.json"
        while True:
            self.check_stop()
            if path.is_file():
                state = read_json(path)
                status = state.get("status")
                if status in {"failed", "stopped"}:
                    raise RuntimeError(f"Upstream queue {status}; inspect {path}")
                if status == "complete":
                    return
            self.update("waiting_for_upstream_queue", waiting_for=[str(path)],
                        models_validated_for_control=False, all_real_rig_roles_trained=False)
            time.sleep(30)

    def run(self):
        self.wait_upstream()  # No new GPU work can overlap the predecessor queue.
        upstream = verify_upstream(self.project)
        save_new(self.output / "verified_upstream.json", upstream)
        prepared = self.project / "data" / STAGE
        run = self.project / "runs" / STAGE
        if prepared.exists() or run.exists():
            raise FileExistsError("Follow-up stage paths must be new; previous results are preserved")
        spec = {"epochs": 1, "samples_per_epoch": 32000,
                "rig_id": "proposed_eight_exterior_cameras_v1", "sources": [
                    {"name": kind, "kind": kind, "weight": 1.0,
                     "manifests": upstream["sources"][name]}
                    for name, kind in SOURCE_ROOTS.items()]}
        spec_path = self.output / "source_spec.json"
        save_new(spec_path, spec)
        self.command("prepare_camera_roles", ["scripts/prepare_camera_role_curriculum.py", "--spec", spec_path,
                                               "--output", prepared])
        report = read_json(prepared / "report.json")
        validate_public_role_report(report)
        config = read_json(prepared / "stage.json")
        # Retain existing independent CamVid/KITTI regressions without adding
        # unsliced duplicates of the four source-validation datasets.
        seen = {str(Path(entry["manifest"]).resolve()) for entry in config["data"]["val_manifests"]}
        for name in sorted(VALIDATION_ROOTS):
            path = upstream["sources"][name]["val"]
            if path in seen:
                raise ValueError("Regression validation manifest already appears in role slices")
            config["data"]["val_manifests"].append({"name": "regression_" + name, "manifest": path})
            seen.add(path)
        config["training"].update(output_dir=str(run), device="cuda", amp_init_scale=1024.0,
                                  log_every_steps=50)
        config_path = self.output / "training_stage.json"
        save_new(config_path, config)
        # Recheck all original manifest hashes after potentially long decoding.
        if verify_upstream(self.project) != upstream:
            raise ValueError("Upstream inputs changed during role preparation")
        self.update("prepared", stage=STAGE, config=str(config_path), config_sha256=digest(config_path),
                    source_spec_sha256=digest(spec_path), initialized_from=upstream["checkpoint"],
                    initialization_sha256=upstream["checkpoint_sha256"], training_complete=False,
                    models_validated_for_control=False, all_real_rig_roles_trained=False,
                    real_rig_coverage_report=str(prepared / "report.json"))
        self.command("camera_roles_train", ["-m", "camera_rc", "train", "--config", config_path,
                                             "--init-from", upstream["checkpoint"]])
        summary = read_json(run / "summary.json")
        if summary.get("training_complete") is not True or summary.get("diagnostic") is not False:
            raise ValueError("Follow-up training is incomplete or diagnostic")
        if summary.get("initialization", {}).get("sha256") != upstream["checkpoint_sha256"]:
            raise ValueError("Follow-up initialization differs from the verified predecessor")
        evaluation = run / "evaluation"
        evaluation.mkdir()
        for label, checkpoint in (("baseline", upstream["checkpoint"]), ("candidate", run / "best.pt")):
            self.command("camera_roles_evaluate_" + label, ["scripts/evaluate_road_safety.py", "--config", config_path,
                "--checkpoint", checkpoint, "--output", evaluation / (label + ".json"), "--device", "cuda",
                "--obstacle-root", self.datasets / "road_obstacle21/dataset_ObstacleTrack/dataset_ObstacleTrack",
                "--anomaly-root", self.datasets / "road_anomaly21/dataset_AnomalyTrack/dataset_AnomalyTrack"])
        baseline_report, candidate_report = read_json(evaluation / "baseline.json"), read_json(evaluation / "candidate.json")
        if (baseline_report["checkpoint"]["sha256"] != upstream["checkpoint_sha256"]
                or digest(upstream["checkpoint"]) != upstream["checkpoint_sha256"]):
            raise ValueError("Evaluated predecessor checkpoint changed after verification")
        comparison = compare_reports(baseline_report, candidate_report)
        comparison.update(models_validated_for_control=False, all_real_rig_roles_trained=False,
                          synchronized_multiview_training=False)
        save_new(evaluation / "comparison.json", comparison)
        self.update("complete", stage=STAGE, completed_utc=datetime.now(timezone.utc).isoformat(),
                    training_complete=True, diagnostic=False, comparison=str(evaluation / "comparison.json"),
                    checkpoint=str(run / "best.pt"), checkpoint_sha256=digest(run / "best.pt"),
                    regressions=comparison["regressions"], deployed=False,
                    models_validated_for_control=False, all_real_rig_roles_trained=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "datasets", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--timeout-hours", type=float, default=72)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_hours) or not 0 < args.timeout_hours <= 168:
        raise ValueError("timeout-hours must be in (0,168]")
    queue = CameraRoleQueue(**vars(args))
    try:
        queue.run()
    except BaseException as error:
        queue.update("stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                     error=f"{type(error).__name__}: {error}", deployed=False,
                     models_validated_for_control=False, all_real_rig_roles_trained=False)
        raise


if __name__ == "__main__":
    main()
