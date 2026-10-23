"""Queue a synthetic stereo/video waypoint experiment after role pretraining.

The experiment uses two forward simulated cameras and three history frames.
Its nominal frame period is an explicit assumption, not measured exposure time.
Nothing in this queue connects to a vehicle or deploys a model.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time

from run_large_data_queue import Queue, digest, read_json


ROLE_STAGE = "camera_role_balanced_v1"
DATA_STAGE = "vkitti_joint_policy_v1"
TRAIN_STAGE = "joint_policy_vkitti_v1"
FUTURE_DT_S = [0.5, 1.0, 1.5, 2.0, 2.5]
EXPERIMENT_FLAGS = {"trained_views": 2, "temporal_history": 3, "simulator_data": True,
                    "nominal_timing_assumed": True, "all_real_rig_roles_trained": False,
                    "models_validated_for_control": False, "deployed": False}


def save_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def verify_role_predecessor(project):
    """The role queue, trained checkpoint and finished evaluation must agree."""
    project = Path(project).resolve()
    status_path = project / "runs/camera_role_queue_v1/status.json"
    state = read_json(status_path)
    if (state.get("status") != "complete" or state.get("training_complete") is not True
            or state.get("diagnostic") is not False or state.get("models_validated_for_control") is not False
            or state.get("deployed") is not False):
        raise ValueError("Role predecessor must be complete, non-diagnostic and research-only")
    run = project / "runs" / ROLE_STAGE
    checkpoint = (run / "best.pt").resolve()
    summary_path = run / "summary.json"
    summary = read_json(summary_path)
    if (summary.get("training_complete") is not True or summary.get("diagnostic") is not False
            or Path(summary.get("best_checkpoint", "")).resolve() != checkpoint
            or Path(state.get("checkpoint", "")).resolve() != checkpoint):
        raise ValueError("Role training summary is incomplete or checkpoint paths disagree")
    checkpoint_hash = digest(checkpoint)
    comparison_path = run / "evaluation/comparison.json"
    comparison = read_json(comparison_path)
    if (state.get("checkpoint_sha256") != checkpoint_hash
            or comparison.get("candidate_checkpoint", {}).get("sha256") != checkpoint_hash
            or comparison.get("deployment_approved") is not False
            or comparison.get("models_validated_for_control") is not False):
        raise ValueError("Role checkpoint SHA256 or research evaluation does not match")
    return {"checkpoint": str(checkpoint), "checkpoint_sha256": checkpoint_hash,
            "queue_status": str(status_path), "queue_status_sha256": digest(status_path),
            "summary": str(summary_path), "summary_sha256": digest(summary_path),
            "comparison": str(comparison_path), "comparison_sha256": digest(comparison_path)}


def verify_download_markers(root):
    """Require official downloader completion; the adapter checks the assets."""
    result = {}
    for asset in ("metadata", "rgb"):
        marker = root / (asset + "_complete.json")
        value = read_json(marker)
        downloaded, extracted = value.get("download", {}), value.get("extraction", {})
        checksum = downloaded.get("sha256")
        if (downloaded.get("verified") is not True or not isinstance(checksum, str)
                or not re.fullmatch(r"[a-f0-9]{64}", checksum)
                or extracted.get("asset") != asset or extracted.get("archive_sha256") != checksum
                or (root / asset / "INCOMPLETE").exists()
                or read_json(root / asset / "EXTRACTION_COMPLETE.json") != extracted):
            raise ValueError(f"Incomplete or inconsistent VKITTI {asset} verification")
        result[asset] = {"marker": str(marker), "marker_sha256": digest(marker),
                         "archive_sha256": checksum}
    return result


def prepared_manifests(root):
    if (root / ".incomplete").exists():
        raise ValueError("Joint preparation remains incomplete")
    report = read_json(root / "report.json")
    if (report.get("complete") is not True or report.get("partial_dataset", False) is not False
            or report.get("nominal_timing_assumed") is not True
            or report.get("configured_frame_period_s") != 0.1
            or report.get("synthetic_simulator_data") is not True
            or report.get("models_validated_for_control") is not False
            or report.get("history") != 3 or report.get("views") != 2
            or report.get("future_dt_s") != FUTURE_DT_S):
        raise ValueError("Joint preparation is incomplete or differs from the stated stereo/video experiment")
    entries = {}
    for split in ("train", "val", "test"):
        path = root / (split + ".jsonl")
        if not path.is_file() or path.stat().st_size == 0:
            raise ValueError(f"Missing or empty joint {split} manifest")
        entry = report.get("manifests", {}).get(split, {})
        checksum = digest(path)
        count = entry.get("samples")
        if (Path(entry.get("manifest", "")).resolve() != path.resolve() or entry.get("sha256") != checksum
                or isinstance(count, bool) or not isinstance(count, int) or count < 1):
            raise ValueError(f"Joint {split} manifest differs from its completed preparation")
        entries[split] = {"path": str(path), "sha256": checksum, "samples": count}
    return {"report": str(root / "report.json"), "report_sha256": digest(root / "report.json"),
            "manifests": entries}


def training_config(template, prepared, run, initialization):
    """Preserve the architecture template while selecting verified local inputs."""
    config = json.loads(json.dumps(template))
    data = config.get("data", {})
    if (config.get("task") != "joint_policy" or config.get("models_validated_for_control") is not False
            or config.get("diagnostic_only", False) is not False or data.get("image_size") != [192, 320]
            or data.get("expected_history") != 3 or data.get("expected_views") != 2
            or data.get("future_dt_s") != FUTURE_DT_S):
        raise ValueError("Expected non-diagnostic research joint policy with 192x320, history3, views2 and five future targets")
    config["data"].update({split + "_manifest": str(prepared / (split + ".jsonl"))
                           for split in ("train", "val", "test")})
    config.setdefault("training", {}).update(output_dir=str(run), epochs=3, batch_size=2,
                                               device="cuda", init_encoder=str(initialization))
    if any(config["training"].get(field) is not None for field in ("max_steps", "max_train_steps", "max_val_steps")):
        raise ValueError("Joint queue cannot silently use diagnostic step caps")
    return config


class JointPolicyQueue(Queue):
    def wait_role_queue(self):
        path = self.project / "runs/camera_role_queue_v1/status.json"
        while True:
            self.check_stop()
            if path.is_file():
                state = read_json(path)
                if state.get("status") in {"failed", "stopped"}:
                    raise RuntimeError(f"Role queue {state['status']}; inspect {path}")
                if state.get("status") == "complete":
                    return
            # 'trained_views' is populated only after successful training below.
            self.update("waiting_for_role_queue", waiting_for=[str(path)], planned_views=2,
                        temporal_history=3, simulator_data=True, nominal_timing_assumed=True,
                        training_complete=False, models_validated_for_control=False, deployed=False)
            time.sleep(30)

    def run(self):
        self.wait_role_queue()
        predecessor = verify_role_predecessor(self.project)
        save_new(self.output / "verified_predecessor.json", predecessor)
        vk = self.datasets / "virtual_kitti2"
        downloads = verify_download_markers(vk)
        save_new(self.output / "verified_download_markers.json", downloads)
        prepared, run = self.project / "data" / DATA_STAGE, self.project / "runs" / TRAIN_STAGE
        if prepared.exists() or run.exists():
            raise FileExistsError("Joint experiment paths must be new; previous results are preserved")
        self.command("prepare_joint_policy", ["scripts/prepare_vkitti_joint_policy.py", "--root", vk,
                    "--output", prepared, "--frame-period-s", "0.1", "--stride", "5"])
        provenance = prepared_manifests(prepared)
        save_new(self.output / "data_provenance.json", provenance)
        template_path = self.project / "configs/joint_policy_8gb.json"
        config = training_config(read_json(template_path), prepared, run, predecessor["checkpoint"])
        config_path = self.output / "training_config.json"
        save_new(config_path, config)
        # Preparation can take time. A modified predecessor must not initialize
        # the next experiment without matching its completed evaluation.
        if verify_role_predecessor(self.project) != predecessor:
            raise ValueError("Role predecessor changed during joint preparation")
        if verify_download_markers(vk) != downloads or prepared_manifests(prepared) != provenance:
            raise ValueError("Joint inputs changed before training")
        self.update("prepared", stage=TRAIN_STAGE, config=str(config_path), config_sha256=digest(config_path),
                    template_sha256=digest(template_path), initialization_sha256=predecessor["checkpoint_sha256"],
                    data_provenance=str(self.output / "data_provenance.json"), planned_views=2, temporal_history=3,
                    simulator_data=True, nominal_timing_assumed=True, training_complete=False,
                    models_validated_for_control=False, deployed=False)
        self.command("train_joint_policy", ["scripts/train_joint_policy.py", "--config", config_path])
        summary_path = run / "summary.json"
        summary = read_json(summary_path)
        checkpoint = (run / "best.pt").resolve()
        if (summary.get("training_complete") is not True or summary.get("diagnostic") is not False
                or summary.get("models_validated_for_control") is not False
                or summary.get("full_requested_epochs_completed") is not True
                or summary.get("requested_epochs") != 3 or summary.get("completed_epochs") != 3
                or summary.get("trained_views") != 2 or summary.get("history_frames") != 3
                or summary.get("test_split_evaluated") is not False or summary.get("deployed") is not False
                or Path(summary.get("best_checkpoint", "")).resolve() != checkpoint
                or not checkpoint.is_file()
                or summary.get("initialization", {}).get("sha256") != predecessor["checkpoint_sha256"]):
            raise ValueError("Joint training summary is incomplete, mismatched or incorrectly control-validated")
        if summary.get("checkpoint_sha256") != digest(checkpoint):
            raise ValueError("Joint checkpoint SHA256 differs from its completed summary")
        if prepared_manifests(prepared) != provenance:
            raise ValueError("Joint dataset manifests changed during training")
        self.update("complete", stage=TRAIN_STAGE, completed_utc=datetime.now(timezone.utc).isoformat(),
                    training_complete=True, diagnostic=False, summary=str(summary_path), summary_sha256=digest(summary_path),
                    checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), **EXPERIMENT_FLAGS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "datasets", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--timeout-hours", type=float, default=72)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_hours) or not 0 < args.timeout_hours <= 168:
        raise ValueError("timeout-hours must be in (0,168]")
    queue = JointPolicyQueue(**vars(args))
    try:
        queue.run()
    except BaseException as error:
        queue.update("stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                     error=f"{type(error).__name__}: {error}", deployed=False,
                     models_validated_for_control=False, all_real_rig_roles_trained=False)
        raise


if __name__ == "__main__":
    main()
