"""One-shot local preparation/training queue for verified large public datasets.

This is not a recurring service. Already-started downloaders own the network
transfers. This process waits for their completion markers, prepares data, runs
full epochs, and evaluates stable checkpoints. It never edits runtime profiles
or deployment models. A STOP_REQUESTED file in its output folder cancels work.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024**2), b""):
            hasher.update(block)
    return hasher.hexdigest()


def _count(value):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Pixel/image counts must be nonnegative integers")
    return value


def _rate(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Rates must be finite values in [0,1]")
    return value


def _native_denominators(report):
    matrix = report["confusion_matrix"]
    if not isinstance(matrix, list) or len(matrix) != 2 or any(not isinstance(row, list) or len(row) != 2 for row in matrix):
        raise ValueError("Expected binary confusion matrix")
    totals = tuple(sum(_count(value) for value in row) for row in matrix)
    if sum(totals) != _count(report["valid_pixels"]):
        raise ValueError("Native valid pixel total differs from confusion matrix")
    return totals


def _public_denominators(report, threshold):
    rows = report["per_image"]
    if len(rows) != _count(report["aggregate"]["images"]) or not rows:
        raise ValueError("Public image counts differ")
    ids = [row["frame_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate public regression image")
    errors, total = 0, 0
    denominators = []
    for row in rows:
        decision = row["decisions"]["runtime_road"]
        pixels = _count(row["labelled_obstacle_pixels"])
        wrong = _count(decision["obstacle_pixels_predicted_road"])
        if (decision["threshold"], decision["comparison"]) != (threshold, ">=") or decision["labelled_obstacle_pixels"] != pixels or wrong > pixels:
            raise ValueError("Public per-image decision or denominator mismatch")
        rate = decision["obstacle_as_road_rate"]
        if (pixels and not math.isclose(_rate(rate), wrong / pixels, rel_tol=1e-12, abs_tol=1e-12)) or (not pixels and rate is not None):
            raise ValueError("Public per-image error rate mismatch")
        total += pixels
        errors += wrong
        denominators.append(pixels)
    aggregate = report["aggregate"]["decisions"]["runtime_road"]
    if not total or _count(aggregate["labelled_obstacle_pixels"]) != total or _count(aggregate["obstacle_pixels_predicted_road"]) != errors:
        raise ValueError("Public aggregate pixel counts differ")
    if (aggregate["threshold"], aggregate["comparison"]) != (threshold, ">=") or not math.isclose(_rate(aggregate["micro_obstacle_as_road_rate"]), errors / total, rel_tol=1e-12, abs_tol=1e-12):
        raise ValueError("Public aggregate rate or threshold differs")
    return denominators


def compare_reports(baseline, candidate):
    """Reject incomparable protocols and expose regressions, never grant control."""
    for key in ("config", "validation_sources", "input_size", "label_space_id", "runtime_threshold"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Evaluation protocol changed: {key}")
    if baseline["checkpoint_diagnostic"] or candidate["checkpoint_diagnostic"]:
        raise ValueError("Diagnostic checkpoints cannot be compared as full runs")
    if set(baseline["native_sources"]) != set(candidate["native_sources"]):
        raise ValueError("Native source sets differ")
    comparisons, regressions = {}, []
    for name, old in baseline["native_sources"].items():
        new = candidate["native_sources"][name]
        if _native_denominators(old) != _native_denominators(new) or old["partial_validation"] or new["partial_validation"]:
            raise ValueError("Native validation denominators differ or are partial")
        before, after = old["road_safety_pixels"]["false_road_rate"], new["road_safety_pixels"]["false_road_rate"]
        _rate(before)
        _rate(after)
        for report, rate in ((old, before), (new, after)):
            denominator = sum(report["confusion_matrix"][1])
            if not denominator or not math.isclose(rate, report["confusion_matrix"][1][0] / denominator, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("Native nonroad error rate inconsistent with confusion matrix")
        comparisons[name] = {"road_iou": [old["class_iou"][0], new["class_iou"][0]],
                             "nonroad_as_road_rate": [before, after]}
        if after > before:
            regressions.append(name + ":nonroad_as_road")
    public_old, public_new = baseline["public_obstacle_anomaly_subsets"], candidate["public_obstacle_anomaly_subsets"]
    if set(public_old) != {"obstacle", "anomaly"} or set(public_new) != set(public_old):
        raise ValueError("Both public regression subsets are required")
    for name, old in public_old.items():
        new = public_new[name]
        if _public_denominators(old, baseline["runtime_threshold"]) != _public_denominators(new, candidate["runtime_threshold"]):
            raise ValueError("Public per-image ground truth denominators differ")
        identities = lambda report: [(row["frame_id"], row["image"], row["ground_truth"]) for row in report["per_image"]]
        if identities(old) != identities(new):
            raise ValueError("Public regression image/GT identities changed")
        a, b = (report["aggregate"]["decisions"]["runtime_road"] for report in (old, new))
        if (a["threshold"], a["comparison"], a["labelled_obstacle_pixels"]) != (b["threshold"], b["comparison"], b["labelled_obstacle_pixels"]):
            raise ValueError("Public regression denominator or decision changed")
        before, after = a["micro_obstacle_as_road_rate"], b["micro_obstacle_as_road_rate"]
        comparisons[name] = {"labelled_obstacle_as_road_rate": [before, after],
                             "images": new["aggregate"]["images"]}
        if after > before:
            regressions.append(name + ":obstacle_as_road")
    return {"compared": comparisons, "regressions": regressions,
            "decision": "research_only_do_not_deploy", "deployment_approved": False,
            "baseline_checkpoint": baseline["checkpoint"], "candidate_checkpoint": candidate["checkpoint"],
            "notice": "Pixel metrics are not collision probabilities or proof of driving safety. Public obstacle subsets are reused development regression data, not fresh blind tests.",
            "native_identity_limit": "Native manifests and per-class GT counts are compared, but the evaluator does not hash every native image/mask byte. Input files must remain immutable during evaluation."}


class Queue:
    def __init__(self, project, datasets, output, timeout_hours=72):
        self.project, self.datasets, self.output = map(lambda p: Path(p).resolve(), (project, datasets, output))
        self.output.mkdir(parents=True, exist_ok=False)
        self.deadline = time.monotonic() + timeout_hours * 3600
        self.state = {"started_utc": datetime.now(timezone.utc).isoformat(), "pid": __import__("os").getpid(),
                      "status": "starting", "completed_steps": [], "deployed": False,
                      "stop_file": str(self.output / "STOP_REQUESTED")}
        self.update("starting")

    def update(self, status, **values):
        self.state.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **values)
        temporary = self.output / "status.json.tmp"
        temporary.write_text(json.dumps(self.state, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.output / "status.json")
        print(json.dumps({"status": status, **values}, ensure_ascii=False), flush=True)

    def check_stop(self):
        if (self.output / "STOP_REQUESTED").exists():
            raise InterruptedError("Stopped by STOP_REQUESTED file")
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Queue time budget exceeded; existing downloads and results preserved")

    def wait_markers(self, markers):
        while True:
            self.check_stop()
            missing = [str(path) for path in markers if not path.is_file()]
            if not missing:
                for path in markers:
                    read_json(path)  # Incomplete JSON must not advance a stage.
                return
            self.update("waiting_for_verified_downloads", waiting_for=missing)
            time.sleep(30)

    def command(self, name, arguments):
        self.check_stop()
        logfile = self.output / (name + ".log")
        command = [sys.executable, "-X", "utf8", "-B", "-u", *map(str, arguments)]
        self.update("running", step=name, command=command, log=str(logfile), waiting_for=[])
        with logfile.open("x", encoding="utf-8") as log:
            child = subprocess.Popen(command, cwd=self.project, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    self.check_stop()
                    time.sleep(2)
            except BaseException:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"], capture_output=True, check=False)
                else:
                    child.terminate()
                child.wait(timeout=30)
                raise
        if child.returncode:
            raise RuntimeError(f"{name} failed with exit {child.returncode}; inspect {logfile}")
        self.state["completed_steps"].append(name)
        self.update("step_complete", step=name)

    def prepared(self, root):
        if not root.exists():
            return False
        report = root / "report.json"
        if (root / ".incomplete").exists() or not report.exists() or read_json(report).get("complete") is not True:
            raise ValueError(f"Incomplete preparation preserved; investigate before continuing: {root}")
        return True

    def stage(self, name, sources):
        baseline = self.project / "runs/bdd100k_2018_baseline_v1/best.pt"
        config = self.project / "configs/local" / (name + ".json")
        arguments = ["scripts/prepare_large_data_stage.py", "--project", self.project, "--name", name,
                     "--init-from", baseline, "--epochs", "1"]
        for source in sources:
            arguments += ["--source-root", source]
        for root in ("camvid_road_v1", "kitti_road_v2"):
            arguments += ["--validation-root", self.project / "data" / root]
        self.command(name + "_recipe", arguments)
        self.command(name + "_train", ["-m", "camera_rc", "train", "--config", config])
        run = self.project / "runs" / name
        summary = read_json(run / "summary.json")
        if summary.get("training_complete") is not True or summary.get("diagnostic") is not False:
            raise ValueError("Incomplete or diagnostic run must not advance")
        evaluation = run / "evaluation"
        evaluation.mkdir()
        for label, checkpoint in (("baseline", baseline), ("candidate", run / "best.pt")):
            self.command(name + "_evaluate_" + label, ["scripts/evaluate_road_safety.py", "--config", config,
                "--checkpoint", checkpoint, "--output", evaluation / (label + ".json"), "--device", "cuda",
                "--obstacle-root", self.datasets / "road_obstacle21/dataset_ObstacleTrack/dataset_ObstacleTrack",
                "--anomaly-root", self.datasets / "road_anomaly21/dataset_AnomalyTrack/dataset_AnomalyTrack"])
        comparison = compare_reports(read_json(evaluation / "baseline.json"), read_json(evaluation / "candidate.json"))
        (evaluation / "comparison.json").write_text(json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8")
        self.update("stage_complete", stage=name, comparison=str(evaluation / "comparison.json"),
                    regressions=comparison["regressions"])

    def run(self):
        vk = self.datasets / "virtual_kitti2"
        prepared_vk = self.project / "data/vkitti2_v1"
        self.wait_markers([vk / (asset + "_complete.json") for asset in ("semantic", "metadata", "rgb")])
        if not self.prepared(prepared_vk):
            self.command("prepare_vkitti2", ["scripts/prepare_vkitti2.py", "--rgb-root", vk / "rgb",
                "--segmentation-root", vk / "semantic", "--metadata-root", vk / "metadata", "--output", prepared_vk,
                "--related-val-manifest", self.project / "data/kitti_road_v2/val.jsonl"])
        self.stage("large_vkitti_stage1_v1", [prepared_vk])
        acdc, a2d2 = self.datasets / "acdc", self.datasets / "a2d2_semantic_v1"
        self.wait_markers([acdc / "DOWNLOAD_COMPLETE.json", a2d2 / "DOWNLOAD_COMPLETE.json"])
        prepared_acdc, prepared_a2d2 = self.project / "data/acdc_road_v1", self.project / "data/a2d2_road_v1"
        if not self.prepared(prepared_acdc):
            self.command("prepare_acdc", ["scripts/prepare_acdc.py", "--image-root", acdc / "rgb_anon_trainvaltest/rgb_anon",
                "--gt-root", acdc / "gt_trainval/gt", "--output", prepared_acdc])
        if not self.prepared(prepared_a2d2):
            self.command("prepare_a2d2", ["scripts/prepare_a2d2.py", "--root", a2d2, "--output", prepared_a2d2])
        self.stage("large_all_stage2_v1", [prepared_vk, prepared_acdc, prepared_a2d2])
        self.update("complete", completed_utc=datetime.now(timezone.utc).isoformat())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("project", "datasets", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--timeout-hours", type=float, default=72)
    args = parser.parse_args()
    if not math.isfinite(args.timeout_hours) or not 0 < args.timeout_hours <= 168:
        raise ValueError("timeout-hours must be in (0,168]")
    queue = Queue(**vars(args))
    try:
        queue.run()
    except BaseException as error:
        queue.update("stopped" if isinstance(error, (InterruptedError, KeyboardInterrupt)) else "failed",
                     error=f"{type(error).__name__}: {error}")
        raise


if __name__ == "__main__":
    main()
