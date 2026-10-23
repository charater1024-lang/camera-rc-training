"""Calculate declared speed/latency/optics budgets; does not simulate perception."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def positive(value, name, allow_zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")
    return float(value)


def stopping_distance(speed_kmh, reaction_s, deceleration_mps2, margin_m=0):
    speed = positive(speed_kmh, "speed", True) / 3.6
    delay = positive(reaction_s, "reaction", True)
    deceleration = positive(deceleration_mps2, "deceleration")
    margin = positive(margin_m, "margin", True)
    return {"speed_kmh": speed_kmh, "speed_mps": speed, "reaction_s": delay,
            "assumed_deceleration_mps2": deceleration, "reaction_distance_m": speed * delay,
            "braking_distance_m": speed * speed / (2 * deceleration), "margin_m": margin,
            "required_static_stopping_distance_m": speed * delay + speed * speed / (2 * deceleration) + margin}


def projected_height(width, horizontal_fov_deg, object_height_m, distance_m):
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ValueError("width must be a positive integer")
    fov = positive(horizontal_fov_deg, "fov")
    if fov >= 180:
        raise ValueError("Pinhole FOV must be less than 180 degrees")
    height, distance = positive(object_height_m, "height"), positive(distance_m, "distance")
    focal = width / (2 * math.tan(math.radians(fov) / 2))
    return {"input_width": width, "horizontal_fov_deg": fov, "object_height_m": height,
            "distance_m": distance, "focal_pixels": focal, "projected_height_pixels": focal * height / distance}


def design(config_path, output):
    path, output = Path(config_path).resolve(), Path(output).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if cfg.get("scope") != "offline_replay_and_simulation_only" or cfg.get("deployment_approved") is not False:
        raise ValueError("Experiment must remain offline research")
    stopping = [stopping_distance(speed, delay, decel, cfg["stopping_margin_m"])
                for speed, delay, decel in itertools.product(cfg["speeds_kmh"], cfg["reaction_delays_s"], cfg["assumed_decelerations_mps2"])]
    optics = cfg["optics"]
    projections = [projected_height(*values) for values in itertools.product(optics["network_input_widths"],
                   optics["horizontal_fov_deg"], optics["object_heights_m"], optics["distances_m"])]
    target_speed = positive(cfg["target_speed_kmh"], "target speed") / 3.6
    spacing = [{"capture_hz": positive(fps, "fps"), "distance_between_frames_m": target_speed / fps}
               for fps in cfg["camera_fps"]]
    output.mkdir(parents=True, exist_ok=False)
    scenario_path = output / "scenario_matrix.jsonl"
    number = 0
    with scenario_path.open("x", encoding="utf-8") as stream:
        for speed, family, condition in itertools.product(cfg["speeds_kmh"], cfg["scenario_families"], cfg["conditions"]):
            number += 1
            stream.write(json.dumps({"scenario_id": f"{family}__{condition}__{speed}kmh", "requested_speed_kmh": speed,
                "family": family, "condition": condition, "minimum_independent_seeds": cfg["simulation"]["minimum_independent_seeds_per_cell"],
                "attained_speed_verified": False, "executed": False, "deployed": False}) + "\n")
    result = {"experiment_id": cfg["experiment_id"], "config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
              "analysis_only": True, "model_performance_measured": False, "simulation_executed": False,
              "hardware_test_executed": False, "deployment_approved": False,
              "stopping_distance_cases": stopping, "optical_projection_cases": projections,
              "frame_spacing_at_target_speed": spacing,
              "target_speed_mps": target_speed, "travel_during_100ms_m": target_speed * .1,
              "current_policy_2_5s_horizon_distance_at_80kmh_m": target_speed * 2.5,
              "scenario_cells": number, "minimum_runs_before_fault_grid": number * cfg["simulation"]["minimum_independent_seeds_per_cell"],
              "scenario_manifest": str(scenario_path), "missing_required_evidence": [
                  "verified real 75..85km/h synchronized footage", "held-out independent high-speed drives",
                  "distance/relative-speed labels", "long-range small-object recall", "target-hardware exposure timestamps",
                  "actuator/brake-onset timing", "closed-loop simulator trajectories and collisions"],
              "assumption_notice": "v*t + v^2/(2*a) + margin is a constant-deceleration straight-line stationary-obstacle calculation. Delays and deceleration are assumed experiment inputs, not measured vehicle capabilities; gradients, tyre slip, moving objects and control transients are not modelled."}
    (output / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = design(args.config, args.output)
    print(json.dumps({key: result[key] for key in ("analysis_only", "scenario_cells", "minimum_runs_before_fault_grid", "travel_during_100ms_m", "missing_required_evidence")}, ensure_ascii=False, indent=2))
