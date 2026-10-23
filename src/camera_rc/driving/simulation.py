"""Deterministic ideal-sensor bicycle simulation for software integration tests."""
import copy
from dataclasses import asdict
import json
import math
from pathlib import Path
import numpy as np
from .config import DEFAULTS
from .geometry import GroundProjector
from .planner import LocalPlanner
from .runtime import ControlLoop
from .transport import DryRunTransport
from .types import Command, Perception, VehicleState


class SimulatedVehicle(DryRunTransport):
    def __init__(self, wheelbase=0.26):
        super().__init__()
        self.x, self.y, self.yaw, self.speed = 0.0, 0.0, 0.0, 0.0
        self.wheelbase = wheelbase
        self.last_command = Command()

    def advance(self, dt):
        cmd = self.last_command
        acceleration = (1.5 * cmd.throttle if cmd.enabled else 0.0) - 1.5 * cmd.brake - 0.25 * self.speed
        self.speed = max(0.0, self.speed + acceleration * dt)
        yaw_rate = self.speed * math.tan(cmd.steering_rad) / self.wheelbase
        self.yaw += yaw_rate * dt
        self.x += self.speed * math.cos(self.yaw) * dt
        self.y += self.speed * math.sin(self.yaw) * dt

    def read_state(self, now):
        yaw_rate = self.speed * math.tan(self.last_command.steering_rad) / self.wheelbase
        return VehicleState(now, self.speed, yaw_rate, True, False)


def run_simulation(output, steps=250):
    if steps < 120:
        raise ValueError("At least 120 simulation steps are needed to exercise the fault scenario")
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg = copy.deepcopy(DEFAULTS)
    calibration = {"calibrated": True, "rectified": True, "image_size": [192, 320],
                   "image_to_ground": [[0, -0.012, 2.0], [-0.00625, 0, 1.0], [0, 0, 1]],
                   "supported_range": {"min_forward_m": -0.2, "max_forward_m": 1.9, "max_abs_lateral_m": 0.9}}
    projector = GroundProjector(calibration)
    vv, uu = np.mgrid[:192, :320]
    ground = projector.pixel_to_ground(np.column_stack((uu.ravel(), vv.ravel()))).reshape(192, 320, 2)
    results = {}
    for scenario in ("clear_straight", "obstacle_stop", "perception_loss"):
        vehicle = SimulatedVehicle()
        controller = ControlLoop(cfg, vehicle, output)
        planner = LocalPlanner(cfg["planner"], calibration)
        collision, enabled_ticks = False, 0
        trace = output / f"{scenario}.jsonl"
        with trace.open("w", encoding="utf-8") as stream:
            for step in range(steps):
                dt, now = 0.05, 100 + step * 0.05
                vehicle.advance(dt)
                gx, gy = ground[..., 0], ground[..., 1]
                world_x = vehicle.x + math.cos(vehicle.yaw) * gx - math.sin(vehicle.yaw) * gy
                world_y = vehicle.y + math.sin(vehicle.yaw) * gx + math.cos(vehicle.yaw) * gy
                road = np.abs(world_y) < 0.55
                if scenario == "obstacle_stop":
                    road &= world_x < 2.0
                    collision |= vehicle.x + 0.305 >= 2.0
                if scenario != "perception_loss" or step < 90:
                    perception = Perception(road.astype(np.float32))
                    plan = planner.plan(perception, vehicle.speed)
                    controller.publish(plan, now)
                tick = controller.step(now)
                enabled_ticks += int(tick["command"]["enabled"])
                stream.write(json.dumps({"step": step, "x_m": vehicle.x, "speed_mps": vehicle.speed, **tick}) + "\n")
        vehicle.send(Command(reason="simulation_end"), 100 + steps * 0.05)
        vehicle.close()
        results[scenario] = {"progress_m": vehicle.x, "final_speed_mps": vehicle.speed,
                             "collision": bool(collision), "enabled_ticks": enabled_ticks,
                             "latched_fault": controller.latched_fault, "control_reasons": dict(controller.counts)}
    checks = {"clear_path_moves": results["clear_straight"]["progress_m"] > 0.5,
              "obstacle_stops_before_contact": not results["obstacle_stop"]["collision"] and results["obstacle_stop"]["final_speed_mps"] < 0.01,
              "stale_perception_stops_and_latches": results["perception_loss"]["final_speed_mps"] < 0.01 and results["perception_loss"]["latched_fault"] is not None}
    report = {"diagnostic_only": True, "success": all(checks.values()), "physical_output": False,
              "checks": checks, "scenarios": results,
              "note": "Ideal segmentation from synthetic ground truth and simplified bicycle dynamics. Verifies software integration only; not learned-model accuracy or real braking capability."}
    (output / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
