"""Create future metric trajectories from timestamped, calibrated planar poses."""
import bisect
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


def prepare_rc(csv_path, output, offsets=(0.5, 1.0, 1.5, 2.0, 2.5), max_gap=0.2):
    csv_path, output = Path(csv_path).resolve(), Path(output).resolve()
    offsets = list(map(float, offsets))
    if not offsets or offsets != sorted(set(offsets)) or offsets[0] <= 0 or max_gap <= 0:
        raise ValueError("Future offsets must be unique, increasing, positive; max_gap>0")
    if output.exists():
        raise FileExistsError(output)
    groups = defaultdict(list)
    required = {"image", "group_id", "timestamp_s", "x_m", "y_m", "yaw_rad", "speed_mps", "yaw_rate_rps"}
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Required CSV columns: {sorted(required)}")
        for r in reader:
            if not r["group_id"]:
                raise ValueError("group_id required")
            for key in required - {"image", "group_id"}:
                r[key] = float(r[key])
                if not math.isfinite(r[key]):
                    raise ValueError(f"Nonfinite {key}")
            if r["speed_mps"] < 0:
                raise ValueError("Reverse driving is not supported by this baseline")
            image = Path(r["image"])
            r["image"] = str((image if image.is_absolute() else csv_path.parent / image).resolve())
            if not Path(r["image"]).is_file():
                raise FileNotFoundError(r["image"])
            groups[(r.get("dataset_id") or "own_rc", r["group_id"])].append(r)
    records, skipped = [], 0
    for (dataset_id, group), rows in groups.items():
        rows.sort(key=lambda r: r["timestamp_s"])
        times = [r["timestamp_s"] for r in rows]
        if len(times) != len(set(times)):
            raise ValueError(f"Duplicate timestamps in {group}")
        for i, row in enumerate(rows):
            target, valid = [], True
            end = bisect.bisect_left(times, row["timestamp_s"] + offsets[-1])
            # Do not label paths through a missing/unobserved portion of the drive.
            if end >= len(rows) or any(times[k + 1] - times[k] > max_gap for k in range(i, end)):
                skipped += 1
                continue
            c, s = math.cos(row["yaw_rad"]), math.sin(row["yaw_rad"])
            for delta in offsets:
                t = row["timestamp_s"] + delta
                hi = bisect.bisect_left(times, t)
                lo = max(0, hi - 1)
                if times[hi] == t:
                    lo = hi
                weight = 0.0 if lo == hi else (t - times[lo]) / (times[hi] - times[lo])
                interp = lambda key: rows[lo][key] * (1 - weight) + rows[hi][key] * weight
                dx, dy = interp("x_m") - row["x_m"], interp("y_m") - row["y_m"]
                target.append([c * dx + s * dy, -s * dx + c * dy, interp("speed_mps")])
            records.append({"image": row["image"], "group_id": group, "dataset_id": dataset_id,
                            "task": "trajectory", "timestamp_s": row["timestamp_s"],
                            "state": [row["speed_mps"], row["yaw_rate_rps"]], "target": target,
                            "future_offsets_s": offsets})
    if not records:
        raise ValueError("No complete future trajectories; check duration, timestamps and max_gap")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    return {"output": str(output), "samples": len(records), "groups": len(groups), "skipped_tail_or_gaps": skipped}
