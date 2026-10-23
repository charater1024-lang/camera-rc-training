"""Run bounded synthetic safety-contract cases; no models, data, or hardware."""
import argparse
import hashlib
import json
from pathlib import Path

from camera_rc.driving.shadow_scenarios import run_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New JSON path; never overwritten")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve it and choose a new path")
    result = run_scenarios()
    source = Path(__file__).resolve().parents[1]
    names = ["src/camera_rc/driving/shadow_safety.py", "src/camera_rc/driving/shadow_scenarios.py",
             "scripts/check_shadow_safety.py"]
    result["source_sha256"] = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: result[key] for key in ("scope", "passed", "total", "actuation_enabled")}))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
