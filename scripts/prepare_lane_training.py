"""Prepare explicit ego-lane annotations; never start model training."""
import argparse
import json
from pathlib import Path
import sys


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from camera_rc.lane_training import prepare_lane_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", nargs=2, type=int, default=[180, 320], metavar=("HEIGHT", "WIDTH"))
    args = parser.parse_args(argv)
    report = prepare_lane_dataset(args.annotations, args.output, args.image_size)
    print(json.dumps({key: report[key] for key in ("output", "samples", "label_space_id", "diagnostic_only",
                                                 "train_config", "training_executed", "model_adopted")},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
