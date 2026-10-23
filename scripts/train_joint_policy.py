"""Train an offline multi-camera temporal waypoint policy; no vehicle transport."""
import argparse
import json

from camera_rc.joint_policy.training import load_joint_config, train_joint_policy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(json.dumps(train_joint_policy(load_joint_config(args.config)), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
