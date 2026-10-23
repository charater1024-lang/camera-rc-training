"""Freeze native road holdouts and optional public obstacle development subsets."""
import argparse
from pathlib import Path
from evaluation_contract import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.spec, args.output)


if __name__ == "__main__":
    main()
