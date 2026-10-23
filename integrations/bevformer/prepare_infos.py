"""Call BEVFormer's official nuScenes temporal-info converter for one split version.

This avoids the stock CLI's additional v1.0-test conversion when preparing only
trainval. It does not modify or reimplement the upstream conversion algorithm.
"""

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

try:
    from .run import REFERENCE_COMMIT, git_state, require_file
except ImportError:  # direct `python integrations/bevformer/prepare_infos.py`
    from run import REFERENCE_COMMIT, git_state, require_file


CONVERTER_CODE = (
    "import sys; "
    "from tools.data_converter.nuscenes_converter import create_nuscenes_infos; "
    "create_nuscenes_infos(sys.argv[1], sys.argv[2], sys.argv[3], "
    "'nuscenes', version=sys.argv[4], max_sweeps=10)"
)


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--upstream", type=Path, required=True)
    cli.add_argument("--data-root", type=Path, required=True)
    cli.add_argument("--canbus-root", type=Path, required=True,
                     help="parent directory containing can_bus/, not can_bus/ itself")
    cli.add_argument("--version", choices=("v1.0-mini", "v1.0-trainval"), required=True)
    cli.add_argument("--output-dir", type=Path,
                     help="defaults to data root; must be separate for mini and full annotations")
    cli.add_argument("--allow-unpinned", action="store_true")
    cli.add_argument("--overwrite", action="store_true")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--prepare-only", action="store_true", help="default: inspect command only")
    return cli


def make_command(args):
    upstream = args.upstream.expanduser().resolve()
    root = args.data_root.expanduser().resolve()
    canbus = args.canbus_root.expanduser().resolve()
    output = (args.output_dir or root).expanduser().resolve()
    require_file(upstream / "tools/data_converter/nuscenes_converter.py", "official converter")
    state = git_state(upstream)
    if (state["commit"] != REFERENCE_COMMIT or state["dirty"] is not False) and not args.allow_unpinned:
        raise ValueError("Use the pinned clean upstream checkout or explicitly pass --allow-unpinned")
    require_file(root / args.version / "sample.json", "nuScenes sample metadata")
    for path in (canbus / "can_bus", root / "samples" / "LIDAR_TOP"):
        if not path.is_dir():
            raise ValueError("directory required by the official converter is missing: {}".format(path))
    for split in ("train", "val"):
        destination = output / ("nuscenes_infos_temporal_" + split + ".pkl")
        if destination.exists() and not args.overwrite:
            raise ValueError("annotations already exist: {} (use a new output directory)".format(destination))
    command = [sys.executable, "-c", CONVERTER_CODE, str(root), str(output), str(canbus), args.version]
    return upstream, output, command


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        upstream, output, command = make_command(args)
        print("Working directory: {}".format(upstream))
        print("Conversion command: {}".format(shlex.join(command)))
        if not args.execute:
            print("Prepared only; no conversion, data downloads, or pickle loading started.")
            return 0
        if sys.platform != "linux" or sys.version_info[:2] != (3, 8):
            raise ValueError("Execute in the separate Linux/WSL2 Python 3.8 BEVFormer environment")
        output.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["PYTHONPATH"] = str(upstream) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        return subprocess.run(command, cwd=str(upstream), env=env, check=False).returncode
    except (ValueError, OSError) as error:
        print("BEVFormer conversion error: {}".format(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
