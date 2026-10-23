"""Same frozen accuracy protocol under an explicit CUDA allocator cap.

No speed measurement. The cap does not constrain CUDA context allocations or
other applications. All official cohort and ancestry checks are retained.
"""
import argparse
import sys
import torch
import evaluate_frozen_road


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--gpu-memory-gib", type=float, default=5)
    args, remaining = parser.parse_known_args()
    if not 0 < args.gpu_memory_gib <= 8 or not torch.cuda.is_available():
        raise ValueError("CUDA and a positive <=8GiB allocator cap required")
    total = torch.cuda.get_device_properties(0).total_memory
    cap = int(args.gpu_memory_gib * 1024**3)
    if cap >= total:
        raise ValueError("Physical GPU headroom required")
    torch.cuda.set_per_process_memory_fraction(cap / total, 0)
    sys.argv = [sys.argv[0], *remaining]
    evaluate_frozen_road.main()


if __name__ == "__main__":
    main()
