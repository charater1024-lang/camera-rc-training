"""Synthetic CUDA capacity probe, never an accuracy or trained-model result."""
import argparse
import gc
import json
import os
from pathlib import Path
import time

from train_yolo_semantic import identity, OFFICIAL_SHA256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--batches", nargs="+", type=int, default=[4, 8, 16])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if identity(args.weights)["sha256"] != OFFICIAL_SHA256:
        raise ValueError("Unverified pretrained weights")
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_CONFIG_DIR", str(args.output.parent / "profile_ultralytics_settings"))
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SemanticSegmentationModel
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    torch.set_num_threads(4)
    pretrained = YOLO(str(args.weights), task="semantic").model
    results = []
    for batch in args.batches:
        model = optimizer = scaler = images = target = losses = loss = None
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        row = {"batch": batch, "image_size": args.image_size, "synthetic": True}
        try:
            model = SemanticSegmentationModel(pretrained.yaml, nc=2, verbose=False).cuda().train()
            model.load(pretrained)
            model.args = get_cfg()
            model.class_weights = torch.tensor([1., 2.], device="cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            scaler = torch.amp.GradScaler("cuda", init_scale=1024)
            images = torch.rand(batch, 3, args.image_size, args.image_size, device="cuda")
            target = torch.ones(batch, args.image_size, args.image_size, device="cuda", dtype=torch.long)
            target[:, args.image_size // 2:] = 0
            durations, skipped = [], 0
            for index in range(7):
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                start = time.perf_counter()
                with torch.autocast("cuda", dtype=torch.float16):
                    losses, _ = model({"img": images, "semantic_mask": target})
                    loss = losses.sum()
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite AMP training loss")
                scaler.scale(loss).backward()
                previous = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped += scaler.get_scale() < previous
                torch.cuda.synchronize()
                if index >= 2:
                    durations.append(time.perf_counter() - start)
            row.update(status="passed" if skipped < 7 else "no_successful_optimizer_update", mean_step_seconds=sum(durations) / len(durations),
                       images_per_second=batch * len(durations) / sum(durations), amp_skipped_updates=skipped,
                       final_loss=float(loss.detach()), trainable_parameters=sum(p.numel() for p in model.parameters()),
                       peak_allocated_bytes=torch.cuda.max_memory_allocated(), peak_reserved_bytes=torch.cuda.max_memory_reserved())
        except torch.OutOfMemoryError as error:
            row.update(status="out_of_memory", error=str(error), peak_allocated_bytes=torch.cuda.max_memory_allocated())
        results.append(row)
        del model, optimizer, scaler, images, target, losses, loss
        gc.collect()
        torch.cuda.empty_cache()
        write_json(args.output, {"device": torch.cuda.get_device_name(0), "synthetic": True, "training_complete": False, "results": results})
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
