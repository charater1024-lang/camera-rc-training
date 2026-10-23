"""Bounded CUDA capacity diagnostic on local AIHub train samples; never saves weights.

Run batch4 and batch8 in separate invocations/processes and fresh output dirs.
This cannot establish coexistence safety, sustained throughput or model accuracy.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import copy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from types import SimpleNamespace

GIB = 1024**3
CAP_GIB = 5


def validate_options(batch, steps, samples, headroom_gib, maximum_utilization, wait_seconds):
    if type(batch) is not int or batch not in (4, 8):
        raise ValueError("Use diagnostic batch 4 or 8 in separate fresh processes")
    if type(steps) is not int or not 2 <= steps <= 8:
        raise ValueError("Use 2..8 bounded optimizer attempts")
    if type(samples) is not int or samples != 16:
        raise ValueError("Freeze exactly 16 AIHub training samples")
    if not all(math.isfinite(v) for v in (headroom_gib, maximum_utilization, wait_seconds)):
        raise ValueError("Finite capacity limits required")
    if not 2 <= headroom_gib <= 8 or not 0 <= maximum_utilization <= 95 or not 0 <= wait_seconds <= 60:
        raise ValueError("Require >=2GiB headroom, <=95% prelaunch utilization threshold, <=60s bounded wait")


def budget_ready(sample, reserved_bytes, headroom_gib=2, maximum_utilization=20):
    remaining = max(0, CAP_GIB * GIB - reserved_bytes)
    return (sample["free_mib"] * 1024**2 - remaining >= headroom_gib * GIB
            and sample["utilization_percent"] <= maximum_utilization)


def apply_allocator_cap(torch, device):
    if torch.cuda.memory_allocated(device) or torch.cuda.memory_reserved(device):
        raise RuntimeError("Capacity probe requires a fresh process without CUDA tensor allocations")
    total = torch.cuda.get_device_properties(device).total_memory
    if total < CAP_GIB * GIB:
        raise RuntimeError("GPU total memory is below the fixed 5GiB diagnostic cap")
    torch.cuda.set_per_process_memory_fraction(CAP_GIB * GIB / total, device)
    return total


def select_rows(rows):
    buckets = defaultdict(list)
    for row in rows:
        if row.get("dataset_id") in {"aihub_13", "aihub_14", "aihub_15"}:
            if row.get("split") != "train" or row.get("source_split") != "Training":
                raise ValueError("AIHub probe input must come only from the official training split")
            if row.get("label_space_id") != "road_surface_v1" or row.get("task") != "segmentation":
                raise ValueError("Canonical road0/nonroad1/ignore255 semantic samples required")
            buckets[row["dataset_id"]].append(row)
    if set(buckets) != {"aihub_13", "aihub_14", "aihub_15"} or min(map(len,buckets.values())) < 6:
        raise ValueError("Need at least six eligible training rows from each AIHub source")
    for values in buckets.values():
        values.sort(key=lambda r: hashlib.sha256(("42:" + r["dataset_id"] + ":" + r["sample_id"]).encode()).hexdigest())
    selected = [buckets[ds][index] for index in range(6) for ds in sorted(buckets)][:16]
    if len({r["image"] for r in selected}) != 16:
        raise ValueError("Duplicate selected probe image")
    return selected


def validate_mask_values(mask):
    import numpy as np
    if mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer):
        raise ValueError("Mask must be two-dimensional integer class IDs")
    values = set(np.unique(mask).tolist())
    if not values <= {0, 1, 255} or not values & {0, 1}:
        raise ValueError("Mask must contain supervised road0/nonroad1 and optional ignore255")


def write(path, value):
    pending = path.with_name(path.name + ".pending")
    with pending.open("x",encoding="utf-8") as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write("\n")
    pending.replace(path)


def gpu_sample(device):
    result = subprocess.run(["nvidia-smi",f"--id={device}","--query-gpu=memory.free,memory.used,memory.total,utilization.gpu",
                             "--format=csv,noheader,nounits"],capture_output=True,text=True,timeout=15,check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    values = [float(v.strip()) for v in lines[0].split(",")] if len(lines)==1 else []
    if len(values)!=4 or any(not math.isfinite(v) or v<0 for v in values):
        raise RuntimeError("Invalid nvidia-smi GPU capacity observation")
    return dict(zip(("free_mib","used_mib","total_mib","utilization_percent"),values),observed_utc=datetime.now(timezone.utc).isoformat())


def run(args):
    validate_options(args.batch,args.steps,16,args.headroom_gib,args.maximum_utilization,args.wait_seconds)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True,exist_ok=False)
    report_path = args.output/"capacity_report.json"
    report = {"schema_version":1,"diagnostic":True,"training_complete":False,"deployment_approved":False,
              "status":"preparing_cpu","batch":args.batch,"input_size":[384,384],"allocator_cap_gib":CAP_GIB,
              "minimum_observed_headroom_gib":args.headroom_gib,"optimizer_attempts":0,"successful_updates":0,
              "skipped_updates":0,"steps":[],"checkpoints_saved":False,"started_utc":datetime.now(timezone.utc).isoformat(),
              "limits":["Sixteen real training samples and a few updates measure capacity only, never accuracy or completed training.",
                        "The 5GiB limit caps PyTorch caching allocations, not CUDA context, libraries or total process VRAM.",
                        "Global utilization is gated before CUDA allocation only; subsequent steps check memory headroom, avoiding self-utilization false aborts.",
                        "Step times include shared-GPU contention and observation overhead; they are not deployment latency or isolated throughput.",
                        "Guards observe free memory before/after steps, cannot reserve memory against other processes.",
                        "Semantic trainer validation uses batch equal to training batch; optional2x case is extra stress only.",
                        "Uses actual SemanticDataset augmentation/loss, EMA, fused AdamW grouping and clipping, but no epoch schedule/warmup/checkpoint serialization/fullvalidation.",
                        "Validation-shaped batches reuse training images solely for memory stress; no validation metric or benchmark accuracy is reported."]}
    write(report_path,report)
    torch = None
    started = time.perf_counter()
    try:
        from train_yolo_semantic import identity, verify_export, verify_initialization, validate_finetune_model
        dataset, config_id = verify_export(args.dataset,args.config)
        parent = json.loads(args.init_training_report.read_text(encoding="utf-8-sig"))
        weights, initialization = verify_initialization(parent["checkpoints"]["best"]["path"],args.init_training_report)
        provenance = json.loads(Path(dataset["provenance_path"]).read_text(encoding="utf-8"))
        manifest = provenance["manifest_outputs"]["train"]
        if identity(manifest["path"])["sha256"] != manifest["sha256"]:
            raise ValueError("Changed training export manifest")
        rows = select_rows(json.loads(line) for line in Path(manifest["path"]).open(encoding="utf-8") if line.strip())
        report.update(initialization=initialization,dataset=dataset,source_config=config_id,inputs=[])
        import numpy as np
        from PIL import Image
        for index,row in enumerate(rows):
            image_id,target_id = identity(row["image"]),identity(row["target"])
            if image_id["sha256"]!=row["image_sha256"] or target_id["sha256"]!=row["target_sha256"]:
                raise ValueError("Probe image/mask differs from frozen export")
            with Image.open(row["image"]) as image, Image.open(row["target"]) as mask:
                if image.size != mask.size: raise ValueError("Probe image/mask dimensions differ")
                validate_mask_values(np.asarray(mask))
                image.convert("RGB").load()
            image_out=args.output/"images"/"train"/(f"{index:02}"+Path(row["image"]).suffix)
            mask_out=args.output/"masks"/"train"/(f"{index:02}.png")
            for source,target in ((row["image"],image_out),(row["target"],mask_out)):
                target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
            if identity(image_out)["sha256"]!=image_id["sha256"] or identity(mask_out)["sha256"]!=target_id["sha256"]:
                raise ValueError("Probe input copy mismatch")
            report["inputs"].append({"dataset_id":row["dataset_id"],"sample_id":row["sample_id"],"group_id":row["group_id"],
                                     "image":image_id,"target":target_id,"copy_image":str(image_out),"copy_target":str(mask_out)})
        # Package settings/caches remain inside this diagnostic output; no detector download or AMP auto-check is used.
        os.environ.update(YOLO_CONFIG_DIR=str(args.output/"ultralytics_settings"),YOLO_OFFLINE="true",YOLO_AUTOINSTALL="false",
                          NO_ALBUMENTATIONS_UPDATE="1",MPLCONFIGDIR=str(args.output/"matplotlib"))
        import torch
        import psutil
        import ultralytics
        from ultralytics import YOLO,settings
        from ultralytics.cfg import get_cfg
        from ultralytics.data.dataset import SemanticDataset
        from ultralytics.engine.trainer import BaseTrainer
        from ultralytics.models.yolo.detect.train import DetectionTrainer
        from ultralytics.utils.torch_utils import ModelEMA,init_seeds
        settings.update({k:False for k in ("sync","hub","wandb","mlflow","clearml","comet","dvc","neptune","raytune","tensorboard") if k in settings})
        process=psutil.Process()
        torch.set_num_threads(4)
        dev=torch.device("cuda",args.device)

        def guard(phase, *, check_utilization=False):
            limit=time.monotonic()+args.wait_seconds
            while True:
                sample=gpu_sample(args.device)
                reserved=torch.cuda.memory_reserved(args.device) if torch.cuda.is_initialized() else 0
                if budget_ready(sample,reserved,args.headroom_gib,args.maximum_utilization if check_utilization else 100): return sample
                report.update(status="waiting_for_capacity",waiting_phase=phase,last_gpu_sample=sample)
                write(report_path,report)
                if time.monotonic()>=limit: raise RuntimeError(f"GPU headroom/utilization guard aborted before {phase}; other processes untouched")
                time.sleep(2)

        report["gpu_before"]=guard("CUDA allocation",check_utilization=True)
        # get_device_properties may initialize the driver context; no model/input CUDA tensors exist yet.
        total=apply_allocator_cap(torch,args.device)
        report["cap_applied_before_model_or_input_cuda_allocation"]=True
        report["environment"]={"torch":torch.__version__,"ultralytics":ultralytics.__version__,"device":torch.cuda.get_device_name(args.device),"total_vram_bytes":total}
        init_seeds(42,deterministic=True)
        hyp=get_cfg(overrides={"task":"semantic","imgsz":384,"batch":args.batch,"multi_scale":0.0,"seed":42,"deterministic":True})
        data={"names":{0:"road_surface",1:"not_road_surface"},"nc":2,"channels":3,"masks_dir":"masks"}
        train_data=SemanticDataset(img_path=str(args.output/"images/train"),imgsz=384,batch_size=args.batch,stride=32,
                                   rect=False,augment=True,cache=False,hyp=copy(hyp),data=data,task="semantic")
        val_data=SemanticDataset(img_path=str(args.output/"images/train"),imgsz=384,batch_size=args.batch,stride=32,
                                 rect=False,augment=False,cache=False,hyp=copy(hyp),data=data,task="semantic")
        if len(train_data)!=16 or len(val_data)!=16: raise ValueError("Dataset loader excluded probe samples")
        wrapper=YOLO(weights["path"],task="semantic")
        model=validate_finetune_model(wrapper).float()
        for p in model.parameters(): p.requires_grad_(True)
        model.args=hyp
        model.criterion=None
        guard("model and EMA construction")
        model=model.to(dev).train()
        model.class_weights=torch.tensor([1.,2.],device=dev)
        ema=ModelEMA(model)
        optimizer=BaseTrainer.build_optimizer(SimpleNamespace(args=hyp),model,name="AdamW",lr=0.0001,
                                             momentum=hyp.momentum,decay=hyp.weight_decay)
        scaler=torch.amp.GradScaler("cuda",init_scale=1024.)
        report.update(status="probing",parameters=sum(p.numel() for p in model.parameters()),amp=True,amp_init_scale=1024,
                      ema_included=True,optimizer="Installed BaseTrainer grouped fused AdamW",train_augmentation=str(train_data.transforms),
                      validation_shape_policy="Square384x384 instead of rectangular validation: conservative memory stress")
        write(report_path,report)

        def batch_from(ds,number,offset=0):
            return SemanticDataset.collate_fn([ds[(offset+i)%len(ds)] for i in range(number)])

        def snapshot(phase,begin,before,**extra):
            torch.cuda.synchronize(dev)
            after=gpu_sample(args.device)
            item={"phase":phase,"seconds":time.perf_counter()-begin,"allocated_bytes":torch.cuda.memory_allocated(dev),
                  "reserved_bytes":torch.cuda.memory_reserved(dev),"peak_allocated_bytes":torch.cuda.max_memory_allocated(dev),
                  "peak_reserved_bytes":torch.cuda.max_memory_reserved(dev),"rss_bytes":process.memory_info().rss,
                  "gpu_before":before,"gpu_after":after,**extra}
            report["steps"].append(item);write(report_path,report)
            print(json.dumps(item,ensure_ascii=False),flush=True)
            if after["free_mib"]*1024**2 < args.headroom_gib*GIB:
                raise RuntimeError("Observed GPU free memory dropped below required headroom; probe aborts, other processes untouched")

        for index in range(args.steps):
            batch=batch_from(train_data,args.batch,index*args.batch)
            before=guard(f"optimizer attempt{index+1}")
            torch.cuda.reset_peak_memory_stats(dev);begin=time.perf_counter()
            batch=DetectionTrainer.preprocess_batch(SimpleNamespace(device=dev,args=hyp,stride=32),batch)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda",dtype=torch.float16): loss,items=model(batch)
            if not torch.isfinite(loss).all(): raise ValueError("Nonfinite probe loss")
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),10.)
            scale_before=scaler.get_scale();scaler.step(optimizer);scaler.update()
            skipped=scaler.get_scale()<scale_before
            report["optimizer_attempts"]+=1
            report["skipped_updates"]+=int(skipped);report["successful_updates"]+=int(not skipped)
            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad(): ema.update(model)
            snapshot("train_step",begin,before,attempt=index+1,loss=float(loss.detach().cpu()),
                     losses={k:float(v.cpu()) for k,v in items.items()},successful_update=not skipped,
                     gradient_norm=float(norm.cpu()) if torch.isfinite(norm) else None,amp_scale=scaler.get_scale())
            del batch,loss,items,norm
        for multiplier in ((1,2) if args.extra_double_validation else (1,)):
            count=args.batch*multiplier
            batch=batch_from(val_data,count)
            before=guard(f"validation-shaped stress batch{count}")
            torch.cuda.reset_peak_memory_stats(dev);begin=time.perf_counter()
            batch=DetectionTrainer.preprocess_batch(SimpleNamespace(device=dev,args=hyp,stride=32),batch)
            batch["semantic_mask"]=batch["semantic_mask"].to(torch.int32)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.float16):
                preds=ema.ema(batch["img"])
                val_loss,val_items=ema.ema.loss(batch,preds)
            if not torch.isfinite(val_loss).all(): raise ValueError("Nonfinite validation-shaped stress loss")
            snapshot("validation_shaped_stress",begin,before,batch=count,extra_2x_stress=multiplier==2,
                     loss=float(val_loss.cpu()))
            del batch,preds,val_loss,val_items
        if report["successful_updates"]<1: raise ValueError("No successful CUDA optimizer update; capacity not established")
        if verify_initialization(weights["path"],args.init_training_report)[0]!=weights:
            raise ValueError("Initializer changed during diagnostic")
        report.update(status="capacity_probe_completed",gpu_after=gpu_sample(args.device))
    except BaseException as error:
        report.update(status="probe_failed_or_aborted",error=f"{type(error).__name__}: {error}")
        if torch is not None and torch.cuda.is_initialized():
            report.update(failure_peak_allocated_bytes=torch.cuda.max_memory_allocated(args.device),
                          failure_peak_reserved_bytes=torch.cuda.max_memory_reserved(args.device))
        raise
    finally:
        report.update(elapsed_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write(report_path,report)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("dataset","config","init-training-report","output"): p.add_argument("--"+name,required=True,type=Path)
    p.add_argument("--batch",required=True,type=int,choices=(4,8))
    p.add_argument("--steps",type=int,default=4)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--headroom-gib",type=float,default=2)
    p.add_argument("--maximum-utilization",type=float,default=80)
    p.add_argument("--wait-seconds",type=float,default=30)
    p.add_argument("--extra-double-validation",action="store_true")
    run(p.parse_args())


if __name__=="__main__": main()
