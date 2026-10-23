"""Configuration paths are relative to the JSON config, never the caller's cwd."""
import json
from pathlib import Path


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text(encoding="utf-8-sig"))
    if cfg.get("task") == "detection":
        for key in ("data", "project"):
            if cfg.get(key):
                p = Path(cfg[key]).expanduser()
                cfg[key] = str((p if p.is_absolute() else path.parent / p).resolve())
        # A model name such as yolo11n.pt is resolved by Ultralytics. Explicit
        # model paths with directories are relative to this config file.
        model = cfg.get("model", "")
        if isinstance(model, str) and ("/" in model or "\\" in model):
            p = Path(model).expanduser()
            cfg["model"] = str((p if p.is_absolute() else path.parent / p).resolve())
        return cfg
    def resolve(value):
        p = Path(value).expanduser()
        return str((p if p.is_absolute() else path.parent / p).resolve())

    for section, keys in (("data", ("train_manifest", "val_manifest")),
                          ("training", ("output_dir", "init_from")),
                          ):
        for key in keys:
            value = cfg.get(section, {}).get(key)
            if value:
                cfg[section][key] = resolve(value)
    for key in ("train_sources", "val_manifests"):
        sources = cfg.get("data", {}).get(key)
        if sources is not None:
            if not isinstance(sources, list) or not sources:
                raise ValueError(f"data.{key} must be a nonempty list")
            for index, source in enumerate(sources):
                if isinstance(source, str):
                    if key == "train_sources":
                        raise ValueError("train_sources entries require {manifest, weight}")
                    sources[index] = resolve(source)
                elif isinstance(source, dict) and isinstance(source.get("manifest"), str):
                    source["manifest"] = resolve(source["manifest"])
                else:
                    raise ValueError(f"data.{key} entries require a manifest path")
    return cfg
