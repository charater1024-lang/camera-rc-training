"""Packaged engineering progress, independent of simulated driving metrics."""
import json
from importlib.resources import files


def improvement_progress():
    data=json.loads(files("camera_rc.simulator").joinpath("assets","improvement-progress.json").read_text(encoding="utf-8"))
    data["implemented_items"]=sum(x["status"]=="implemented" for x in data["items"])
    data["partial_items"]=sum(x["status"]=="partial" for x in data["items"])
    data["total_items"]=len(data["items"])
    return data
