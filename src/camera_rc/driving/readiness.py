"""Offline evidence audit. A successful audit NEVER authorizes actuation."""
import hashlib
import json
import math
from pathlib import Path

REQUIRED_CHECKS=("metric_range","tracking","localization","lane_and_rules","exposure_timing",
                 "braking_and_steering","independent_watchdog","closed_loop_odd")


def audit_bundle(manifest_path):
    path=Path(manifest_path).resolve()
    data=json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data,dict):
        raise ValueError("Validation manifest must be a JSON object")
    reasons=[]
    platform=data.get("platform")
    if platform not in {"simulator","rc_car","road_vehicle"}:
        reasons.append("explicit_platform_required")
    odd=data.get("odd",{})
    if not isinstance(odd,dict) or not all(odd.get(k) for k in ("area","road_type","conditions","max_speed_mps")):
        reasons.append("explicit_odd_required")
    if isinstance(odd,dict):
        speed=odd.get("max_speed_mps")
        if type(speed) not in (int,float) or not math.isfinite(speed) or speed<=0:
            reasons.append("positive_measured_speed_limit_required")
    pins=data.get("artifacts",{})
    if not isinstance(pins,dict) or not {"checkpoint","calibration","vehicle_config","code_revision"}.issubset(pins):
        reasons.append("artifact_pins_missing")
        pins={}
    actual={}
    for name,pin in pins.items():
        try:
            target=Path(pin["path"])
            target=target if target.is_absolute() else path.parent/target
            digest=hashlib.sha256(target.read_bytes()).hexdigest()
            actual[name]=digest
            if digest!=pin["sha256"]:
                reasons.append("artifact_hash_mismatch:"+name)
        except (KeyError,TypeError,OSError):
            reasons.append("artifact_unavailable:"+name)
    checks={}
    supplied=data.get("checks",{})
    for name in REQUIRED_CHECKS:
        item=supplied.get(name,{}) if isinstance(supplied,dict) else {}
        ok=False
        try:
            report_path=path.parent/item["report"]
            raw=report_path.read_bytes()
            report=json.loads(raw.decode("utf-8-sig"))
            if not isinstance(report,dict):
                raise ValueError("Report must be an object")
            ok=(hashlib.sha256(raw).hexdigest()==item["sha256"] and report.get("check")==name
                and report.get("passed") is True and report.get("measurement_source")=="measured"
                and report.get("artifact_hashes")==actual and bool(actual)
                and report.get("odd")==odd and report.get("platform")==platform
                and isinstance(report.get("independent_test_set"),str) and bool(report["independent_test_set"].strip()))
        except (KeyError,TypeError,OSError,ValueError):
            pass
        checks[name]="evidence_present" if ok else "missing_or_invalid"
        if not ok:
            reasons.append("measurement_required:"+name)
    return {"schema_version":1,"audit_passed":not reasons,"checks":checks,"reasons":reasons,
            "artifact_hashes":actual,"physical_output":False,"actuation_authorized":False,
            "platform":platform,
            "scope":"File identity and declared evidence only; requires human engineering review and independent measurement."}
