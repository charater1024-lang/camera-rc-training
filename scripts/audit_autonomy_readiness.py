"""Audit measured evidence files without loading models or opening hardware."""
import argparse
import json
from pathlib import Path
from camera_rc.driving.readiness import audit_bundle

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("manifest")
    parser.add_argument("--output", required=True)
    args=parser.parse_args()
    result=audit_bundle(args.manifest)
    target=Path(args.output)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open("x",encoding="utf-8",newline="\n") as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    raise SystemExit(0 if result["audit_passed"] else 2)
