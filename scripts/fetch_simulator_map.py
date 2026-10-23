"""Explicit one-shot map preparation. The simulator itself never downloads maps."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlencode
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from camera_rc.simulator.region import QUERY, SOURCE_URL, build_region, canonical_bytes, sanitize_source, validate_region


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New folder; existing files are never overwritten")
    parser.add_argument("--source", type=Path, help="Existing raw Overpass response; avoids network access")
    parser.add_argument("--fetched-at", help="Required UTC ISO timestamp for --source")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("Output must be a new folder")
    if args.source and not args.fetched_at:
        parser.error("--fetched-at is required with --source")
    if args.source:
        raw = args.source.read_bytes()
        fetched = args.fetched_at
    else:
        request = Request(SOURCE_URL + "?" + urlencode({"data": QUERY}),
                          headers={"User-Agent": "camera-rc-training-map-preparation/0.1"})
        with urlopen(request, timeout=60) as response:
            raw = response.read(16 * 1024 * 1024 + 1)
        fetched = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    if len(raw) > 16 * 1024 * 1024:
        raise ValueError("OSM response exceeds the 16 MiB limit")
    source = sanitize_source(json.loads(raw), hashlib.sha256(raw).hexdigest(), fetched)
    region = build_region(source)
    validate_region(region)
    args.output.mkdir(parents=True, exist_ok=False)
    # Raw response is kept here for local audit; only filtered source is bundled.
    (args.output / "overpass.raw.json").write_bytes(raw)
    (args.output / "seoul_seongsu.source.json").write_bytes(canonical_bytes(source) + b"\n")
    (args.output / "seoul_seongsu.json").write_bytes(canonical_bytes(region) + b"\n")
    print(json.dumps({"roads": len(region["roads"]), "buildings": len(region["buildings"]),
                      "route_m": region["route"]["length_m"], "full_route_m": region["full_route"]["length_m"],
                      "geometry_sha256": region["geometry_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
