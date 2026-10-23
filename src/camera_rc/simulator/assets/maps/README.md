# Seoul Seongsu OpenStreetMap extract

Map data © OpenStreetMap contributors. The source extract and derived map in
this directory are available under **ODbL 1.0**, separately from the software
licence: <https://www.openstreetmap.org/copyright> and
<https://opendatacommons.org/licenses/odbl/1-0/>.

`seoul_seongsu.source.json` retains the original OSM way/node IDs, geographic
geometry, and an allowlist of physical road/building tags. It omits contributor
identities, building names, contact fields, and address fields. The complete
query, OSM database timestamp, retrieval timestamp, original HTTP-response
SHA-256 and licence are embedded. The unfiltered HTTP response is retained only
in the preparer's local audit folder, not distributed with the package.

`seoul_seongsu.json` is the reproducibly derived map: 161 road/path polylines,
554 building footprints, an 80 m reference interval on Seongsui-ro (성수이로),
and its 473.55 m connected source road. The requested bounding box is
37.5415–37.5485 north, 127.0500–127.0610 east (approximately 970 × 778 m).
Coordinates use a local equirectangular projection around 37.545, 127.0555,
Earth radius 6,371,008.8 m, x east and y north. Road segments are clipped to the
box; building footprints crossing its edge retain their original closed rings.
Route endpoints are interpolated on source OSM segments; no route smoothing or
invented street segments are applied.

OSM centreline and footprint coordinates do **not** establish surveyed road
clearance or lawful lane navigation. Missing road widths use declared simulator
half-width assumptions: primary 6 m, secondary 5.5 m, tertiary 5 m, residential
and unclassified 3.5 m, service 2.5 m. Other path/link widths are declared in
`region.DEFAULT_HALF_WIDTHS`. An OSM `width` tag takes precedence. Missing building
height uses OSM levels × an assumed 3 m, or a declared default 9 m. Every object
stores the source of its width/height. Assumptions are not map measurements.

Footways, cycleways, steps and other non-vehicle path classes are excluded from
the vehicle ground union. Explicit no/private/permit access, conditional access,
bridges, tunnels and nonzero OSM layers are also excluded. Source tags remain
available for inspection. The reference route is a **centreline simulation
demonstration**, not a route recommendation. Turn-restriction relations, signals,
lane geometry, traffic rules, ground slope, and live traffic are not modelled.
This bounded extract includes building ways, not multipolygon relations, and
OSM coverage may be incomplete. Synthetic pedestrians/traffic are separate from
the OSM geometry.

To prepare a fresh extract explicitly (network access required):

```text
python scripts/fetch_simulator_map.py --output runs/seongsu-map-fresh
```

To rebuild from a locally retained original HTTP response without network:

```text
python scripts/fetch_simulator_map.py --source path/to/overpass.raw.json --fetched-at 2026-10-23T10:28:40.770306Z --output runs/seongsu-map-rebuild
```

The output must be new. Only the filtered `.source.json` and derived `.json`
belong in this directory; do not copy the raw response here. The simulator loads
and validates the bundled source-file and geometry hashes without network access.
`build_region()` deterministically reproduces the derived document from the
filtered source. Downloading an updated map is a deliberate maintenance action.
