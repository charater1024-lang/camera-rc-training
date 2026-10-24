"""Dependency-free semantic contract; class indices never define meaning alone."""
LABEL_SPACES = {
    "road_surface_v1": {"0": "road", "1": "nonroad", "255": "ignore"},
    "ego_lane_v1": {"0": "annotated_outside_ego_lane", "1": "ego_lane", "255": "unobserved_or_unannotated"},
}
DRIVABLE = {"road_surface_v1": [0], "ego_lane_v1": [1]}


def validate_label_contract(runtime, stored):
    data = stored.get("data", {})
    actual = data.get("label_space_id")
    expected = runtime.get("expected_label_space_id")
    if expected not in LABEL_SPACES or actual != expected:
        raise ValueError("Explicit expected_label_space_id must match a supported checkpoint label_space_id")
    ids = runtime.get("drivable_class_ids")
    if not isinstance(ids, list) or any(type(i) is not int for i in ids) or ids != DRIVABLE[expected]:
        raise ValueError("drivable_class_ids contradict the label-space semantics")
    if stored.get("model", {}).get("num_classes") != 2:
        raise ValueError("Registered label space requires exactly two model classes")
    labels = data.get("labels")
    if labels is not None and labels != LABEL_SPACES[expected]:
        raise ValueError("Checkpoint class names contradict its registered label space")
    return {"label_space_id": actual, "labels": dict(LABEL_SPACES[actual]), "drivable_class_ids": list(ids)}
