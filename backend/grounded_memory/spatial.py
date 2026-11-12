"""Sensor geometry -> 3D extents -> observed spatial relations. No semantic vocabulary.

Coordinates follow ARKit's world: metres, +Y up (gravity aligned), one world per capture session.
Relations computed here are perceptual observations with sensor provenance, never human teaching.
"""

import numpy as np

# Percentiles trim depth bleeding across object edges.
TRIM = (10, 90)
TOUCH = 0.06  # metres: "on" when the gap between bottom and top is this small
NEAR = 0.30  # metres: "next to" when separated by at most this much


def camera_from(meta):
    """Pose, intrinsics and sensor resolution as numpy; ARKit sends simd matrices column-major."""
    return dict(
        transform=np.asarray(meta["transform"], float).reshape(4, 4, order="F"),
        intrinsics=np.asarray(meta["intrinsics"], float).reshape(3, 3, order="F"),
        resolution=np.asarray(meta["resolution"], float),  # sensor image width, height
    )


def sensor_from_upright(u, v):
    """Upright (portrait) normalized coordinates -> sensor (landscape) normalized coordinates.

    The phone rotates sensor images 90° clockwise before analysis (iOS orientation `.right`).
    """
    return v, 1 - u


def world_extent(mask, depth, camera, confidence=None, max_points=3000):
    """3D axis-aligned extent of an upright-image mask, using a sensor-oriented depth map in metres.

    Returns None when too few pixels have usable depth.
    """
    vs, us = np.nonzero(mask)
    if len(us) < 20:
        return None
    if len(us) > max_points:
        pick = np.random.default_rng(0).choice(len(us), max_points, replace=False)
        us, vs = us[pick], vs[pick]
    h, w = mask.shape
    xs, ys = sensor_from_upright((us + 0.5) / w, (vs + 0.5) / h)
    dh, dw = depth.shape
    di = np.clip((ys * dh).astype(int), 0, dh - 1)
    dj = np.clip((xs * dw).astype(int), 0, dw - 1)
    d = depth[di, dj]
    usable = np.isfinite(d) & (d > 0.1) & (d < 8)
    if confidence is not None:
        usable &= confidence[di, dj] >= 1  # ARKit: 0 low, 1 medium, 2 high
    if usable.sum() < 20:
        return None
    width, height = camera["resolution"]
    k = camera["intrinsics"]
    px, py, d = xs[usable] * width, ys[usable] * height, d[usable]
    # ARKit camera space: +X right, +Y up, looking down -Z; image rows grow downward.
    local = np.stack([(px - k[0, 2]) / k[0, 0] * d, -(py - k[1, 2]) / k[1, 1] * d, -d, np.ones_like(d)])
    points = (camera["transform"] @ local)[:3].T
    low, high = np.percentile(points, TRIM, axis=0)
    return dict(
        min=low.round(4).tolist(),
        max=high.round(4).tolist(),
        center=np.median(points, axis=0).round(4).tolist(),
        points=int(usable.sum()),
    )


def project(point, camera, depth=None, tolerance=0.15):
    """World point -> upright normalized (u, v) in this camera's frame, or None if behind, outside,
    or (with a depth map in metres) hidden behind something nearer."""
    local = np.linalg.inv(camera["transform"]) @ np.append(np.asarray(point, float), 1)
    distance = -local[2]
    if distance <= 0.05:
        return None
    k = camera["intrinsics"]
    width, height = camera["resolution"]
    xs = (k[0, 0] * local[0] / distance + k[0, 2]) / width
    ys = (-k[1, 1] * local[1] / distance + k[1, 2]) / height
    if not (0 <= xs < 1 and 0 <= ys < 1):
        return None
    if depth is not None:
        seen = depth[int(ys * depth.shape[0]), int(xs * depth.shape[1])]
        if 0.1 < seen < distance - tolerance:
            return None
    return float(1 - ys), float(xs)  # inverse of sensor_from_upright


def _footprint_overlap(a, b):
    """Fraction of the smaller horizontal (x, z) footprint shared by both boxes."""
    lo = np.maximum(a["min"], b["min"])[[0, 2]]
    hi = np.minimum(a["max"], b["max"])[[0, 2]]
    shared = np.prod(np.clip(hi - lo, 0, None))
    area = lambda box: np.prod(np.clip((box["max"] - box["min"])[[0, 2]], 1e-4, None))  # noqa: E731
    return float(shared / min(area(a), area(b)))


def relate(a, b):
    """Observed relation of subject extent `a` to object extent `b`, or None."""
    a = {k: np.asarray(v, float) for k, v in a.items() if k in ("min", "max")}
    b = {k: np.asarray(v, float) for k, v in b.items() if k in ("min", "max")}
    volume = lambda box: np.prod(np.clip(box["max"] - box["min"], 1e-3, None))  # noqa: E731
    margin = 0.03
    if (
        np.all(a["min"] >= b["min"] - margin)
        and np.all(a["max"] <= b["max"] + margin)
        and volume(a) < 0.5 * volume(b)
    ):
        return "inside"
    overlap = _footprint_overlap(a, b)
    gap = a["min"][1] - b["max"][1]
    if overlap > 0.2 and abs(gap) <= TOUCH:
        return "on"
    if overlap > 0.2 and gap > TOUCH:
        return "above"
    separation = np.max(np.maximum(b["min"] - a["max"], a["min"] - b["max"]))
    vertical = min(a["max"][1], b["max"][1]) - max(a["min"][1], b["min"][1])
    if overlap <= 0.2 and 0 < separation <= NEAR and vertical > 0:
        return "next_to"
    return None


def relations(located):
    """All pairwise relations among located entities: {entity_id: {"label":..., "extent": {...}}}."""
    found = []
    for a_id, a in located.items():
        for b_id, b in located.items():
            if a_id == b_id:
                continue
            predicate = relate(a["extent"], b["extent"])
            # next_to is symmetric; keep one direction.
            if predicate and not (predicate == "next_to" and a_id > b_id):
                found.append((a_id, predicate, b_id))
    return found
