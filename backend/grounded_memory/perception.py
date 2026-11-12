"""Independent worker: pixels -> regions/features. No semantic vocabulary."""

import hashlib
from pathlib import Path

import cv2
import numpy as np

from .spatial import camera_from, world_extent
from .vendor.attention_maps import compute_color_saliency, compute_contrast

_encoder = None
_segmenter = None


def features(rgb):
    global _encoder
    try:
        import torch
        from transformers import AutoImageProcessor, AutoModel

        if _encoder is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            _encoder = (
                AutoImageProcessor.from_pretrained("facebook/dinov2-small", local_files_only=True),
                AutoModel.from_pretrained("facebook/dinov2-small", local_files_only=True).to(device).eval(),
                device,
            )
        processor, model, device = _encoder
        with torch.inference_mode():
            inputs = {k: v.to(device) for k, v in processor(images=rgb, return_tensors="pt").items()}
            v = model(**inputs).last_hidden_state[:, 0].cpu().numpy()[0]
        revision = getattr(model.config, "_commit_hash", None) or "local"
        return (v / (np.linalg.norm(v) + 1e-8)).tolist(), "dinov2-small:" + revision
    except (ImportError, OSError, RuntimeError) as e:
        if isinstance(e, RuntimeError) and _encoder is not None and _encoder[2] == "mps":
            _encoder = (_encoder[0], _encoder[1].to("cpu"), "cpu")
            return features(rgb)
        # A visible diagnostic fallback; never fabricate model features or object names.
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        v = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256]).flatten()
        return (v / (np.linalg.norm(v) + 1e-8)).tolist(), "color-histogram-v1 (DINO unavailable: " + type(
            e
        ).__name__ + ")"


def overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy / (aw * ah + bw * bh - ix * iy)


def propose(gray, salience, motion, limit=3):
    """Object-sized, mostly central, non-overlapping boxes from nested edge contours.

    Observed on a phone: outer contours merged whole shelves into one box, while slivers touching
    the frame edge outranked the object in the middle of the view.
    """
    h, w = gray.shape
    edges = cv2.dilate(cv2.Canny(gray, 40, 110), np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    scored = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        if (
            not 0.01 <= bw * bh / (w * h) <= 0.5
            or min(bw, bh) < 0.06 * min(w, h)
            or max(bw / bh, bh / bw) > 4
        ):
            continue
        border = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        dx, dy = (x + bw / 2) / w - 0.5, (y + bh / 2) / h - 0.5
        centrality = 1 - min(1, (dx * dx + dy * dy) ** 0.5 / 0.7)
        score = (
            salience[y : y + bh, x : x + bw].mean()
            + 2 * motion[y : y + bh, x : x + bw].mean()
            + 0.35 * centrality
            - (0.25 if border else 0)
        )
        scored.append((float(score), (x, y, bw, bh)))
    chosen = []
    for score, box in sorted(scored, reverse=True):
        if all(overlap(box, other) < 0.3 for _, other in chosen):
            chosen.append((score, box))
        if len(chosen) == limit:
            break
    return chosen


def _load(frame_path, longest=640):
    image = cv2.imread(frame_path)
    if image is None:
        raise ValueError("Frame cannot be decoded")
    h, w = image.shape[:2]
    scale = min(1, longest / max(h, w))
    return cv2.resize(image, (round(w * scale), round(h * scale)))


def _segment(image):
    """SAM predictor primed with `image`, or None and a visible reason."""
    global _segmenter
    try:
        # Explicit local checkpoint; no background weight downloads during capture.
        import os

        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = os.getenv("GM_SAM_CHECKPOINT", "")
        if not checkpoint:
            import json

            config = Path(
                os.getenv("GM_DATA_DIR") or Path.home() / "Library/Application Support/Grounded Memory"
            )
            config = config / "models.json"
            if not config.exists():
                config = Path.home() / "Library/Application Support/Grounded Memory/models.json"
            if config.exists():
                checkpoint = json.loads(config.read_text()).get("sam_checkpoint", "")
        if not (checkpoint and Path(checkpoint).is_file()):
            return None, "SAM unavailable; rectangular proposals"
        if _segmenter is None:
            from sam2.build_sam import build_sam2

            _segmenter = SAM2ImagePredictor(
                build_sam2("configs/sam2.1/sam2.1_hiera_t.yaml", checkpoint, device="cpu")
            )
        _segmenter.set_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return _segmenter, "sam2.1-tiny:cpu"
    except (ImportError, OSError, RuntimeError) as e:
        return None, "SAM unavailable: " + str(e)[:120]


def clean(mask, points=()):
    """Tidy a SAM mask: close small gaps, drop specks, fill holes, keep the part the person meant.

    Keeps connected pieces containing an include point; otherwise the largest piece.
    """
    mask = np.asarray(mask, bool).astype(np.uint8)
    if not mask.any():
        return mask.astype(bool)
    h, w = mask.shape
    k = max(3, round(min(h, w) / 160)) | 1
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count <= 1:
        return _fill_small_holes(mask).astype(bool)
    wanted = {
        labels[min(h - 1, int(y * h)), min(w - 1, int(x * w))]
        for x, y, include in points
        if include and labels[min(h - 1, int(y * h)), min(w - 1, int(x * w))] > 0
    }
    if not wanted:
        wanted = {1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))}
    mask = np.isin(labels, list(wanted)).astype(np.uint8)
    return _fill_small_holes(mask).astype(bool)


def _fill_small_holes(mask):
    """Fill holes under 5% of the object's area (speckle); keep real openings such as a mug handle."""
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return mask
    limit = 0.05 * mask.sum()
    holes = [
        c for c, (*_, parent) in zip(contours, hierarchy[0]) if parent >= 0 and cv2.contourArea(c) < limit
    ]
    if holes:
        mask = mask.copy()
        cv2.drawContours(mask, holes, -1, 1, thickness=cv2.FILLED)
    return mask


def pick_resize(current, candidates, direction):
    """Next distinctly larger (or smaller) outline that still agrees with the current one, or None."""
    area = current.sum()
    best = None
    for mask in candidates:
        size = mask.sum()
        if not size:
            continue
        shared = np.logical_and(mask, current).sum()
        if direction == "larger" and size >= 1.2 * area and shared >= 0.8 * area:
            if best is None or size < best.sum():
                best = mask
        if direction == "smaller" and size <= 0.83 * area and shared >= 0.8 * size:
            if best is None or size > best.sum():
                best = mask
    return best


def _region(image, box, mask, score, asset_dir):
    h, w = image.shape[:2]
    x, y, bw, bh = box
    crop = image[y : y + bh, x : x + bw]
    rgba = cv2.cvtColor(crop, cv2.COLOR_BGR2BGRA)
    if mask is not None:
        rgba[:, :, 3] = (mask[y : y + bh, x : x + bw] * 255).astype(np.uint8)
    key = hashlib.sha256(crop.tobytes()).hexdigest() + ".png"
    path = Path(asset_dir) / key
    if not path.exists():
        cv2.imwrite(str(path), rgba)
    vector, encoder = features(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    return dict(
        id=key[:16],
        bbox=[x / w, y / h, bw / w, bh / h],
        asset=key,
        salience=score,
        features=vector,
        encoder=encoder,
        mask=mask is not None,
    )


def _geometry(geometry):
    """Optional depth stream for a frame: {"depth": .npz path, "camera": pose/intrinsics metadata}."""
    if not geometry:
        return None
    data = np.load(geometry["depth"])
    confidence = data["confidence"] if "confidence" in data else None
    return data["depth"], confidence, camera_from(geometry["camera"])


def _locate(region, mask, geometry):
    if geometry is not None and mask is not None:
        depth, confidence, camera = geometry
        region["extent"] = world_extent(mask, depth, camera, confidence)
    return region


def world_motion(prior, gray):
    """Residual motion after cancelling the camera's own movement between two frames.

    Tracked corners give a homography for the camera's motion; what still differs after warping
    the earlier frame is something moving in the world (or parallax from nearby surfaces). Returns
    None when the motion cannot be modelled, so callers fall back rather than guess.
    """
    corners = cv2.goodFeaturesToTrack(prior, 400, 0.01, 8)
    if corners is None or len(corners) < 30:
        return None
    moved, status, _ = cv2.calcOpticalFlowPyrLK(prior, gray, corners, None, winSize=(31, 31), maxLevel=4)
    tracked = status.ravel() == 1
    if tracked.sum() < 30:
        return None
    homography, inliers = cv2.findHomography(corners[tracked], moved[tracked], cv2.RANSAC, 3.0)
    if homography is None or inliers.mean() < 0.5:
        return None
    h, w = gray.shape
    warped = cv2.warpPerspective(prior, homography, (w, h))
    valid = cv2.warpPerspective(np.full_like(prior, 255), homography, (w, h)) > 0
    valid = cv2.erode(valid.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    residual = cv2.absdiff(cv2.GaussianBlur(gray, (5, 5), 0), cv2.GaussianBlur(warped, (5, 5), 0)) / 255
    residual[~valid] = 0
    moving = cv2.morphologyEx((residual > 0.15).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    count, _, stats, _ = cv2.connectedComponentsWithStats(moving)
    flow = np.linalg.norm((moved[tracked] - corners[tracked]).reshape(-1, 2), axis=1)
    result = dict(
        residual=residual,
        camera_motion=float(np.median(flow) / np.hypot(w, h)),
        world_motion=0.0,
        world_motion_box=None,
    )
    if count > 1:
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        x, y, bw, bh, area = stats[largest]
        result.update(world_motion=float(area / moving.size), world_motion_box=[x / w, y / h, bw / w, bh / h])
    return result


def analyze(frame_path, previous_path, asset_dir, geometry=None):
    image = _load(frame_path)
    located = _geometry(geometry)
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    salience = (compute_contrast(image) + compute_color_saliency(image)) / 2
    motion = np.zeros_like(salience)
    compensated = None
    if previous_path:
        prior = cv2.imread(previous_path)
        if prior is not None:
            prior = cv2.cvtColor(cv2.resize(prior, (w, h)), cv2.COLOR_BGR2GRAY)
            motion = cv2.absdiff(gray, prior) / 255
            compensated = world_motion(prior, gray)
    # Proposals favour what moved in the world, not edges that shifted because the camera did.
    candidates = propose(gray, salience, compensated["residual"] if compensated else motion)
    # Motion blur makes poor evidence; such frames still count for recognition but not for questions.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    predictor, sam_status = _segment(image) if candidates else (None, "no proposals")
    regions = []
    for score, (x, y, bw, bh) in candidates:
        mask = None
        if predictor is not None:
            masks, _, _ = predictor.predict(box=np.array([x, y, x + bw, y + bh]), multimask_output=False)
            mask = clean(masks[0])
        regions.append(_locate(_region(image, (x, y, bw, bh), mask, score, asset_dir), mask, located))
    scene, encoder = features(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return dict(
        regions=regions,
        scene_features=scene,
        encoder=encoder,
        segmentation=sam_status,
        sharpness=sharpness,
        brightness=float(gray.mean() / 255),
        global_motion=float(np.median(motion)),
        local_motion=float(np.quantile(motion, 0.95)),
        **({k: v for k, v in compensated.items() if k != "residual"} if compensated else {}),
        compute_device=_encoder[2] if _encoder else "cpu-fallback",
    )


def segment(frame_path, prompt, asset_dir, geometry=None):
    """Segment from the person's prompt on a frozen frame, at higher resolution than live analysis.

    prompt: {"points": [[x, y, include], ...] normalized, "box": [x, y, w, h] normalized or None}.
    """
    image = _load(frame_path, longest=960)
    h, w = image.shape[:2]
    predictor, status = _segment(image)
    if predictor is None:
        raise ValueError(status)
    points = prompt.get("points") or []
    box = prompt.get("box")
    masks, scores, _ = predictor.predict(
        point_coords=np.array([[x * w, y * h] for x, y, _ in points]) if points else None,
        point_labels=np.array([1 if include else 0 for _, _, include in points]) if points else None,
        box=np.array([box[0] * w, box[1] * h, (box[0] + box[2]) * w, (box[1] + box[3]) * h]) if box else None,
        multimask_output=True,
    )
    # SAM offers part / object / surroundings; ignore empty masks and ones covering most of the view.
    options = [(float(s), clean(m, points)) for s, m in zip(scores, masks) if 0.002 < m.mean() < 0.85]
    if not options:
        raise ValueError("Nothing distinct there; tap the middle of an object")
    score, mask = max(options, key=lambda o: o[0])
    return _finish(image, mask, score, asset_dir, geometry, dict(points=points, box=box))


def segment_excluding(frame_path, box, known, asset_dir, geometry=None):
    """Outline what is in `box` minus things already known at `known` points (upright normalized).

    A single negative point only nudges SAM, so each known thing is segmented on its own and its whole
    outline removed; the main remaining piece is the unknown thing.
    """
    image = _load(frame_path, longest=960)
    h, w = image.shape[:2]
    predictor, status = _segment(image)
    if predictor is None:
        raise ValueError(status)
    corners = np.array([box[0] * w, box[1] * h, (box[0] + box[2]) * w, (box[1] + box[3]) * h])
    masks, scores, _ = predictor.predict(box=corners, multimask_output=False)
    candidate = clean(masks[0])
    covered = np.zeros_like(candidate)
    for u, v in known:
        options, quality, _ = predictor.predict(
            point_coords=np.array([[u * w, v * h]]), point_labels=np.array([1]), multimask_output=True
        )
        # The known thing: SAM's best outline that is not most of the candidate itself.
        fitting = [(q, m) for q, m in zip(quality, options) if m.sum() < 0.7 * max(candidate.sum(), 1)]
        if fitting:
            covered |= max(fitting, key=lambda o: o[0])[1].astype(bool)
    k = max(3, round(min(h, w) / 120)) | 1
    covered = cv2.dilate(covered.astype(np.uint8), np.ones((k, k), np.uint8)).astype(bool)
    remaining = clean(candidate & ~covered)
    if remaining.sum() < 0.15 * max(candidate.sum(), 1):
        raise ValueError("Only already-known things there")
    region = _finish(image, remaining, float(scores[0]), asset_dir, geometry, dict(points=[], box=list(box)))
    region["excluded_known"] = len(known)
    return region


def resize(frame_path, region, direction, asset_dir, geometry=None, crop_path=None):
    """Find the next distinctly larger or smaller outline around the current one.

    SAM gives only three options per prompt, often nearly identical, so this re-prompts with grown or
    shrunk boxes and the original points, then picks the nearest outline that differs by at least ~20%.
    """
    image = _load(frame_path, longest=960)
    h, w = image.shape[:2]
    predictor, status = _segment(image)
    if predictor is None:
        raise ValueError(status)
    crop = cv2.imread(str(crop_path or Path(asset_dir) / region["asset"]), cv2.IMREAD_UNCHANGED)
    x, y, bw, bh = [round(v * s) for v, s in zip(region["bbox"], (w, h, w, h))]
    current = np.zeros((h, w), bool)
    if crop is not None and crop.shape[2] == 4:
        alpha = cv2.resize(crop[:, :, 3], (max(bw, 1), max(bh, 1)), interpolation=cv2.INTER_NEAREST) > 127
        current[y : y + alpha.shape[0], x : x + alpha.shape[1]] = alpha[: h - y, : w - x]
    else:
        current[y : y + bh, x : x + bw] = True
    prompt = region.get("prompt") or {}
    points = prompt.get("points") or []
    labels = np.array([1 if include else 0 for *_, include in points]) if points else None
    coords = np.array([[px * w, py * h] for px, py, _ in points]) if points else None
    cx, cy = x + bw / 2, y + bh / 2
    factors = (1.4, 1.9, 2.6, 3.5) if direction == "larger" else (0.85, 0.7, 0.55, 0.4)
    candidates = []
    for factor in factors:
        half_w, half_h = bw * factor / 2, bh * factor / 2
        box = np.array([max(0, cx - half_w), max(0, cy - half_h), min(w, cx + half_w), min(h, cy + half_h)])
        masks, _, _ = predictor.predict(
            point_coords=coords, point_labels=labels, box=box, multimask_output=True
        )
        candidates += [clean(m, points) for m in masks if 0.0005 < m.mean() < 0.9]
        chosen = pick_resize(current, candidates, direction)
        if chosen is not None:
            region = _finish(image, chosen, 1.0, asset_dir, geometry, dict(points=points, box=None))
            return region
    raise ValueError(f"That's as {'large' if direction == 'larger' else 'small'} as I can find here")


def _finish(image, mask, score, asset_dir, geometry, prompt):
    h, w = image.shape[:2]
    ys, xs = np.nonzero(mask)
    box_px = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    region = _locate(_region(image, box_px, mask, score, asset_dir), mask, _geometry(geometry))
    region["prompt"] = prompt
    return region


def segment_point(frame_path, point, asset_dir, geometry=None):
    """The object under a point the person tapped (normalized x, y)."""
    return segment(frame_path, {"points": [[point[0], point[1], True]]}, asset_dir, geometry)


def preview(frame_path, region, asset_dir):
    """The frozen frame with the region's outline, for checking and correcting boundaries."""
    image = _load(frame_path)
    h, w = image.shape[:2]
    crop = cv2.imread(str(Path(asset_dir) / region["asset"]), cv2.IMREAD_UNCHANGED)
    x, y = round(region["bbox"][0] * w), round(region["bbox"][1] * h)
    mask = np.zeros((h, w), np.uint8)
    if crop is not None and crop.shape[2] == 4:
        # Crops may come from a higher-resolution segmentation than this preview.
        bw, bh = max(1, round(region["bbox"][2] * w)), max(1, round(region["bbox"][3] * h))
        alpha = cv2.resize(crop[:, :, 3], (bw, bh), interpolation=cv2.INTER_LINEAR) > 127
        mask[y : y + bh, x : x + bw] = alpha[: h - y, : w - x]
    else:
        bw, bh = round(region["bbox"][2] * w), round(region["bbox"][3] * h)
        mask[y : y + bh, x : x + bw] = 1
    shaded = (image * 0.45).astype(np.uint8)
    out = np.where(mask[..., None] > 0, image, shaded)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, (0, 220, 255), 3)
    for px, py, include in (region.get("prompt") or {}).get("points", []):
        centre = (round(px * w), round(py * h))
        cv2.circle(out, centre, 11, (255, 255, 255), -1)
        cv2.circle(out, centre, 8, (80, 200, 80) if include else (60, 60, 230), -1)
    name = hashlib.sha256(out.tobytes()).hexdigest()[:24] + ".preview.jpg"
    cv2.imwrite(str(Path(asset_dir) / name), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return name


def sensor_relation(example, region):
    """How two observations' capture devices relate. Different projections are never compared."""
    if example.get("sensor_id") and example.get("sensor_id") == region.get("sensor_id"):
        return "same"
    a, b = example.get("projection", "unknown"), region.get("projection", "unknown")
    if "unknown" in (a, b):
        return "unverified"
    return "compatible" if a == b else "incompatible"


def rank_matches(region, entities, place_id=None):
    """Appearance matches against remembered examples from compatible sensors and the same encoder."""
    ranked = []
    for entity in entities:
        best, sensors = 0.0, None
        for example in entity.get("examples", []):
            relation = sensor_relation(example, region)
            if example.get("encoder") != region.get("encoder") or relation == "incompatible":
                continue
            a = np.asarray(example.get("features", []))
            b = np.asarray(region.get("features", []))
            if a.shape != b.shape or not a.size:
                continue
            similarity = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
            if similarity > best:
                best, sensors = similarity, relation
        if best > 0.5:
            context = 0.03 if any(e.get("place_id") == place_id for e in entity.get("examples", [])) else 0
            ranked.append(
                {
                    "id": entity["id"],
                    "label": entity["label"],
                    "appearance": best,
                    "score": min(1, best + context),
                    "sensors": sensors,
                }
            )
    return sorted(ranked, key=lambda x: x["score"], reverse=True)
