"""
Feature map computations: contrast, color saliency, movement.
All outputs are float32 arrays normalized to [0, 1].
"""

import cv2
import numpy as np


def compute_contrast(frame_bgr: np.ndarray, ksize: int = 15) -> np.ndarray:
    """Local standard deviation of grayscale intensity (sliding window)."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    k = (ksize, ksize)
    mean = cv2.blur(gray, k)
    mean_sq = cv2.blur(gray ** 2, k)
    variance = np.clip(mean_sq - mean ** 2, 0, None)
    std = np.sqrt(variance)
    return _normalize(std)


def compute_color_saliency(frame_bgr: np.ndarray) -> np.ndarray:
    """Per-pixel Euclidean distance from the frame mean color in CIE LAB."""
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    mean_color = lab.mean(axis=(0, 1), keepdims=True)
    diff = lab - mean_color
    dist = np.sqrt((diff ** 2).sum(axis=2))
    return _normalize(dist)


def compute_movement(
    curr_bgr: np.ndarray, prev_bgr: np.ndarray | None, blur_ksize: int = 5
) -> np.ndarray:
    """Absolute grayscale frame difference. Returns zeros if prev_bgr is None."""
    if prev_bgr is None:
        h, w = curr_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.float32)
    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    diff = np.abs(curr_gray - prev_gray)
    if blur_ksize > 1:
        diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
    return _normalize(diff)


def compute_movement_multiframe(
    curr_bgr: np.ndarray,
    prev_frames: list[np.ndarray],
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Weighted frame-difference movement using multiple prior frames.
    More recent frames contribute more. Falls back to zeros if no prior frames.
    Weights: exponentially decaying — [0.5, 0.3, 0.2] for 3 frames, normalized.
    """
    if not prev_frames:
        h, w = curr_bgr.shape[:2]
        return np.zeros((h, w), dtype=np.float32)

    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    n = len(prev_frames)

    # Generate exponentially decaying weights (most recent = highest)
    raw_weights = [0.6 ** i for i in range(n)]
    total = sum(raw_weights)
    weights = [w / total for w in raw_weights]

    h, w_dim = curr_bgr.shape[:2]
    accum = np.zeros((h, w_dim), dtype=np.float32)

    for idx, prev in enumerate(reversed(prev_frames)):
        prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY).astype(np.float32)
        diff = np.abs(curr_gray - prev_gray)
        if blur_ksize > 1:
            diff = cv2.GaussianBlur(diff, (blur_ksize, blur_ksize), 0)
        accum += weights[idx] * diff

    return _normalize(accum)


def compute_novelty(
    curr_bgr: np.ndarray,
    prev_bgr: np.ndarray | None,
    change_rate_avg: np.ndarray | None,
    ema_alpha: float = 0.3,
    blur_ksize: int = 9,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect unexpected luminance changes between consecutive frames.

    Returns (novelty_map, updated_change_rate_avg).
    - novelty = |delta| * (1 - expectedness)
    - expectedness is a running EMA of per-pixel absolute change rate.
    An expected flicker (steady motion) won't spike; an unexpected flash will.
    """
    h, w = curr_bgr.shape[:2]

    if prev_bgr is None:
        zeros = np.zeros((h, w), dtype=np.float32)
        return zeros, zeros

    curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    delta = np.abs(curr_gray - prev_gray)

    if blur_ksize > 1:
        delta = cv2.GaussianBlur(delta, (blur_ksize, blur_ksize), 0)

    # Update running average of change rate (EMA)
    if change_rate_avg is None:
        updated_avg = delta.copy()
    else:
        updated_avg = (ema_alpha * delta + (1 - ema_alpha) * change_rate_avg).astype(np.float32)

    # Expectedness: how typical is this level of change at each pixel?
    # Normalize the running average to [0, 1] for an expectedness score
    avg_norm = _normalize(updated_avg)
    unexpectedness = 1.0 - avg_norm

    # Novelty = magnitude of change * how unexpected it is
    novelty = delta * unexpectedness

    return _normalize(novelty), updated_avg


def detect_proto_objects(
    frame_bgr: np.ndarray,
    visibility_mask: np.ndarray | None = None,
    visibility_threshold: float = 0.3,
    canny_low: int = 50,
    canny_high: int = 150,
    dilate_iter: int = 2,
    min_area_fraction: float = 0.001,
    max_area_fraction: float = 0.25,
) -> list[dict]:
    """
    Detect proto-objects via Canny edge detection + contour analysis.

    Returns a list of dicts, each with:
      - bbox: (x, y, w, h)
      - centroid: (cx, cy)
      - area: int (pixels)
      - mean_color: (B, G, R) average color within the bounding box
      - contour_signature: float — perimeter/sqrt(area) shape compactness
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, canny_low, canny_high)

    # Dilate to close small gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = frame_bgr.shape[:2]
    total_pixels = h * w
    min_area = int(total_pixels * min_area_fraction)
    max_area = int(total_pixels * max_area_fraction)

    objects = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, bw, bh = cv2.boundingRect(cnt)
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        # Skip objects outside the perception zone
        if visibility_mask is not None:
            cy_c = min(max(cy, 0), h - 1)
            cx_c = min(max(cx, 0), w - 1)
            if visibility_mask[cy_c, cx_c] < visibility_threshold:
                continue

        # Mean color in bounding box
        roi = frame_bgr[y:y + bh, x:x + bw]
        mean_color = tuple(float(v) for v in roi.mean(axis=(0, 1)))

        # Shape compactness: perimeter / sqrt(area)
        perimeter = cv2.arcLength(cnt, True)
        compactness = perimeter / (area ** 0.5 + 1e-6)

        objects.append({
            "bbox": (x, y, bw, bh),
            "centroid": (cx, cy),
            "area": int(area),
            "mean_color": mean_color,
            "contour_signature": round(float(compactness), 4),
        })

    return objects


def _normalize(arr: np.ndarray) -> np.ndarray:
    """Min-max normalize to [0, 1]. Returns zeros if flat."""
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-8:
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - mn) / (mx - mn)).astype(np.float32)
