"""
Object memory: track proto-objects across frames with memory buckets.
Scene memory: track scene-level signatures for recognition and prediction.

Each memory bucket stores a detected object's shape/color signature,
bounding box, timestamps, and accumulated attention time. Objects are
matched across frames via bbox IoU + color similarity. Unmatched
detections create new buckets; stale buckets decay over time.

SceneMemory stores scene-level signatures (color histogram, spatial
layout, object arrangement) and compares current frames against stored
scenes to enable recognition and hypothesis-driven attention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class MemoryBucket:
    """A single tracked object in memory."""
    uid: int
    bbox: tuple[int, int, int, int]       # (x, y, w, h)
    centroid: tuple[int, int]
    mean_color: tuple[float, float, float] # BGR
    contour_signature: float
    first_seen: int                        # frame index
    last_seen: int                         # frame index
    attention_frames: int = 0              # frames where attention point was inside bbox
    familiarity: float = 0.0              # 0 = brand new, 1 = very familiar
    strength: float = 1.0                  # decays when not seen; removed when ≤ 0
    discovered_frame: int = -1             # frame when first perceived via attention (-1 = never)


class ObjectMemory:
    """
    Manages a collection of MemoryBuckets across frames.

    - `update()` matches current detections to existing buckets
    - `decay()` reduces strength of unseen objects
    - `get_objects_for_frame()` returns overlay info (color-coded by familiarity)
    """

    def __init__(
        self,
        iou_threshold: float = 0.2,
        color_threshold: float = 80.0,
        decay_rate: float = 0.1,
        familiarity_gain: float = 0.08,
        remove_threshold: float = 0.05,
    ):
        self.iou_threshold = iou_threshold
        self.color_threshold = color_threshold
        self.decay_rate = decay_rate
        self.familiarity_gain = familiarity_gain
        self.remove_threshold = remove_threshold
        self.buckets: list[MemoryBucket] = []
        self._next_uid: int = 0

    def update(
        self,
        detected_objects: list[dict],
        attention_point_xy: tuple[int, int],
        frame_idx: int,
    ) -> list[dict]:
        """
        Match detections to existing buckets, create new ones, and return
        per-object overlay descriptors for this frame.

        Returns list of dicts:
          {uid, bbox, centroid, status, familiarity, attended}
          status: "new" | "known" | "familiar"
        """
        ax, ay = attention_point_xy
        matched_bucket_uids: set[int] = set()
        matched_det_indices: set[int] = set()
        frame_objects: list[dict] = []

        # Greedy matching: best IoU+color match per detection
        pairs: list[tuple[float, int, int]] = []  # (score, det_idx, bucket_idx)
        for di, det in enumerate(detected_objects):
            for bi, bucket in enumerate(self.buckets):
                iou = _bbox_iou(det["bbox"], bucket.bbox)
                cdist = _color_distance(det["mean_color"], bucket.mean_color)
                if iou >= self.iou_threshold and cdist <= self.color_threshold:
                    score = iou - cdist / 255.0  # higher is better
                    pairs.append((score, di, bi))

        pairs.sort(key=lambda t: t[0], reverse=True)

        for score, di, bi in pairs:
            if di in matched_det_indices or self.buckets[bi].uid in matched_bucket_uids:
                continue
            matched_det_indices.add(di)
            matched_bucket_uids.add(self.buckets[bi].uid)

            det = detected_objects[di]
            bucket = self.buckets[bi]
            # Update bucket with latest detection
            bucket.bbox = det["bbox"]
            bucket.centroid = det["centroid"]
            bucket.mean_color = det["mean_color"]
            bucket.contour_signature = det["contour_signature"]
            bucket.last_seen = frame_idx
            bucket.strength = min(bucket.strength + 0.3, 1.0)

            # Check if attention is on this object
            attended = _point_in_bbox(ax, ay, bucket.bbox)
            if attended:
                bucket.attention_frames += 1
                bucket.familiarity = min(bucket.familiarity + self.familiarity_gain, 1.0)

            status = "familiar" if bucket.familiarity > 0.5 else "known"
            frame_objects.append({
                "uid": bucket.uid,
                "bbox": bucket.bbox,
                "centroid": bucket.centroid,
                "status": status,
                "familiarity": round(bucket.familiarity, 4),
                "attended": attended,
                "discovered_frame": bucket.discovered_frame,
            })

        # Create new buckets for unmatched detections
        for di, det in enumerate(detected_objects):
            if di in matched_det_indices:
                continue
            uid = self._next_uid
            self._next_uid += 1
            attended = _point_in_bbox(ax, ay, det["bbox"])
            bucket = MemoryBucket(
                uid=uid,
                bbox=det["bbox"],
                centroid=det["centroid"],
                mean_color=det["mean_color"],
                contour_signature=det["contour_signature"],
                first_seen=frame_idx,
                last_seen=frame_idx,
                attention_frames=1 if attended else 0,
                familiarity=self.familiarity_gain if attended else 0.0,
                strength=1.0,
                discovered_frame=frame_idx,
            )
            self.buckets.append(bucket)
            frame_objects.append({
                "uid": uid,
                "bbox": det["bbox"],
                "centroid": det["centroid"],
                "status": "new",
                "familiarity": round(bucket.familiarity, 4),
                "attended": attended,
                "discovered_frame": frame_idx,
            })

        # Include unmatched existing buckets (still in memory but not detected this frame)
        for bucket in self.buckets:
            if bucket.uid not in matched_bucket_uids and bucket.uid not in {o["uid"] for o in frame_objects}:
                status = "familiar" if bucket.familiarity > 0.5 else "known"
                frame_objects.append({
                    "uid": bucket.uid,
                    "bbox": bucket.bbox,
                    "centroid": bucket.centroid,
                    "status": status,
                    "familiarity": round(bucket.familiarity, 4),
                    "attended": False,
                    "discovered_frame": bucket.discovered_frame,
                })

        return frame_objects

    def decay(self, frame_idx: int) -> None:
        """Reduce strength of all buckets and remove dead ones."""
        alive = []
        for bucket in self.buckets:
            frames_since = frame_idx - bucket.last_seen
            if frames_since > 0:
                bucket.strength -= self.decay_rate * frames_since
                bucket.familiarity *= 0.98  # slow familiarity decay for unseen objects
            if bucket.strength > self.remove_threshold:
                alive.append(bucket)
        self.buckets = alive

    def reset(self) -> None:
        """Clear all memory."""
        self.buckets.clear()
        self._next_uid = 0

    def snapshot(self) -> list[dict]:
        """Return serializable snapshot of all memory buckets."""
        return [
            {
                "uid": b.uid,
                "bbox": list(b.bbox),
                "centroid": list(b.centroid),
                "mean_color": [round(c, 1) for c in b.mean_color],
                "contour_signature": b.contour_signature,
                "first_seen": b.first_seen,
                "last_seen": b.last_seen,
                "attention_frames": b.attention_frames,
                "familiarity": round(b.familiarity, 4),
                "strength": round(b.strength, 4),
                "discovered_frame": b.discovered_frame,
            }
            for b in self.buckets
        ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Intersection over Union for two (x, y, w, h) bounding boxes."""
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    ax2, ay2 = ax1 + aw, ay1 + ah
    bx2, by2 = bx1 + bw, by1 + bh

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _color_distance(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    """Euclidean distance between two BGR color tuples."""
    return math.sqrt(sum((ai - bi) ** 2 for ai, bi in zip(a, b)))


def _point_in_bbox(px: int, py: int, bbox: tuple[int, int, int, int]) -> bool:
    """Check if point (px, py) is inside (x, y, w, h) bounding box."""
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


# ---------------------------------------------------------------------------
# Scene Memory
# ---------------------------------------------------------------------------

@dataclass
class SceneSignature:
    """Lightweight scene-level signature for recognition."""
    color_histogram: np.ndarray       # 48 floats: 16 bins × 3 channels (BGR)
    spatial_layout: np.ndarray        # 16 floats: 4×4 grid mean luminance
    object_arrangement: list[dict]    # [{centroid_norm: (nx, ny), mean_color: (b,g,r), area_ratio: float}]
    frame_idx: int

    @staticmethod
    def compute(frame_bgr: np.ndarray, frame_objects: list[dict], frame_idx: int) -> SceneSignature:
        """Build a scene signature from a frame and its detected objects."""
        h, w = frame_bgr.shape[:2]

        # Color histogram: 16 bins per channel, normalized
        hist_bins = 16
        hist_vec = []
        for ch in range(3):
            hist = np.histogram(frame_bgr[:, :, ch], bins=hist_bins, range=(0, 256))[0]
            hist = hist.astype(np.float32)
            total = hist.sum()
            if total > 0:
                hist /= total
            hist_vec.append(hist)
        color_histogram = np.concatenate(hist_vec)  # shape (48,)

        # Spatial layout: 4×4 grid of mean luminance
        gray = np.mean(frame_bgr, axis=2).astype(np.float32) / 255.0
        grid = np.zeros((4, 4), dtype=np.float32)
        cell_h, cell_w = h // 4, w // 4
        for gy in range(4):
            for gx in range(4):
                y0, y1 = gy * cell_h, (gy + 1) * cell_h if gy < 3 else h
                x0, x1 = gx * cell_w, (gx + 1) * cell_w if gx < 3 else w
                grid[gy, gx] = gray[y0:y1, x0:x1].mean()
        spatial_layout = grid.flatten()  # shape (16,)

        # Object arrangement: normalized centroids + color + area
        frame_area = h * w
        object_arrangement = []
        for obj in frame_objects:
            cx, cy_obj = obj["centroid"]
            bx, by, bw, bh = obj["bbox"]
            area_ratio = (bw * bh) / frame_area if frame_area > 0 else 0.0
            mean_color = obj.get("mean_color", (0, 0, 0))
            # Normalize color to [0,1] if it's in [0,255]
            if isinstance(mean_color, (list, tuple)) and any(c > 1.0 for c in mean_color):
                mean_color = tuple(c / 255.0 for c in mean_color)
            object_arrangement.append({
                "centroid_norm": (cx / w, cy_obj / h),
                "mean_color": tuple(round(c, 3) for c in mean_color),
                "area_ratio": round(area_ratio, 5),
            })

        return SceneSignature(
            color_histogram=color_histogram,
            spatial_layout=spatial_layout,
            object_arrangement=object_arrangement,
            frame_idx=frame_idx,
        )

    def to_vector(self) -> np.ndarray:
        """Flatten signature to a single vector for similarity comparison."""
        return np.concatenate([self.color_histogram, self.spatial_layout])

    def to_dict(self) -> dict:
        return {
            "color_histogram": [round(float(v), 4) for v in self.color_histogram],
            "spatial_layout": [round(float(v), 4) for v in self.spatial_layout],
            "object_arrangement": self.object_arrangement,
            "frame_idx": self.frame_idx,
        }


@dataclass
class SceneRecord:
    """A stored scene in scene memory."""
    scene_id: int
    signature: SceneSignature
    times_seen: int = 1
    confidence: float = 0.5
    last_seen: int = 0
    first_seen: int = 0

    def to_dict(self) -> dict:
        return {
            "scene_id": self.scene_id,
            "times_seen": self.times_seen,
            "confidence": round(self.confidence, 4),
            "last_seen": self.last_seen,
            "first_seen": self.first_seen,
            "object_count": len(self.signature.object_arrangement),
        }


class SceneMemory:
    """
    Scene-level memory: stores and compares scene signatures across frames.

    - `compare()` returns similarity scores against all stored scenes
    - `get_best_match()` returns the top match above a threshold
    - `store()` adds a new scene; `reinforce()` strengthens an existing one
    - Persists across frames within a job; resets per job
    """

    def __init__(self, similarity_threshold: float = 0.6, max_scenes: int = 50):
        self.similarity_threshold = similarity_threshold
        self.max_scenes = max_scenes
        self.scenes: list[SceneRecord] = []
        self._next_id: int = 0

    def compare(self, current: SceneSignature) -> list[dict]:
        """Compare current signature against all stored scenes.

        Returns list of {scene_id, similarity} sorted by similarity descending.
        """
        if not self.scenes:
            return []

        current_vec = current.to_vector()
        results = []
        for record in self.scenes:
            sim = _cosine_similarity(current_vec, record.signature.to_vector())
            # Bonus for object arrangement similarity
            obj_sim = _object_arrangement_similarity(
                current.object_arrangement, record.signature.object_arrangement
            )
            combined = 0.7 * sim + 0.3 * obj_sim
            results.append({
                "scene_id": record.scene_id,
                "similarity": round(float(combined), 4),
                "vector_similarity": round(float(sim), 4),
                "object_similarity": round(float(obj_sim), 4),
            })

        results.sort(key=lambda r: r["similarity"], reverse=True)
        return results

    def get_best_match(self, current: SceneSignature) -> Optional[dict]:
        """Return the best matching scene above threshold, or None."""
        matches = self.compare(current)
        if matches and matches[0]["similarity"] >= self.similarity_threshold:
            record = self._find_record(matches[0]["scene_id"])
            return {
                **matches[0],
                "times_seen": record.times_seen if record else 0,
                "confidence": record.confidence if record else 0,
                "object_arrangement": record.signature.object_arrangement if record else [],
            }
        return None

    def store(self, signature: SceneSignature, frame_idx: int) -> int:
        """Store a new scene. Returns the scene_id."""
        scene_id = self._next_id
        self._next_id += 1
        record = SceneRecord(
            scene_id=scene_id,
            signature=signature,
            times_seen=1,
            confidence=0.5,
            last_seen=frame_idx,
            first_seen=frame_idx,
        )
        self.scenes.append(record)
        # Evict oldest if over capacity
        if len(self.scenes) > self.max_scenes:
            self.scenes.sort(key=lambda s: s.last_seen)
            self.scenes = self.scenes[-self.max_scenes:]
        return scene_id

    def reinforce(self, scene_id: int, frame_idx: int) -> None:
        """Strengthen a recognized scene."""
        record = self._find_record(scene_id)
        if record:
            record.times_seen += 1
            record.confidence = min(1.0, record.confidence + 0.1)
            record.last_seen = frame_idx

    def update_signature(self, scene_id: int, signature: SceneSignature) -> None:
        """Update a scene's signature with a refined version."""
        record = self._find_record(scene_id)
        if record:
            # Blend old and new (weighted toward existing for stability)
            old_vec = record.signature.to_vector()
            new_vec = signature.to_vector()
            blended = 0.7 * old_vec + 0.3 * new_vec
            record.signature.color_histogram = blended[:48]
            record.signature.spatial_layout = blended[48:]
            record.signature.object_arrangement = signature.object_arrangement

    def reset(self) -> None:
        """Clear all scene memory."""
        self.scenes.clear()
        self._next_id = 0

    def snapshot(self) -> list[dict]:
        """Serializable snapshot of all scene records."""
        return [s.to_dict() for s in self.scenes]

    def _find_record(self, scene_id: int) -> Optional[SceneRecord]:
        for s in self.scenes:
            if s.scene_id == scene_id:
                return s
        return None


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors. Returns 0.0 if either is zero."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < 1e-9 or norm_b < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def _object_arrangement_similarity(
    arr_a: list[dict], arr_b: list[dict]
) -> float:
    """Compare two object arrangements by spatial proximity matching.

    Greedy match: for each object in A, find closest in B by normalized
    centroid distance. Average of matched similarities.
    """
    if not arr_a or not arr_b:
        # Both empty = same (no objects either way); one empty = 0
        return 1.0 if (not arr_a and not arr_b) else 0.0

    matched_b: set[int] = set()
    total_sim = 0.0

    for oa in arr_a:
        best_sim = 0.0
        best_idx = -1
        ca = oa["centroid_norm"]
        for bi, ob in enumerate(arr_b):
            if bi in matched_b:
                continue
            cb = ob["centroid_norm"]
            dist = math.sqrt((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2)
            # Max possible dist in normalized space is sqrt(2) ≈ 1.414
            spatial_sim = max(0.0, 1.0 - dist / 1.0)  # generous: 1.0 distance → 0 sim
            if spatial_sim > best_sim:
                best_sim = spatial_sim
                best_idx = bi
        if best_idx >= 0:
            matched_b.add(best_idx)
        total_sim += best_sim

    # Penalize count mismatch: if A has 5 objects and B has 2, unmatched A objects contribute 0
    return total_sim / max(len(arr_a), len(arr_b))
