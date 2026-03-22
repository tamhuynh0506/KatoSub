"""
watermark_detector.py — Temporal Consistency Watermark Detector

Strategy:
  1. Sample frames uniformly across the video (or per scene segment).
  2. Compute pixel-wise standard-deviation map across sampled frames.
     Regions with near-zero std-dev AND non-zero mean are static overlays → watermarks.
  3. Threshold + morphological cleanup → binary mask.
  4. Find connected components → bounding rectangles.
  5. Supports multiple watermarks in a single video by splitting the video into
     temporal segments (scene-change aware) and running detection per segment.

Output:
  List of WatermarkRegion(x, y, w, h, start_frame, end_frame)
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import time


@dataclass
class WatermarkRegion:
    x: int
    y: int
    w: int
    h: int
    start_frame: int   # inclusive
    end_frame: int     # inclusive (total_frames-1 if static throughout)
    confidence: float = 1.0  # 0.0–1.0

    @property
    def box_points(self) -> List[List[int]]:
        """4-corner polygon [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]."""
        return [
            [self.x,         self.y],
            [self.x + self.w, self.y],
            [self.x + self.w, self.y + self.h],
            [self.x,         self.y + self.h],
        ]

    def to_inpaint_boxes(self) -> List[List[List[int]]]:
        """Wrap into the list-of-boxes format expected by AIInpainter."""
        return [self.box_points]


class WatermarkDetector:
    """
    Detects semi-transparent or opaque watermarks via temporal pixel variance.

    Key parameters
    --------------
    sample_rate        : fraction of frames to sample (0.05 = 5 % of all frames)
    std_threshold      : pixels with std-dev below this are 'stable'
    min_brightness     : ignore near-black stable pixels (they're just dark scenes)
    min_area           : minimum watermark area in pixels (filters noise)
    max_watermarks     : safety cap on returned regions
    n_segments         : split video into this many temporal segments to detect
                         watermarks that appear / disappear mid-video
    margin             : extra pixels added around each detected bbox
    """

    def __init__(
        self,
        sample_rate: float = 0.04,
        std_threshold: float = 12.0,
        min_brightness: float = 18.0,
        min_area: int = 400,
        max_watermarks: int = 8,
        n_segments: int = 3,
        margin: int = 6,
    ):
        self.sample_rate = sample_rate
        self.std_threshold = std_threshold
        self.min_brightness = min_brightness
        self.min_area = min_area
        self.max_watermarks = max_watermarks
        self.n_segments = n_segments
        self.margin = margin

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def detect(
        self,
        video_path: str,
        progress_callback=None,
        exclude_bottom_fraction: float = 0.20,
    ) -> List[WatermarkRegion]:
        """
        Main entry: detect all watermark regions in *video_path*.

        exclude_bottom_fraction: skip the bottom N% of frame height to avoid
        treating hardcoded subtitle bars as watermarks.
        """
        def _log(msg):
            if progress_callback:
                progress_callback(msg)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()

        if total_frames < 30:
            _log("⚠ Video too short for watermark detection.")
            return []

        # Subtitle zone exclusion boundary
        excl_top = int(h * (1.0 - exclude_bottom_fraction))

        _log(f"🔍 Watermark Detector: {w}×{h} | {total_frames} frames | {self.n_segments} segments")

        # Split into temporal segments
        seg_size = total_frames // self.n_segments
        segments = []
        for i in range(self.n_segments):
            seg_start = i * seg_size
            seg_end   = (i + 1) * seg_size if i < self.n_segments - 1 else total_frames
            segments.append((seg_start, seg_end))

        all_regions: List[WatermarkRegion] = []

        for seg_idx, (seg_start, seg_end) in enumerate(segments):
            pct = int((seg_idx / self.n_segments) * 100)
            _log(f"Watermark Detection: {pct}% — scanning segment {seg_idx+1}/{self.n_segments} (frames {seg_start}–{seg_end})")

            frames = self._sample_frames(video_path, seg_start, seg_end)
            if len(frames) < 4:
                continue

            mask = self._compute_stability_mask(frames, excl_top, h, w)
            regions = self._mask_to_regions(mask, seg_start, seg_end - 1, h, w)

            if regions:
                _log(f"   Segment {seg_idx+1}: {len(regions)} watermark region(s) found")
            all_regions.extend(regions)

        # Merge regions that are nearly identical across segments (static watermark)
        merged = self._merge_regions(all_regions, total_frames)
        merged = merged[:self.max_watermarks]

        _log(f"✅ Watermark Detection complete: {len(merged)} unique region(s) detected")
        for i, r in enumerate(merged):
            _log(f"   [{i+1}] ({r.x},{r.y}) {r.w}×{r.h} | frames {r.start_frame}→{r.end_frame} | conf={r.confidence:.2f}")

        return merged

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _sample_frames(self, video_path: str, start: int, end: int) -> List[np.ndarray]:
        """Read a uniform subset of frames in [start, end)."""
        count = end - start
        step  = max(1, int(1.0 / self.sample_rate))
        # Cap at 80 frames per segment for speed
        step  = max(step, count // 80)

        cap = cv2.VideoCapture(video_path)
        frames = []
        idx = start
        while idx < end:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                break
            # Downscale to 960-wide for speed while preserving aspect
            if frame.shape[1] > 960:
                scale = 960 / frame.shape[1]
                frame = cv2.resize(frame, (960, int(frame.shape[0] * scale)))
            frames.append(frame.astype(np.float32))
            idx += step
        cap.release()
        return frames

    def _compute_stability_mask(
        self,
        frames: List[np.ndarray],
        excl_top: int,
        orig_h: int,
        orig_w: int,
    ) -> np.ndarray:
        """
        Returns a binary mask (same hw as frames) where stable, bright pixels = 255.
        Works in grayscale for efficiency.
        """
        fh, fw = frames[0].shape[:2]

        # Scale excl_top to match possibly-downscaled frames
        scale_y = fh / orig_h
        excl_top_scaled = int(excl_top * scale_y)

        grays = np.stack([cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                          for f in frames], axis=0)  # (N, H, W)

        mean_map = np.mean(grays, axis=0)   # (H, W)
        std_map  = np.std(grays,  axis=0)   # (H, W)

        # Stable = low variance AND not too dark
        stable_mask = (std_map < self.std_threshold) & (mean_map > self.min_brightness)

        # Exclude the bottom subtitle zone
        stable_mask[excl_top_scaled:, :] = False

        binary = (stable_mask.astype(np.uint8)) * 255

        # Morphological cleanup: close small gaps, remove tiny blobs
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        # --- Subtract the background (large uniform sky/wall areas are also stable)
        # Use edge density: real watermarks live on edges, uniform backgrounds don't.
        # We dilate the edge map so thin logos get included.
        edges = cv2.Canny(mean_map.astype(np.uint8), 30, 80)
        edge_dilated = cv2.dilate(edges, np.ones((11, 11), np.uint8), iterations=2)
        binary = cv2.bitwise_and(binary, edge_dilated)

        # Second morphological pass after edge masking
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        return binary

    def _mask_to_regions(
        self,
        mask: np.ndarray,
        start_frame: int,
        end_frame: int,
        orig_h: int,
        orig_w: int,
    ) -> List[WatermarkRegion]:
        """Find connected components in the mask and return bounding boxes (in original coords)."""
        fh, fw = mask.shape[:2]
        scale_x = orig_w / fw
        scale_y = orig_h / fh

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        regions = []

        for label in range(1, n_labels):  # skip background (0)
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_area:
                continue

            x = int(stats[label, cv2.CC_STAT_LEFT]   * scale_x)
            y = int(stats[label, cv2.CC_STAT_TOP]    * scale_y)
            w = int(stats[label, cv2.CC_STAT_WIDTH]  * scale_x)
            h = int(stats[label, cv2.CC_STAT_HEIGHT] * scale_y)

            # Add margin
            m = self.margin
            x = max(0, x - m);  y = max(0, y - m)
            w = min(orig_w - x, w + 2 * m)
            h = min(orig_h - y, h + 2 * m)

            # Compute confidence: ratio of stable pixels within bbox vs bbox area
            comp_mask = (labels == label).astype(np.uint8)
            conf = float(area) / ((w / scale_x) * (h / scale_y) + 1e-6)
            conf = min(1.0, conf)

            regions.append(WatermarkRegion(x, y, w, h, start_frame, end_frame, confidence=conf))

        # Sort by area descending
        regions.sort(key=lambda r: r.w * r.h, reverse=True)
        return regions

    def _merge_regions(
        self,
        regions: List[WatermarkRegion],
        total_frames: int,
    ) -> List[WatermarkRegion]:
        """
        Merge spatially overlapping regions across segments into single entries
        spanning their full temporal range.  Distinct watermarks (non-overlapping)
        are kept separate.
        """
        if not regions:
            return []

        merged: List[WatermarkRegion] = []

        def iou(a: WatermarkRegion, b: WatermarkRegion) -> float:
            ix1 = max(a.x, b.x);  iy1 = max(a.y, b.y)
            ix2 = min(a.x+a.w, b.x+b.w); iy2 = min(a.y+a.h, b.y+b.h)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2-ix1)*(iy2-iy1)
            union = a.w*a.h + b.w*b.h - inter
            return inter / (union + 1e-6)

        used = [False] * len(regions)
        for i, r in enumerate(regions):
            if used[i]:
                continue
            group = [r]
            used[i] = True
            for j in range(i+1, len(regions)):
                if not used[j] and iou(r, regions[j]) > 0.25:
                    group.append(regions[j])
                    used[j] = True

            # Union bbox
            gx = min(g.x for g in group)
            gy = min(g.y for g in group)
            gx2 = max(g.x+g.w for g in group)
            gy2 = max(g.y+g.h for g in group)
            gs  = min(g.start_frame for g in group)
            ge  = max(g.end_frame   for g in group)
            gc  = float(np.mean([g.confidence for g in group]))
            
            # If a watermark is detected in *most* segments, assume it's static
            # across the entire video to prevent flashing or sudden reappearance.
            if len(group) >= (self.n_segments / 2.0):
                gs = 0
                ge = total_frames - 1

            merged.append(WatermarkRegion(gx, gy, gx2-gx, gy2-gy, gs, ge, gc))

        merged.sort(key=lambda r: r.start_frame)
        return merged
