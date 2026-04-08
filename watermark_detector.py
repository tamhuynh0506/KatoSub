"""
watermark_detector.py  Temporal Consistency Watermark Detector

Strategy:
  1. Sample frames uniformly across the video (or per scene segment).
  2. Compute pixel-wise standard-deviation map across sampled frames.
     Regions with near-zero std-dev AND non-zero mean are static overlays  watermarks.
  3. Threshold + morphological cleanup  binary mask.
  4. Find connected components  bounding rectangles.
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
    confidence: float = 1.0  # 0.01.0

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
        sample_rate: float = 0.06,
        std_threshold: float = 12.0,
        min_brightness: float = 18.0,
        min_area: int = 250,
        max_watermarks: int = 12,
        n_segments: int = 6,
        margin: int = 8,
    ):
        self.sample_rate = sample_rate
        self.std_threshold = std_threshold
        self.min_brightness = min_brightness
        self.min_area = min_area
        self.max_watermarks = max_watermarks
        self.n_segments = n_segments
        self.margin = margin
        self.max_watermarks = 5 # Force max 5 for speed
        self.ocr_engine: Optional[object] = None
        self.watermark_keywords = [
            '.rest', '.lo', '.tv', '.com', '.net', '.org', '.me',
            'dramacool', 'sub', 'myasian', 'viewasian', 'kissasian',
            'english', 'drama', '.site', '.icu', '.vip'
        ]

    # 
    # Public API
    # 

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
            _log(" Video too short for watermark detection.")
            return []

        # Subtitle zone exclusion boundary
        excl_top = int(h * (1.0 - exclude_bottom_fraction))

        _log(f" Watermark Detector: {w}{h} | {total_frames} frames | {self.n_segments} segments")

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
            _log(f"Watermark Detection: {pct}%  scanning epoch {seg_idx+1}/{self.n_segments} (frames {seg_start}{seg_end})")

            frames = self._sample_frames(video_path, seg_start, seg_end)
            if len(frames) < 4:
                continue

            mask, mean_map = self._compute_stability_mask(frames, excl_top, h, w)
            
            # --- OCR Seeding: sample up to 5 evenly-spaced frames per epoch ---
            n_ocr = min(5, len(frames))
            ocr_sample = [frames[i] for i in np.linspace(0, len(frames) - 1, n_ocr, dtype=int)]
            seed_regions = self._get_ocr_seeds(ocr_sample, h, w)
            if seed_regions:
                _log(f"   Epoch {seg_idx+1}: {len(seed_regions)} text seed(s) identified across {n_ocr} frames")
            
            # Stability regions
            stab_regions = self._mask_to_regions(mask, seg_start, seg_end - 1, h, w, mean_map=mean_map)
            
            # Combine Seeds + Stability
            final_epoch_regions = self._merge_seeds_and_stability(seed_regions, stab_regions, seg_start, seg_end - 1)
            
            if final_epoch_regions:
                _log(f"   Epoch {seg_idx+1}: {len(final_epoch_regions)} candidate region(s) validated")
            all_regions.extend(final_epoch_regions)

        # Merge regions that are nearly identical across segments (static watermark)
        merged = self._merge_regions(all_regions, total_frames)
        # Sort by confidence descending, then by area
        merged.sort(key=lambda r: (r.confidence, r.w * r.h), reverse=True)
        merged = merged[:self.max_watermarks]

        _log(f" Watermark Detection complete: {len(merged)} unique region(s) detected")
        for i, r in enumerate(merged):
            _log(f"   [{i+1}] ({r.x},{r.y}) {r.w}{r.h} | frames {r.start_frame}{r.end_frame} | conf={r.confidence:.2f}")

        return merged

    # 
    # Internal helpers
    # 

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns (binary_mask, mean_map).
        Stable, bright pixels = 255 in binary_mask.
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
        
        return binary, mean_map

    def _mask_to_regions(
        self,
        mask: np.ndarray,
        start_frame: int,
        end_frame: int,
        orig_h: int,
        orig_w: int,
        mean_map: Optional[np.ndarray] = None,
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

            # --- Refined Hybrid Filter: OCR + Broad Edge Heuristic ---
            is_valid = False
            
            # 1. Broad Edge heuristic (fast)
            if self._is_on_edge(x, y, w, h, orig_w, orig_h):
                is_valid = True
            
            # 2. OCR verification (deep)
            if not is_valid and mean_map is not None:
                px = stats[label, cv2.CC_STAT_LEFT]
                py = stats[label, cv2.CC_STAT_TOP]
                pw = stats[label, cv2.CC_STAT_WIDTH]
                ph = stats[label, cv2.CC_STAT_HEIGHT]
                patch = mean_map[py:py+ph, px:px+pw]
                if self._check_ocr(patch):
                    is_valid = True

            if not is_valid:
                continue

            # Compute confidence: ratio of stable pixels within bbox vs bbox area
            comp_mask = (labels == label).astype(np.uint8)
            conf = float(area) / ((w / scale_x) * (h / scale_y) + 1e-6)
            conf = min(1.0, conf)

            regions.append(WatermarkRegion(x, y, w, h, start_frame, end_frame, confidence=conf))

        # Sort by area descending
        regions.sort(key=lambda r: r.w * r.h, reverse=True)
        return regions

    # 
    # OCR & Heuristic Logic
    # 

    def _get_ocr(self):
        """Lazy initialization of EasyOCR (mimicking PaddleOCR interface to avoid DLL conflicts)."""
        if self.ocr_engine is None:
            try:
                import easyocr
                class EasyOCRWrapper:
                    def __init__(self):
                        self.reader = easyocr.Reader(['en'], gpu=True)
                    def ocr(self, img, cls=False):
                        # EasyOCR: [[box], text, conf]
                        # PaddleOCR expects: [ [[box], (text, conf)], ... ]
                        results = self.reader.readtext(img)
                        formatted = []
                        for res in results:
                            formatted.append([res[0], (res[1], res[2])])
                        return [formatted] if formatted else None

                self.ocr_engine = EasyOCRWrapper()
                print("DEBUG: WatermarkDetector using EasyOCR (PyTorch) to avoid DLL conflict.")
            except Exception as e:
                print(f"DEBUG: EasyOCR init failed: {e}. Falling back to DummyOCR.")
                class DummyOCR:
                    def ocr(self, *args, **kwargs): return []
                self.ocr_engine = DummyOCR()
        return self.ocr_engine

    def _get_ocr_seeds(self, frames_sample: List[np.ndarray], orig_h: int, orig_w: int) -> List[WatermarkRegion]:
        """
        Run OCR on each frame in *frames_sample* and return de-duplicated
        text-based watermark candidates.

        Running on multiple frames per epoch lets us catch watermarks that
        move, change text, or appear only intermittently.
        """
        all_candidates: List[WatermarkRegion] = []

        for frame in frames_sample:
            fh, fw = frame.shape[:2]
            scale_x = orig_w / fw
            scale_y = orig_h / fh

            try:
                # Ensure uint8 BGR
                frm = frame
                if frm.dtype != np.uint8:
                    frm = frm.astype(np.uint8)
                if len(frm.shape) == 2:
                    frm = cv2.cvtColor(frm, cv2.COLOR_GRAY2BGR)

                ocr = self._get_ocr()
                result = ocr.ocr(frm, cls=False)
                if result and result[0]:
                    for line in result[0]:
                        box  = line[0]   # [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]
                        conf = line[1][1]

                        if conf < 0.4:
                            continue

                        # Convert to original video coordinates
                        x1 = min(p[0] for p in box); x2 = max(p[0] for p in box)
                        y1 = min(p[1] for p in box); y2 = max(p[1] for p in box)
                        ox = int(x1 * scale_x); oy = int(y1 * scale_y)
                        ow = int((x2 - x1) * scale_x); oh = int((y2 - y1) * scale_y)

                        # Add OCR specific padding to cover text shadows/outlines
                        pm = 15  # text shadow margin
                        ox = max(0, ox - pm)
                        oy = max(0, oy - pm)
                        ow = min(orig_w - ox, ow + 2*pm)
                        oh = min(orig_h - oy, oh + 2*pm)

                        # Skip subtitle exclusion zone: bottom-25% AND centre-60% width.
                        # Use box centre for horizontal check so wide caption bars
                        # that start slightly outside 20% are still excluded.
                        cx = ox + ow // 2
                        if oy > orig_h * 0.75 and (orig_w * 0.2 < cx < orig_w * 0.8):
                            continue

                        all_candidates.append(WatermarkRegion(ox, oy, ow, oh, 0, 0, confidence=conf))
            except Exception:
                pass

        # De-duplicate: collapse candidates whose bounding boxes overlap heavily
        # (IOU > 0.4) by keeping the one with the highest confidence.
        return self._dedup_ocr_candidates(all_candidates)

    def _dedup_ocr_candidates(self, candidates: List[WatermarkRegion]) -> List[WatermarkRegion]:
        """Collapse overlapping OCR detections, keeping highest-confidence box."""
        if not candidates:
            return []

        # Sort by confidence descending so we greedily keep the best
        candidates = sorted(candidates, key=lambda r: r.confidence, reverse=True)
        kept: List[WatermarkRegion] = []

        def iou(a: WatermarkRegion, b: WatermarkRegion) -> float:
            ix1 = max(a.x, b.x);       iy1 = max(a.y, b.y)
            ix2 = min(a.x + a.w, b.x + b.w); iy2 = min(a.y + a.h, b.y + b.h)
            if ix2 <= ix1 or iy2 <= iy1:
                return 0.0
            inter = (ix2 - ix1) * (iy2 - iy1)
            union = a.w * a.h + b.w * b.h - inter
            return inter / (union + 1e-6)

        for c in candidates:
            if all(iou(c, k) < 0.4 for k in kept):
                kept.append(c)

        return kept

    def _check_ocr(self, patch: np.ndarray) -> bool:
        """Return True if OCR detects text in the patch."""
        if patch.size == 0: return False
        try:
            if patch.dtype != np.uint8:
                patch = patch.astype(np.uint8)
            if len(patch.shape) == 2:
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            
            ocr = self._get_ocr()
            result = ocr.ocr(patch, cls=False)
            if result and result[0]:
                return True
        except Exception:
            pass
        return False

    def _is_on_edge(self, x, y, w, h, frame_w, frame_h, strict: bool = False) -> bool:
        """Universal Detection: Returns True if outside the Subtitle Exclusion Zone.

        Uses box CENTER for the horizontal check so that wide caption bars
        (which may start slightly outside the 20% margin) are still excluded.
        """
        # Subtitle Exclusion Zone: bottom 25% height AND centre 60% width
        cx = x + w // 2
        is_subtitle_h = y > frame_h * 0.75
        is_subtitle_w = (frame_w * 0.2 < cx < frame_w * 0.8)

        if is_subtitle_h and is_subtitle_w:
            return False

        return True  # Universal range!

    def _merge_seeds_and_stability(
        self, 
        seeds: List[WatermarkRegion], 
        stab: List[WatermarkRegion],
        start: int,
        end: int
    ) -> List[WatermarkRegion]:
        """Merge OCR-seeded regions with stability-detected regions."""
        if not seeds: return stab
        if not stab: 
            # If only seeds exist, mark them with current epoch frames
            for s in seeds:
                s.start_frame = start
                s.end_frame = end
            return seeds
            
        final = []
        used_stab = [False] * len(stab)
        
        for s in seeds:
            s.start_frame = start
            s.end_frame = end
            matched = False
            for i, st in enumerate(stab):
                ix1 = max(s.x, st.x); iy1 = max(s.y, st.y)
                ix2 = min(s.x+s.w, st.x+st.w); iy2 = min(s.y+s.h, st.y+st.h)
                if ix2 > ix1 and iy2 > iy1:
                    # TIGHTEN: If OCR found it, use OCR box but Expand slightly to cover anti-aliasing
                    # Instead of Union with the noisy blob.
                    final.append(WatermarkRegion(s.x-2, s.y-2, s.w+4, s.h+4, start, end, max(s.confidence, st.confidence)))
                    used_stab[i] = True
                    matched = True
                    break
            if not matched:
                final.append(s)
        
        # Add unmatched stability regions ONLY if they are small (potential logos)
        for i, st in enumerate(stab):
            if not used_stab[i]:
                if st.w * st.h < 50000: # Tight limit for non-text blobs
                    final.append(st)
                
        return final

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
            
            # --- New: False Positive Filter for Large Static Scenes ---
            # If a region is huge (>15% of frame) and only in 1 epoch, it's likely a static scene.
            # 1280 * 720 * 0.15 ~= 138,000 pixels.
            if len(group) == 1 and (gx2-gx) * (gy2-gy) > 130000:
                continue

            # If a watermark is detected in *many* epochs (at least 50%), assume
            # it is static across the entire video to prevent flashing.
            if len(group) >= (self.n_segments / 2.0):
                gs = 0
                ge = total_frames - 1
            elif len(group) == 1:
                # If only one epoch, slightly expand range to cover transition
                gs = max(0, group[0].start_frame - 10)
                ge = min(total_frames - 1, group[0].end_frame + 10)

            merged.append(WatermarkRegion(gx, gy, gx2-gx, gy2-gy, gs, ge, gc))

        merged.sort(key=lambda r: r.start_frame)
        return merged
