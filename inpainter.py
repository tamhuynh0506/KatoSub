"""
inpainter.py — Two-Stage AI Inpainting: LaMa → ProPainter Refinement

Stage 1: LaMa removes the watermark structure (fast, ~50ms per ROI)
Stage 2: ProPainter inpainting refines texture/detail with temporal consistency (slower, ~100-300ms per frame)

VRAM Management: On low-VRAM GPUs, LaMa is unloaded before ProPainter loads.
Models are cached on disk so reloading is fast.
"""

import cv2
import numpy as np
import torch
import gc
import sys
import os
from PIL import Image
from typing import List, Optional

# Add ProPainter module to path
repo_path = os.path.abspath("propainter_module")
if repo_path not in sys.path:
    sys.path.insert(0, repo_path)

from simple_lama_inpainting import SimpleLama

try:
    from model.modules.flow_comp_raft import RAFT_bi
    from model.recurrent_flow_completion import RecurrentFlowCompleteNet
    from model.propainter import InpaintGenerator
    from core.utils import to_tensors
    PROPAINTER_AVAILABLE = True
except ImportError:
    PROPAINTER_AVAILABLE = False


class AIInpainter:
    def __init__(self, use_propainter=True):
        self._lama_wrapper = None
        self._propainter_wrapper = None
        self.lama_model = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_propainter = use_propainter and PROPAINTER_AVAILABLE

        # ProPainter parameters
        self.pp_neighbor_length = 10
        self.pp_ref_stride = 10
        self.pp_subvideo_length = 80
        self.pp_raft_iter = 20
        self.pp_fp16 = True if torch.cuda.is_available() else False

    # ─── Model Lifecycle ──────────────────────────────────────────────────

    def _ensure_lama(self):
        """Lazy-load LaMa model onto GPU."""
        if self._lama_wrapper is None:
            self._lama_wrapper = SimpleLama()
            if self._lama_wrapper is not None:
                self.lama_model = self._lama_wrapper.model
                self.lama_model.eval()
        
        if self.lama_model is not None:
            self.lama_model.to(self._device)

    def _unload_lama(self):
        """Free LaMa from VRAM but keep in memory."""
        if self.lama_model is not None:
            self.lama_model.cpu()
            torch.cuda.empty_cache()

    def _ensure_propainter(self):
        """Lazy-load ProPainter pipeline."""
        if self._propainter_wrapper is None and PROPAINTER_AVAILABLE:
            print("Loading ProPainter models (RAFT + Flow + Transformer)...")
            self._propainter_wrapper = ProPainterWrapper(self._device)

    def _unload_propainter(self):
        """Free ProPainter from VRAM."""
        if self._propainter_wrapper is not None:
            self._propainter_wrapper.to_cpu()
            torch.cuda.empty_cache()
            gc.collect()

    # ─── ProPainter Sequence Inpainting ───────────────────────────────────

    def inpaint_sequence(self, frames: List[np.ndarray], boxes_per_frame: List[List]):
        """
        Inpaint a sequence of frames using ProPainter for temporal consistency.
        """
        if not any(boxes_per_frame):
            return frames

        if self.use_propainter:
            self._unload_lama()
            self._ensure_propainter()
            
            if self._propainter_wrapper is None:
                return [self.inpaint_frame(f, b) for f, b in zip(frames, boxes_per_frame)]

            # --- High Precision Temporal Masking ---
            # Calculate a median frame for the entire chunk to identify static watermark pixels
            if len(frames) > 5:
                chunk_stack = np.stack(frames, axis=0)
                median_frame = np.median(chunk_stack, axis=0).astype(np.uint8)
            else:
                median_frame = None

            # Generate masks for ProPainter
            masks = []
            for frame_idx, (frame, boxes) in enumerate(zip(frames, boxes_per_frame)):
                h, w = frame.shape[:2]
                mask = np.zeros((h, w), dtype=np.uint8)
                if boxes:
                    for box in boxes:
                        pts = np.array(box, dtype=np.int32)
                        
                        # 1. Base Brightness Mask (Lower threshold to catch semi-transparent text)
                        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        _, mask_bright = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
                        
                        # 2. Edge-Aware Mask (Canny + Gradient)
                        # This catches the outlines and shadows of the text
                        edges = cv2.Canny(gray, 50, 150)
                        kernel_edge = np.ones((3, 3), np.uint8)
                        mask_edges = cv2.dilate(edges, kernel_edge, iterations=1)
                        
                        # 3. Temporal Stability Mask (If median is available)
                        # We look for pixels that are close to the median (static) 
                        # but have high brightness or edges.
                        if median_frame is not None:
                            diff = cv2.absdiff(frame, median_frame)
                            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                            # Pixels that ARE NOT changing (low diff) but are part of the box
                            mask_static = cv2.threshold(diff_gray, 15, 255, cv2.THRESH_BINARY_INV)[1]
                        else:
                            mask_static = 255 * np.ones((h, w), dtype=np.uint8)

                        # Bounding box polygon (Expand slightly to catch "halo" of semi-transparent text)
                        box_poly = np.zeros((h, w), dtype=np.uint8)
                        cv2.fillPoly(box_poly, [pts], 255)
                        # Dilate the box poly just a bit so we don't clip the surgical mask
                        box_poly = cv2.dilate(box_poly, np.ones((5, 5), np.uint8), iterations=1)
                        
                        # Combined Surgical Mask
                        # (Bright OR Edges) AND within bounding box AND stable(static)
                        mask_combined = cv2.bitwise_or(mask_bright, mask_edges)
                        mask_final = cv2.bitwise_and(mask_combined, box_poly)
                        mask_final = cv2.bitwise_and(mask_final, mask_static)
                        
                        # Aggressive dilation to cover the "halo" of semi-transparent text
                        kernel_dil = np.ones((11, 11), np.uint8)
                        mask_final = cv2.dilate(mask_final, kernel_dil, iterations=1)
                        
                        # Smooth the binary mask slightly to help inpainter blend
                        # (Optional: most models prefer binary, but it reduces aliasing)
                        mask_final = cv2.GaussianBlur(mask_final, (3, 3), 0)
                        _, mask_final = cv2.threshold(mask_final, 5, 255, cv2.THRESH_BINARY)
                        
                        mask = cv2.bitwise_or(mask, mask_final)
                masks.append(mask)

            try:
                results_rgb = self._propainter_wrapper.inpaint(
                    [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames],
                    masks,
                    fp16=self.pp_fp16,
                    subvideo_length=self.pp_subvideo_length,
                    neighbor_length=self.pp_neighbor_length,
                    ref_stride=self.pp_ref_stride
                )
                return [cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in results_rgb]
            except Exception as e:
                print(f"ProPainter sequence error: {e}")
                return [self.inpaint_frame(f, b) for f, b in zip(frames, boxes_per_frame)]

        return [self.inpaint_frame(f, b) for f, b in zip(frames, boxes_per_frame)]

    # ─── Main Entry Point (Single Frame/Backward Compat) ──────────────────

    def inpaint_batch(self, frames, batch_boxes):
        """
        Inpaint a batch of frames.
        If using ProPainter, treats as a sequence for consistency.
        """
        if self.use_propainter:
            return self.inpaint_sequence(frames, batch_boxes)
        
        results = []
        for f, b in zip(frames, batch_boxes):
            results.append(self.inpaint_frame(f, b))
        return results

    def inpaint_frame(self, frame, boxes):
        """
        Single-stage LaMa inpainting on a single frame.
        """
        if not boxes:
            return frame

        # ── Stage 1: LaMa ──
        self._ensure_lama()
        h_f, w_f = frame.shape[:2]

        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)

            pad = 60
            x_start = max(0, int(x_min) - pad)
            y_start = max(0, int(y_min) - pad)
            x_end = min(w_f, int(x_max) + pad)
            y_end = min(h_f, int(y_max) + pad)

            roi = frame[y_start:y_end, x_start:x_end]
            if roi.size == 0:
                continue

            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask_text = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)

            box_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            roi_pts = np.array([[p[0] - x_start, p[1] - y_start] for p in box], dtype=np.int32)
            cv2.fillPoly(box_mask, [roi_pts], 255)

            mask = cv2.bitwise_and(mask_text, box_mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result_pil = self._lama_wrapper(roi_rgb, mask)
                lama_result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                lama_result = lama_result[:(y_end - y_start), :(x_end - x_start)]

                roi_h = y_end - y_start
                roi_w = x_end - x_start
                weight = mask[:roi_h, :roi_w].astype(np.float32) / 255.0
                weight = cv2.GaussianBlur(weight, (0, 0), 3.0)
                weight = np.clip(weight, 0.0, 1.0)
                weight3 = weight[:, :, np.newaxis]

                original_roi = frame[y_start:y_end, x_start:x_end].astype(np.float32)
                inpainted_roi = lama_result.astype(np.float32)
                blended = original_roi * (1.0 - weight3) + inpainted_roi * weight3
                frame[y_start:y_end, x_start:x_end] = np.clip(blended, 0, 255).astype(np.uint8)

            except Exception as e:
                print(f"LaMa inpainting error on box: {e}")

        return frame


class ProPainterWrapper:
    """
    Wrapper for the ProPainter model and its dependencies (RAFT, Flow Completion).
    """
    def __init__(self, device):
        self.device = device
        
        # RAFT
        raft_path = os.path.join("weights", "raft-things.pth")
        self.raft = RAFT_bi(raft_path, self.device)
        
        # Flow Completion
        flow_path = os.path.join("weights", "recurrent_flow_completion.pth")
        self.flow_net = RecurrentFlowCompleteNet(flow_path)
        for p in self.flow_net.parameters():
            p.requires_grad = False
        self.flow_net.to(self.device).eval()
        
        # ProPainter Transformer
        pp_path = os.path.join("weights", "ProPainter.pth")
        self.model = InpaintGenerator(model_path=pp_path).to(self.device).eval()

    def to_cpu(self):
        self.flow_net.cpu()
        self.model.cpu()
        if hasattr(self, 'raft'):
            # RAFT internally moves itself, but let's be safe
            pass

    def inpaint(self, frames_rgb, masks, fp16=True, subvideo_length=80, neighbor_length=10, ref_stride=10):
        h, w = frames_rgb[0].shape[:2]
        ph, pw = (h // 8) * 8, (w // 8) * 8
        if h != ph or w != pw:
            frames_rgb = [cv2.resize(f, (pw, ph)) for f in frames_rgb]
            masks = [cv2.resize(m, (pw, ph), interpolation=cv2.INTER_NEAREST) for m in masks]

        frames_t = to_tensors()(frames_rgb).unsqueeze(0).to(self.device) * 2 - 1
        masks_t = to_tensors()(masks).unsqueeze(0).to(self.device)
        video_length = frames_t.size(1)

        with torch.no_grad():
            gt_flows_bi = self.raft(frames_t, iters=20)
            
            if fp16:
                frames_t, masks_t = frames_t.half(), masks_t.half()
                gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
                self.flow_net.half()
                self.model.half()

            pred_flows_bi, _ = self.flow_net.forward_bidirect_flow(gt_flows_bi, masks_t)
            pred_flows_bi = self.flow_net.combine_flow(gt_flows_bi, pred_flows_bi, masks_t)

            masked_frames = frames_t * (1 - masks_t)
            prop_imgs, updated_local_masks = self.model.img_propagation(
                masked_frames, pred_flows_bi, masks_t, 'nearest'
            )
            updated_frames = frames_t * (1 - masks_t) + prop_imgs.view(1, video_length, 3, ph, pw) * masks_t
            updated_masks = updated_local_masks.view(1, video_length, 1, ph, pw)

            comp_frames = [None] * video_length
            neighbor_stride = neighbor_length // 2
            
            for f in range(0, video_length, neighbor_stride):
                neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
                ref_ids = [i for i in range(0, video_length, ref_stride) if i not in neighbor_ids]
                
                selected_imgs = updated_frames[:, neighbor_ids + ref_ids]
                selected_masks = masks_t[:, neighbor_ids + ref_ids]
                selected_update_masks = updated_masks[:, neighbor_ids + ref_ids]
                selected_flows = (pred_flows_bi[0][:, neighbor_ids[:-1]], pred_flows_bi[1][:, neighbor_ids[:-1]])
                
                pred_img = self.model(selected_imgs, selected_flows, selected_masks, selected_update_masks, len(neighbor_ids))
                pred_img = (pred_img.view(-1, 3, ph, pw) + 1) / 2
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).float().numpy() * 255
                
                for i, idx in enumerate(neighbor_ids):
                    out_frame = pred_img[i].astype(np.uint8)
                    if comp_frames[idx] is None:
                        comp_frames[idx] = out_frame
                    else:
                        comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5 + out_frame.astype(np.float32) * 0.5).astype(np.uint8)

        if h != ph or w != pw:
            comp_frames = [cv2.resize(f, (w, h)) for f in comp_frames]
            
        return comp_frames
