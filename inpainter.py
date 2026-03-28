"""
inpainter.py — Two-Stage AI Inpainting: LaMa → Stable Diffusion Refinement

Stage 1: LaMa removes the watermark structure (fast, ~50ms per ROI)
Stage 2: Stable Diffusion inpainting refines texture/detail (slower, ~1-3s per ROI)

VRAM Management: On low-VRAM GPUs (≤4 GB), LaMa is unloaded before SD loads
to avoid OOM. Models are cached on disk so reloading is fast.
"""

import cv2
import numpy as np
import torch
import gc
from PIL import Image
from simple_lama_inpainting import SimpleLama


class AIInpainter:
    def __init__(self, use_sd_refine=True):
        self._lama_wrapper = None
        self._sd_pipe = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_sd_refine = use_sd_refine

        # SD refinement parameters
        self.sd_model_id = "stabilityai/stable-diffusion-2-inpainting"
        self.sd_strength = 0.35      # Low strength: preserve LaMa structure, refine texture
        self.sd_steps = 15           # Few steps for speed
        self.sd_guidance = 7.5
        self.sd_prompt = "clean background, no text, no watermark, seamless, high quality"
        self.sd_negative = "text, letters, watermark, logo, blurry, artifacts, distorted"

    # ─── Model Lifecycle ──────────────────────────────────────────────────

    def _ensure_lama(self):
        """Lazy-load LaMa model onto GPU."""
        if self._lama_wrapper is None:
            self._lama_wrapper = SimpleLama()
            self.model = self._lama_wrapper.model
            self.model.eval()
            self.model.to(self._device)

    def _unload_lama(self):
        """Free LaMa from VRAM."""
        if self._lama_wrapper is not None:
            self.model.cpu()
            del self.model
            del self._lama_wrapper
            self._lama_wrapper = None
            self.model = None
            torch.cuda.empty_cache()
            gc.collect()

    def _ensure_sd(self):
        """Lazy-load Stable Diffusion inpainting pipeline onto GPU (float16)."""
        if self._sd_pipe is None:
            from diffusers import StableDiffusionInpaintPipeline

            print("Loading Stable Diffusion inpainting model (first run downloads ~5 GB)...")
            self._sd_pipe = StableDiffusionInpaintPipeline.from_pretrained(
                self.sd_model_id,
                torch_dtype=torch.float16,
                safety_checker=None,
                requires_safety_checker=False,
            )
            self._sd_pipe.to(self._device)
            self._sd_pipe.set_progress_bar_config(disable=True)

            # Memory optimizations
            if hasattr(self._sd_pipe, "enable_attention_slicing"):
                self._sd_pipe.enable_attention_slicing()

    def _unload_sd(self):
        """Free Stable Diffusion from VRAM."""
        if self._sd_pipe is not None:
            self._sd_pipe.to("cpu")
            del self._sd_pipe
            self._sd_pipe = None
            torch.cuda.empty_cache()
            gc.collect()

    # ─── Stage 2: SD Refinement ───────────────────────────────────────────

    def _sd_refine_roi(self, roi_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Run Stable Diffusion inpainting on a single ROI to refine detail.

        Args:
            roi_bgr: BGR numpy array from LaMa output
            mask: uint8 mask (255=inpaint, 0=keep)

        Returns:
            Refined BGR numpy array, same size as input
        """
        h, w = roi_bgr.shape[:2]

        # SD requires images sized to multiples of 8, and works best at 512x512
        # Scale to fit within 512 while maintaining aspect ratio
        max_dim = 512
        scale = min(max_dim / w, max_dim / h, 1.0)
        new_w = int(w * scale) // 8 * 8  # round to multiple of 8
        new_h = int(h * scale) // 8 * 8
        new_w = max(8, new_w)
        new_h = max(8, new_h)

        # Prepare PIL images for diffusers
        roi_rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(roi_rgb).resize((new_w, new_h), Image.LANCZOS)
        mask_pil = Image.fromarray(mask).resize((new_w, new_h), Image.NEAREST)

        with torch.no_grad():
            result = self._sd_pipe(
                prompt=self.sd_prompt,
                negative_prompt=self.sd_negative,
                image=img_pil,
                mask_image=mask_pil,
                height=new_h,
                width=new_w,
                strength=self.sd_strength,
                num_inference_steps=self.sd_steps,
                guidance_scale=self.sd_guidance,
            ).images[0]

        # Resize back to original dimensions
        result_np = np.array(result.resize((w, h), Image.LANCZOS))
        result_bgr = cv2.cvtColor(result_np, cv2.COLOR_RGB2BGR)
        return result_bgr

    # ─── Main Entry Point ─────────────────────────────────────────────────

    def inpaint_batch(self, frames, batch_boxes):
        """
        Inpaint a batch of frames.
        Wraps single-frame calls sequentially (LaMa batching is slower on most GPUs).
        """
        results = []
        for f, b in zip(frames, batch_boxes):
            results.append(self.inpaint_frame(f, b))
        return results

    def inpaint_frame(self, frame, boxes):
        """
        Two-stage inpainting on a single frame:
          Stage 1: LaMa removes watermark structure
          Stage 2: SD refines texture/detail (optional)

        Processes each box individually to prevent massive ROI memory blowouts
        when watermarks are far apart on the screen.
        """
        if not boxes:
            return frame

        # ── Stage 1: LaMa ──
        self._ensure_lama()
        h_f, w_f = frame.shape[:2]

        # Collect per-box data for potential SD refinement
        box_data = []  # list of (y_start, y_end, x_start, x_end, mask)

        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)

            # Larger padding gives LaMa more background context
            pad = 60
            x_start = max(0, int(x_min) - pad)
            y_start = max(0, int(y_min) - pad)
            x_end = min(w_f, int(x_max) + pad)
            y_end = min(h_f, int(y_max) + pad)

            roi = frame[y_start:y_end, x_start:x_end]
            if roi.size == 0:
                continue

            # Surgical mask: bright text pixels inside the bounding box
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            _, mask_text = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)

            box_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            roi_pts = np.array([[p[0] - x_start, p[1] - y_start] for p in box], dtype=np.int32)
            cv2.fillPoly(box_mask, [roi_pts], 255)

            mask = cv2.bitwise_and(mask_text, box_mask)
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)

            # Run LaMa inpainting
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result_pil = self._lama_wrapper(roi_rgb, mask)
                lama_result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                lama_result = lama_result[:(y_end - y_start), :(x_end - x_start)]

                # Feather-blend LaMa result into frame
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

                # Save data for SD refinement
                box_data.append((y_start, y_end, x_start, x_end, mask))

            except Exception as e:
                print(f"LaMa inpainting error on box: {e}")

        # ── Stage 2: Stable Diffusion Refinement ──
        if self.use_sd_refine and box_data:
            # Unload LaMa to free VRAM for SD
            self._unload_lama()
            self._ensure_sd()

            for (y_start, y_end, x_start, x_end, mask) in box_data:
                try:
                    # Extract the LaMa-cleaned ROI
                    lama_roi = frame[y_start:y_end, x_start:x_end].copy()
                    roi_h = y_end - y_start
                    roi_w = x_end - x_start

                    # SD refine
                    sd_result = self._sd_refine_roi(lama_roi, mask[:roi_h, :roi_w])
                    sd_result = sd_result[:roi_h, :roi_w]

                    # Feather-blend SD result (only in masked area)
                    weight = mask[:roi_h, :roi_w].astype(np.float32) / 255.0
                    weight = cv2.GaussianBlur(weight, (0, 0), 5.0)
                    weight = np.clip(weight, 0.0, 1.0)
                    weight3 = weight[:, :, np.newaxis]

                    current_roi = frame[y_start:y_end, x_start:x_end].astype(np.float32)
                    refined_roi = sd_result.astype(np.float32)
                    blended = current_roi * (1.0 - weight3) + refined_roi * weight3
                    frame[y_start:y_end, x_start:x_end] = np.clip(blended, 0, 255).astype(np.uint8)

                except Exception as e:
                    print(f"SD refinement error on box: {e}")
                    # LaMa result is already in the frame, so we just continue

            # Unload SD and reload LaMa for next frame
            self._unload_sd()

        return frame
