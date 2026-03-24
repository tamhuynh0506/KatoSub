import cv2
import numpy as np
import torch
from simple_lama_inpainting import SimpleLama
from simple_lama_inpainting.utils.util import pad_img_to_modulo

class AIInpainter:
    def __init__(self):
        self._lama_wrapper = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_lama(self):
        if self._lama_wrapper is None:
            # We initialize the wrapper to let it handle model downloading/loading
            self._lama_wrapper = SimpleLama()
            # But we access the underlying model for batching
            self.model = self._lama_wrapper.model
            self.model.eval()
            self.model.to(self._device)

    def _prepare_tensors(self, frames, batch_boxes):
        """
        Custom preparation for a batch of frames and boxes to maximize throughput.
        """
        img_tensors = []
        mask_tensors = []
        
        for frame, boxes in zip(frames, batch_boxes):
            # 1. Create mask
            mask = np.zeros(frame.shape[:2], dtype=np.uint8)
            for box in boxes:
                pts = np.array(box, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)
            
            # Dilate mask slightly
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # 2. Convert to RGB and normalized float (CHW)
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            mask_norm = mask.astype(np.float32) / 255.0
            
            img_chw = np.transpose(img_rgb, (2, 0, 1))
            mask_chw = mask_norm[np.newaxis, ...]
            
            # 3. Pad to modulo 8 as required by LaMa
            img_padded = pad_img_to_modulo(img_chw, 8)
            mask_padded = pad_img_to_modulo(mask_chw, 8)
            
            img_tensors.append(torch.from_numpy(img_padded))
            mask_tensors.append(torch.from_numpy(mask_padded))
            
        # Stack into batch [B, C, H, W]
        imgs = torch.stack(img_tensors).to(self._device)
        masks = (torch.stack(mask_tensors).to(self._device) > 0).float()
        
        return imgs, masks

    def inpaint_batch(self, frames, batch_boxes):
        """
        Inpaint a batch of frames. 
        Note: Batching is significantly slower for LaMa on most GPUs, 
        so we mostly use this as a wrapper for single-frame calls.
        """
        results = []
        for f, b in zip(frames, batch_boxes):
            results.append(self.inpaint_frame(f, b))
        return results

    def inpaint_frame(self, frame, boxes):
        """
        Inpaint a single frame with selective cropping for performance.
        Now processes each box individually to prevent massive ROI memory blowouts
        when watermarks are far apart on the screen.
        """
        if not boxes:
            return frame
        
        self._ensure_lama()
        
        # Copy the frame so we can iteratively update it
        # We process each box sequentially.
        h_f, w_f = frame.shape[:2]
        
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            
            x_min, y_min = np.min(pts, axis=0)
            x_max, y_max = np.max(pts, axis=0)
            
            # Larger padding gives LaMa more background context → better reconstruction
            pad = 60
            x_start = max(0, int(x_min) - pad)
            y_start = max(0, int(y_min) - pad)
            x_end = min(w_f, int(x_max) + pad)
            y_end = min(h_f, int(y_max) + pad)
            
            # 2. Extract the cropped ROI
            roi = frame[y_start:y_end, x_start:x_end]
            if roi.size == 0:
                continue
                
            # 3. Create a SURGICAL mask for the ROI
            # Instead of a solid rectangle, we find the actual bright text pixels.
            # This preserves faces/objects that are visible BETWEEN the letters.
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            
            # Threshold to find bright text (watermarks are usually white/light gray)
            _, mask_text = cv2.threshold(gray_roi, 160, 255, cv2.THRESH_BINARY)
            
            # Create a box mask to constrain the thresholding to just the watermark area
            box_mask = np.zeros(roi.shape[:2], dtype=np.uint8)
            roi_pts = np.array([[p[0] - x_start, p[1] - y_start] for p in box], dtype=np.int32)
            cv2.fillPoly(box_mask, [roi_pts], 255)
            
            # Final mask is only the bright pixels INSIDE the bounding box
            mask = cv2.bitwise_and(mask_text, box_mask)
            
            # Dilate the text pixels slightly to cover anti-aliasing and shadows
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=2)
            
            # 4. Inpaint the ROI
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                result_pil = self._lama_wrapper(roi_rgb, mask)
                result_roi = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
                result_roi = result_roi[:(y_end - y_start), :(x_end - x_start)]

                # 5. Feather-blend the inpainted region back into the frame.
                roi_h = y_end - y_start
                roi_w = x_end - x_start

                # Blur the mask slightly for a natural seam.
                weight = mask[:roi_h, :roi_w].astype(np.float32) / 255.0
                weight = cv2.GaussianBlur(weight, (0, 0), 3.0)
                weight = np.clip(weight, 0.0, 1.0)
                weight3 = weight[:, :, np.newaxis]  # broadcast over BGR

                original_roi = frame[y_start:y_end, x_start:x_end].astype(np.float32)
                inpainted_roi = result_roi.astype(np.float32)
                blended = original_roi * (1.0 - weight3) + inpainted_roi * weight3
                frame[y_start:y_end, x_start:x_end] = np.clip(blended, 0, 255).astype(np.uint8)
            except Exception as e:
                print(f"AI Inpainting error on box: {e}")
                
        return frame
