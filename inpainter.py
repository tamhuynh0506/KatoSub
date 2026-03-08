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
        """
        if not boxes:
            return frame
        
        self._ensure_lama()
        
        # 1. Calculate the bounding box of all subtitle boxes in this frame
        # We add some padding to ensure the AI has context
        all_pts = []
        for box in boxes:
            all_pts.extend(box)
        all_pts = np.array(all_pts, dtype=np.int32)
        
        x_min, y_min = np.min(all_pts, axis=0)
        x_max, y_max = np.max(all_pts, axis=0)
        
        # Add padding (e.g. 20px)
        pad = 20
        h_f, w_f = frame.shape[:2]
        x_start = max(0, int(x_min) - pad)
        y_start = max(0, int(y_min) - pad)
        x_end = min(w_f, int(x_max) + pad)
        y_end = min(h_f, int(y_max) + pad)
        
        # 2. Extract the cropped ROI
        roi = frame[y_start:y_end, x_start:x_end]
        if roi.size == 0:
            return frame
            
        # 3. Create mask for the ROI
        mask = np.zeros(roi.shape[:2], dtype=np.uint8)
        # Remap boxes to ROI coordinates
        for box in boxes:
            pts = np.array([[p[0] - x_start, p[1] - y_start] for p in box], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        # Apply dilation to the mask
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)
        
        # 4. Inpaint the ROI
        try:
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            result_pil = self._lama_wrapper(roi_rgb, mask)
            result_roi = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
            
            # 5. Paste back into original frame
            frame[y_start:y_end, x_start:x_end] = result_roi[:(y_end-y_start), :(x_end-x_start)]
            return frame
        except Exception as e:
            print(f"AI Inpainting error: {e}")
            return frame
