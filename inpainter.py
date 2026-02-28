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
        Inpaint a single frame.
        """
        if not boxes:
            return frame
        
        self._ensure_lama()
        
        # Create mask
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        for box in boxes:
            pts = np.array(box, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 255)
        
        # Dilate mask slightly to cover edges, glows, and shadows
        kernel = np.ones((7, 7), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=4)
        
        # Convert BGR to RGB for LaMa
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Run LaMa using the wrapper's verified logic
        try:
            result_pil = self._lama_wrapper(frame_rgb, mask)
            # Convert back to BGR numpy array
            return cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        except Exception as e:
            # Fallback to original frame on AI error to prevent pipeline crash
            print(f"AI Inpainting error: {e}")
            return frame
