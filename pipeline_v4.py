import cv2
import os
import time
import subprocess
import easyocr
from srt_utils import frames_to_srt, get_stabilized_segments

def format_eta(seconds):
    if seconds > 3600:
        return f"{int(seconds // 3600)}:{(int(seconds % 3600) // 60):02}:{(int(seconds % 60)):02}"
    else:
        return f"{(int(seconds // 60)):02}:{(int(seconds % 60)):02}"
from ai_translator import AITranslator
from inpainter import AIInpainter

class SelectiveInpaintPipe:
    def __init__(self, api_key=None, source_lang="en"):
        try:
            import paddle
            from paddleocr import PaddleOCR
            paddle_lang_map = {
                "en": "en", "zh-cn": "ch", "zh-tw": "ch", "ja": "japan", "ko": "ko",
                "fr": "fr", "de": "german", "es": "es", "pt": "pt", "ru": "ru", "vi": "vi"
            }
            p_lang = paddle_lang_map.get(source_lang, "en")
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang=p_lang, use_gpu=True, show_log=False)
            self.use_paddle = True
            print(f"DEBUG: V4 OCR Init with PaddleOCR (GPU)")
        except Exception as e:
            print(f"PaddleOCR Init failed: {e}. Falling back to EasyOCR.")
            self.use_paddle = False
            lang_map = {
                "en": "en", "ja": "ja", "ko": "ko", "zh-cn": "ch_sim", "zh-tw": "ch_tra",
                "fr": "fr", "de": "de", "es": "es", "pt": "pt", "ru": "ru", "vi": "vi"
            }
            ocr_lang = lang_map.get(source_lang, "en")
            self.ocr_engine = easyocr.Reader([ocr_lang], gpu=True)
            print(f"DEBUG: V4 OCR Init with EasyOCR (GPU)")

        # 2. Inpainter (LaMa)
        self.inpainter = AIInpainter()
        
        self.api_key = api_key
        self.region_ratio = 0.3 # Catch higher subs (like v2)
        self.frame_skip = 2
        self.ocr_width = 1280 # Better resolution (like v2)

    def extract_metadata(self, video_path, progress_callback=None):
        """Pass 1: Detect subtitle boxes across the video."""
        def _log(msg):
            if progress_callback: progress_callback(msg)

        _log("V4 Pass 1: Detecting Subtitle Regions...")
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        h, w = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        roi_top = int(h * (1 - self.region_ratio))
        
        ocr_history = []
        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % self.frame_skip == 0:
                roi = frame[roi_top:, :]
                # Resize for OCR speed
                h_roi, w_roi = roi.shape[:2]
                if w_roi > self.ocr_width:
                    scale = self.ocr_width / w_roi
                    roi = cv2.resize(roi, (self.ocr_width, int(h_roi * scale)))
                
                try:
                    if self.use_paddle:
                        result = self.ocr_engine.ocr(roi, cls=True)
                        boxes = []; txts = []
                        if result and result[0]:
                            for line in result[0]:
                                bbox, (text, prob) = line
                                if prob > 0.4:
                                    s_back = w_roi / self.ocr_width if w_roi > self.ocr_width else 1.0
                                    remapped_box = [[float(p[0] * s_back), float((p[1] * s_back) + roi_top)] for p in bbox]
                                    boxes.append(remapped_box)
                                    txts.append(text)
                    else:
                        result = self.ocr_engine.readtext(roi)
                        boxes = []; txts = []
                        for (bbox, text, prob) in result:
                            if prob > 0.4:
                                s_back = w_roi / self.ocr_width if w_roi > self.ocr_width else 1.0
                                remapped_box = [[float(p[0] * s_back), float((p[1] * s_back) + roi_top)] for p in bbox]
                                boxes.append(remapped_box)
                                txts.append(text)
                    
                    if txts:
                        ocr_history.append({
                            'frame': frame_idx,
                            'text': " ".join(txts),
                            'boxes': boxes
                        })
                except Exception as e:
                    print(f"OCR Error: {e}")
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = int((frame_idx/total_frames)*100)
                elapsed = time.time() - start_time
                fps_curr = frame_idx / elapsed if elapsed > 0 else 0
                eta_str = ""
                if fps_curr > 0:
                    eta_sec = (total_frames - frame_idx) / fps_curr
                    eta_str = f" | ETA: {format_eta(eta_sec)}"
                _log(f"OCR Detection: {pct}% ({frame_idx}/{total_frames}){eta_str}")
        
        cap.release()
        return ocr_history, fps

    def inpaint_and_render(self, video_path, ocr_history, translated_srt, progress_callback=None):
        """Pass 2 & 3 Combined: Selective Inpainting + Final Encoding."""
        def _log(msg):
            if progress_callback: progress_callback(msg)

        _log("V4 Final Pass: AI Inpainting + Subtitle Overlay...")
        
        # Save temporary SRT for FFmpeg to read
        temp_srt_path = video_path.replace(".mp4", "_v4_translated.srt")
        with open(temp_srt_path, "w", encoding="utf-8") as f:
            f.write(translated_srt)
            
        srt_abs = os.path.abspath(temp_srt_path).replace("\\", "/")
        if ":" in srt_abs:
            drive, path = srt_abs.split(":", 1)
            srt_abs = f"{drive}\\:{path}"

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = video_path.replace(".mp4", "_v4_final.mp4")
        
        # Pre-process segments into a frame-to-boxes map for fast lookup
        # Each frame in the range [start, end] gets the segment's boxes
        frame_to_boxes = {}
        for seg in ocr_history: # ocr_history here is actually 'segments' if called from run_v4
            for f in range(seg['start_frame'], seg['last_frame'] + 1):
                frame_to_boxes[f] = seg['boxes']
        
        # Subtitle Style
        style = "FontSize=22,PrimaryColour=&H00FFFFFF,Outline=1.2,OutlineColour=&H00000000,BorderStyle=1,Shadow=1,Alignment=2,MarginV=15"
        
        # Command: Rawvideo In -> Subtitles Filter -> NVENC Out
        # Note: We don't pipe stderr here to avoid deadlocks; we'll capture it if needed at the end.
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', 
            '-vf', f"subtitles='{srt_abs}':force_style='{style}'",
            '-c:v', 'h264_nvenc', '-preset', 'p4', '-cq', '20',
            '-c:a', 'aac', 
            output_path
        ]
        
        # We remove stderr=subprocess.PIPE to prevent buffer-fill hangs
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        frame_idx = 0
        start_time = time.time()
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            # Use timing-based intervals for inpainting
            boxes = frame_to_boxes.get(frame_idx)
            
            if boxes:
                # Strip-based inpainting for BETTER REPLACEMENT (v2 style)
                # We inpaint the entire bottom strip where subtitles reside
                # This provides more context for LaMa and prevents "floating" artifacts
                roi_height = int(h * self.region_ratio)
                roi_top_clean = h - roi_height
                
                roi = frame[roi_top_clean:, :]
                if roi.size > 0:
                    # Map segment boxes to local ROI coordinates for the mask
                    local_boxes = [[[p[0], p[1]-roi_top_clean] for p in b] for b in boxes]
                    inpainted_roi = self.inpainter.inpaint_frame(roi, local_boxes)
                    hr, wr = roi.shape[:2]
                    frame[roi_top_clean:, :] = inpainted_roi[:hr, :wr]
            
            pipe.stdin.write(frame.tobytes())
            
            frame_idx += 1
            if frame_idx % 25 == 0: # More frequent updates (every 25 frames)
                pct = int((frame_idx/total_frames)*100)
                elapsed = time.time() - start_time
                curr_fps = frame_idx / elapsed if elapsed > 0 else 0
                eta_str = ""
                if curr_fps > 0:
                    eta_sec = (total_frames - frame_idx) / curr_fps
                    eta_str = f" | ETA: {format_eta(eta_sec)}"
                _log(f"Inpainting & Rendering: {pct}% ({frame_idx}/{total_frames}) | {curr_fps:.1f} FPS{eta_str}")
                
        cap.release()
        pipe.stdin.close()
        pipe.wait()
        
        # Final Step: Add the original audio back (NVENC pipe ignores initial file audio)
        final_mp4 = output_path.replace("_v4_final.mp4", "_v4_complete.mp4")
        audio_cmd = [
            'ffmpeg', '-y',
            '-i', output_path,
            '-i', video_path,
            '-map', '0:v', '-map', '1:a',
            '-c', 'copy', final_mp4
        ]
        try:
            subprocess.run(audio_cmd, check=True, capture_output=True)
            if os.path.exists(output_path): os.remove(output_path)
            if os.path.exists(temp_srt_path): os.remove(temp_srt_path)
            return final_mp4
        except:
            return output_path

    def clean_and_translate_srt(self, ocr_history, fps, source_lang, target_lang, progress_callback=None):
        def _log(msg):
            if progress_callback: progress_callback(msg)
        
        _log("V4 Pass 2: AI Translation & Cleaning...")
        srt_content = frames_to_srt(ocr_history, fps)
        
        if not self.api_key:
            return srt_content

        translator = AITranslator(self.api_key)
        prompt = f"""
        Translate the following subtitles from {source_lang} to {target_lang}.
        Maintain the SRT structure (timing and indices) perfectly. 
        Only return the translated SRT content.
        Subtitles:
        {srt_content}
        """
        
        return translator.translate_srt_content(srt_content, source_lang, target_lang, custom_prompt=prompt)

def run_v4(video_path, api_key, source_lang, target_lang, progress_callback=None):
    pipe = SelectiveInpaintPipe(api_key, source_lang=source_lang)
    
    # 1. OCR Extraction
    ocr_history, fps = pipe.extract_metadata(video_path, progress_callback)
    
    # 2. Get Stabilized Segments (Timing-based)
    segments = get_stabilized_segments(ocr_history, fps)
    
    # 3. AI Translation (Use segments for cleaner text)
    translated_srt = pipe.clean_and_translate_srt(ocr_history, fps, source_lang, target_lang, progress_callback)
    
    # 4. Final Pass: Inpaint + Overlay
    return pipe.inpaint_and_render(video_path, segments, translated_srt, progress_callback)
