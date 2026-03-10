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
    def __init__(self):
        try:
            from paddleocr import PaddleOCR
            # 'latin' covers Spanish, French, Portuguese, Italian, German, etc.
            # Use CPU for OCR — PaddlePaddle-GPU needs cuDNN 8 which conflicts with CUDA 12.
            # OCR on CPU is still fast; GPU stays free for LaMa inpainting.
            self.ocr_engine = PaddleOCR(use_angle_cls=True, lang="latin", use_gpu=False, show_log=False)
            self.use_paddle = True
            print(f"DEBUG: V4 OCR Init with PaddleOCR (CPU) - lang=latin")
        except Exception as e:
            print(f"PaddleOCR Init failed: {e}. Falling back to EasyOCR.")
            self.use_paddle = False
            self.ocr_engine = easyocr.Reader(['en', 'es', 'vi'], gpu=True)
            print(f"DEBUG: V4 OCR Init with EasyOCR (GPU) - [en, es, vi]")

        # 2. Inpainter (LaMa)
        self.inpainter = AIInpainter()
        
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
        debug_first = True  # Log the first OCR result for debugging
        
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
                        
                        # Debug: log first OCR result structure
                        if debug_first and result:
                            print(f"DEBUG: PaddleOCR first result type={type(result)}, len={len(result)}")
                            if result[0]:
                                print(f"DEBUG: result[0] type={type(result[0])}, len={len(result[0])}")
                                if len(result[0]) > 0:
                                    print(f"DEBUG: First line: {result[0][0]}")
                            else:
                                print(f"DEBUG: result[0] is None/empty")
                            debug_first = False
                        
                        if result and result[0]:
                            # Group boxes and texts to sort by vertical position
                            detected_lines = []
                            for line in result[0]:
                                bbox, (text, prob) = line
                                if prob > 0.4:
                                    detected_lines.append((bbox, text))
                            
                            # Sort top-to-bottom based on the first point's Y coordinate
                            detected_lines.sort(key=lambda x: x[0][0][1])
                            
                            for bbox, text in detected_lines:
                                s_back = w_roi / self.ocr_width if w_roi > self.ocr_width else 1.0
                                remapped_box = [[float(p[0] * s_back), float((p[1] * s_back) + roi_top)] for p in bbox]
                                boxes.append(remapped_box)
                                txts.append(text)
                    else:
                        result = self.ocr_engine.readtext(roi)
                        boxes = []; txts = []
                        # Group for sorting
                        detected_lines = []
                        for (bbox, text, prob) in result:
                            if prob > 0.4:
                                detected_lines.append((bbox, text))
                        
                        # Sort top-to-bottom
                        detected_lines.sort(key=lambda x: x[0][0][1])
                        
                        for bbox, text in detected_lines:
                            s_back = w_roi / self.ocr_width if w_roi > self.ocr_width else 1.0
                            remapped_box = [[float(p[0] * s_back), float((p[1] * s_back) + roi_top)] for p in bbox]
                            boxes.append(remapped_box)
                            txts.append(text)
                    
                    if txts:
                        ocr_history.append({
                            'frame': frame_idx,
                            'text': "\n".join(txts), # Join with newline to preserve speakers
                            'boxes': boxes
                        })
                except Exception as e:
                    if debug_first:
                        print(f"OCR Error on frame {frame_idx}: {e}")
                        import traceback
                        traceback.print_exc()
                        debug_first = False
            
            frame_idx += 1
            if frame_idx % 100 == 0:
                pct = int((frame_idx/total_frames)*100)
                elapsed = time.time() - start_time
                fps_curr = frame_idx / elapsed if elapsed > 0 else 0
                eta_str = ""
                if fps_curr > 0:
                    eta_sec = (total_frames - frame_idx) / fps_curr
                    eta_str = f" | ETA: {format_eta(eta_sec)}"
                _log(f"OCR Detection: {pct}% ({frame_idx}/{total_frames}) [{len(ocr_history)} found]{eta_str}")

        
        cap.release()
        return ocr_history, fps

    def inpaint_and_render(self, video_path, ocr_history, translated_srt, progress_callback=None):
        """Pass 2 & 3 Combined: Selective Inpainting + Final Encoding using 3-Tier Threading."""
        import queue
        import threading
        
        def _log(msg):
            if progress_callback: progress_callback(msg)

        _log("V4 Final Pass: Concurrent AI Inpainting + Subtitle Overlay...")
        
        # Write SRT with UTF-8 BOM for FFmpeg compatibility
        has_srt = bool(translated_srt and translated_srt.strip())
        temp_srt_path = video_path.replace(".mp4", "_v4_translated.srt")
        if has_srt:
            with open(temp_srt_path, "w", encoding="utf-8-sig") as f:
                f.write(translated_srt)
        
        # Escape path for FFmpeg subtitles filter (Windows needs special handling)
        srt_abs = os.path.abspath(temp_srt_path).replace("\\", "/").replace(":", "\\:")

        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        output_path = video_path.replace(".mp4", "_v4_final.mp4")
        
        frame_to_boxes = {}
        for seg in ocr_history:
            for f in range(seg['start_frame'], seg['last_frame'] + 1):
                frame_to_boxes[f] = seg['boxes']
        
        style = "FontSize=22,PrimaryColour=&H00FFFFFF,Outline=1.2,OutlineColour=&H00000000,BorderStyle=1,Shadow=1,Alignment=2,MarginV=15"
        
        # Build FFmpeg command — only add subtitles filter if we have SRT content
        if has_srt:
            vf_filter = f"subtitles='{srt_abs}':force_style='{style}'"
        else:
            vf_filter = "null"  # No-op filter (passthrough)
            _log("   ⚠ No subtitles to overlay, skipping subtitle filter")
        
        cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', 
            '-vf', vf_filter,
            '-c:v', 'h264_nvenc', '-preset', 'p5', '-cq', '32', '-b:v', '5M', '-maxrate', '8M', '-bufsize', '16M',
            '-c:a', 'aac', 
            output_path
        ]

        
        # Queues for threading
        read_queue = queue.Queue(maxsize=32)
        write_queue = queue.Queue(maxsize=32)
        stop_event = threading.Event()
        frames_done = [0]  # Shared mutable counter
        frames_done_lock = threading.Lock()

        # 1. Producer: Read frames from video (CPU)
        def producer():
            idx = 0
            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret: break
                boxes = frame_to_boxes.get(idx)
                read_queue.put((idx, frame, boxes))
                idx += 1
            read_queue.put(None) # Sentinel

        # 2. Processor: AI Inpainting (GPU)
        def processor():
            while not stop_event.is_set():
                item = read_queue.get()
                if item is None:
                    write_queue.put(None)
                    break
                
                idx, frame, boxes = item
                if boxes:
                    roi_height = int(h * self.region_ratio)
                    roi_top_clean = h - roi_height
                    roi = frame[roi_top_clean:, :]
                    if roi.size > 0:
                        local_boxes = [[[p[0], p[1]-roi_top_clean] for p in b] for b in boxes]
                        inpainted_roi = self.inpainter.inpaint_frame(roi, local_boxes)
                        hr, wr = roi.shape[:2]
                        frame[roi_top_clean:, :] = inpainted_roi[:hr, :wr]
                
                write_queue.put(frame.tobytes())
                with frames_done_lock:
                    frames_done[0] += 1

        # 3. Consumer: Write to FFmpeg (I/O)
        def consumer(pipe):
            try:
                while not stop_event.is_set():
                    data = write_queue.get()
                    if data is None: break
                    pipe.stdin.write(data)
            finally:
                pipe.stdin.close()

        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        
        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=processor)
        t3 = threading.Thread(target=consumer, args=(pipe,))
        
        t1.start(); t2.start(); t3.start()
        
        start_time = time.time()
        
        # Monitoring loop with real progress
        while t3.is_alive():
            with frames_done_lock:
                done = frames_done[0]
            if done > 0:
                pct = int((done / total_frames) * 100)
                elapsed = time.time() - start_time
                curr_fps = done / elapsed if elapsed > 0 else 0
                eta_str = ""
                if curr_fps > 0:
                    eta_sec = (total_frames - done) / curr_fps
                    eta_str = f" | ETA: {format_eta(eta_sec)}"
                _log(f"Inpainting & Rendering: {pct}% ({done}/{total_frames}) | {curr_fps:.1f} FPS{eta_str}")
            time.sleep(1.0)
            
        t1.join(); t2.join(); t3.join()
        pipe.wait()
        cap.release()
        
        final_mp4 = output_path.replace("_v4_final.mp4", "_v4_complete.mp4")
        audio_cmd = ['ffmpeg', '-y', '-i', output_path, '-i', video_path, '-map', '0:v', '-map', '1:a', '-c', 'copy', final_mp4]
        try:
            subprocess.run(audio_cmd, check=True, capture_output=True)
            if os.path.exists(output_path): os.remove(output_path)
            if os.path.exists(temp_srt_path): os.remove(temp_srt_path)
            return final_mp4
        except:
            return output_path

    def clean_and_translate_srt(self, ocr_history, fps, target_lang, translator_model="google", progress_callback=None):
        def _log(msg):
            if progress_callback: progress_callback(msg)
        
        _log("V4 Pass 2: AI Translation & Cleaning...")
        srt_content = frames_to_srt(ocr_history, fps)
        
        # Debug: count and log
        segments = get_stabilized_segments(ocr_history, fps)
        _log(f"   OCR found {len(ocr_history)} detections → {len(segments)} subtitle segments")
        
        if not srt_content.strip():
            _log("   ⚠ No subtitle text detected by OCR!")
            return ""
        
        # Debug: Save original OCR SRT for inspection  
        blocks = [b for b in srt_content.strip().split('\n\n') if b.strip()]
        _log(f"   Generated {len(blocks)} SRT blocks for translation")
        
        translator = AITranslator(model=translator_model)
        result = translator.translate_srt_content(srt_content, target_lang)
        translator.unload()
        
        # Debug: Count translated vs original in result
        result_blocks = [b for b in result.strip().split('\n\n') if b.strip()]
        _log(f"   Translated SRT has {len(result_blocks)} blocks")
        
        return result

def run_v4(video_path, target_lang, translator_model="google", progress_callback=None):
    def _log(msg):
        if progress_callback: progress_callback(msg)
    
    pipe = SelectiveInpaintPipe()
    ocr_history, fps = pipe.extract_metadata(video_path, progress_callback)
    
    _log(f"   OCR detected text on {len(ocr_history)} frames")
    
    segments = get_stabilized_segments(ocr_history, fps)
    _log(f"   Stabilized into {len(segments)} subtitle segments")
    
    # Count total frames that will be inpainted
    inpaint_frames = set()
    for seg in segments:
        for f in range(seg['start_frame'], seg['last_frame'] + 1):
            inpaint_frames.add(f)
    _log(f"   {len(inpaint_frames)} frames marked for inpainting")
    
    translated_srt = pipe.clean_and_translate_srt(ocr_history, fps, target_lang, translator_model, progress_callback)
    
    # Debug: save both SRTs next to the video for inspection
    base = video_path.replace(".mp4", "")
    debug_original = base + "_debug_original.srt"
    debug_translated = base + "_debug_translated.srt"
    original_srt = frames_to_srt(ocr_history, fps)
    with open(debug_original, "w", encoding="utf-8") as f:
        f.write(original_srt)
    with open(debug_translated, "w", encoding="utf-8") as f:
        f.write(translated_srt)
    _log(f"   📄 Debug SRTs saved: _debug_original.srt & _debug_translated.srt")
    
    return pipe.inpaint_and_render(video_path, segments, translated_srt, progress_callback)

