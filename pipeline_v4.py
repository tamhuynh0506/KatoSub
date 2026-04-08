import cv2
import os
import time
import subprocess
import threading
import numpy as np
from queue import Queue
from srt_utils import frames_to_srt, get_stabilized_segments

def format_eta(seconds):
    if seconds > 3600:
        return f"{int(seconds // 3600)}:{(int(seconds % 3600) // 60):02}:{(int(seconds % 60)):02}"
    else:
        return f"{(int(seconds // 60)):02}:{(int(seconds % 60)):02}"
from ai_translator import AITranslator

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
            import easyocr
            self.ocr_engine = easyocr.Reader(['en', 'es', 'vi'], gpu=True)
            print(f"DEBUG: V4 OCR Init with EasyOCR (GPU) - [en, es, vi]")

        # 2. Inpainter (LaMa + ProPainter)
        from inpainter import AIInpainter
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
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            
            if frame_idx % self.frame_skip == 0:
                # ROI for Subtitles
                roi = frame[roi_top:, :]
                
                # Resize for OCR if needed
                if w != self.ocr_width:
                    scale = self.ocr_width / w
                    roi_ocr = cv2.resize(roi, (self.ocr_width, int(roi.shape[0] * scale)))
                else:
                    roi_ocr = roi
                
                if self.use_paddle:
                    result = self.ocr_engine.ocr(roi_ocr, cls=True)
                    if result and result[0]:
                        for line in result[0]:
                            box = line[0]
                            # Scale box back to original coordinates
                            if w != self.ocr_width:
                                box = [[p[0]/scale, p[1]/scale + roi_top] for p in box]
                            else:
                                box = [[p[0], p[1] + roi_top] for p in box]
                            ocr_history.append((frame_idx, box))
                else:
                    result = self.ocr_engine.readtext(roi_ocr)
                    for (box, text, prob) in result:
                        if w != self.ocr_width:
                            box = [[p[0]/scale, p[1]/scale + roi_top] for p in box]
                        else:
                            box = [[p[0], p[1] + roi_top] for p in box]
                        ocr_history.append((frame_idx, box))

            frame_idx += 1
            if frame_idx % 100 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_idx / elapsed
                eta = (total_frames - frame_idx) / fps_proc
                _log(f"Detection: {frame_idx}/{total_frames} ({int(frame_idx/total_frames*100)}%) | {fps_proc:.1f} FPS | ETA: {format_eta(eta)}")

        cap.release()
        
        # Group detections into stable segments
        _log("V4 Pass 1: Finalizing subtitle groups...")
        segments = get_stabilized_segments(ocr_history, total_frames, fps, self.frame_skip)
        
        # Convert segments to frame-by-frame boxes
        frame_to_boxes = {}
        for s in segments:
            for f in range(s['start_frame'], s['end_frame'] + 1):
                if f not in frame_to_boxes: frame_to_boxes[f] = []
                frame_to_boxes[f].append(s['box'])
        
        return frame_to_boxes, total_frames, fps, (h, w)

    def inpaint_and_render(self, video_path, frame_to_boxes, total_frames, fps, size, output_path, progress_callback=None):
        """Pass 2: Three-Tier Pipeline (Producer/Processor/Consumer)."""
        h, w = size
        
        # FFmpeg setup
        cmd = [
            'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}', '-pix_fmt', 'bgr24', '-r', str(fps),
            '-i', '-', '-i', video_path, '-map', '0:v:0', '-map', '1:a:0?',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'ultrafast',
            '-crf', '18', output_path
        ]

        read_queue = Queue(maxsize=120)
        write_queue = Queue(maxsize=120)
        stop_event = threading.Event()
        frames_done = [0]
        frames_done_lock = threading.Lock()

        # 1. Producer: Read from OpenCV (Disk I/O)
        def producer():
            cap = cv2.VideoCapture(video_path)
            idx = 0
            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret: break
                boxes = frame_to_boxes.get(idx)
                read_queue.put((idx, frame, boxes))
                idx += 1
            read_queue.put(None)
            cap.release()

        # 2. Processor: AI Inpainting (GPU)
        def processor():
            chunk_size = 40
            buffer = []
            
            def process_buffer(buf):
                if not buf: return
                
                rois = []
                boxes_list = []
                roi_height = int(h * self.region_ratio)
                roi_top_clean = h - roi_height
                
                for f_idx, frame, boxes in buf:
                    if boxes:
                        roi = frame[roi_top_clean:, :].copy()
                        local_boxes = [[[p[0], p[1]-roi_top_clean] for p in b] for b in boxes]
                        rois.append(roi)
                        boxes_list.append(local_boxes)
                    else:
                        rois.append(None)
                        boxes_list.append(None)
                
                if any(boxes_list):
                    valid_rois = [r if r is not None else np.zeros((roi_height, w, 3), dtype=np.uint8) for r in rois]
                    valid_boxes = [b if b is not None else [] for b in boxes_list]
                    
                    try:
                        inpainted_rois = self.inpainter.inpaint_sequence(valid_rois, valid_boxes)
                        for i, (f_idx, frame, boxes) in enumerate(buf):
                            if rois[i] is not None:
                                ir = inpainted_rois[i]
                                frame[roi_top_clean:roi_top_clean+ir.shape[0], :ir.shape[1]] = ir
                    except Exception as e:
                        print(f"Error in buffer inpainting: {e}")
                
                for f_idx, frame, boxes in buf:
                    write_queue.put(frame.tobytes())
                    with frames_done_lock:
                        frames_done[0] += 1

            while not stop_event.is_set():
                item = read_queue.get()
                if item is None:
                    process_buffer(buffer)
                    write_queue.put(None)
                    break
                
                buffer.append(item)
                if len(buffer) >= chunk_size:
                    process_buffer(buffer)
                    buffer = []

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
        
        t1.start()
        t2.start()
        t3.start()

        # Monitoring
        start_time = time.time()
        while any(t.is_alive() for t in [t1, t2, t3]):
            time.sleep(1)
            with frames_done_lock:
                done = frames_done[0]
            if done > 0:
                elapsed = time.time() - start_time
                fps_proc = done / elapsed
                eta = (total_frames - done) / fps_proc
                if progress_callback:
                    progress_callback(f"Inpainting & Rendering: {done}/{total_frames} ({int(done/total_frames*100)}%) | {fps_proc:.1f} FPS | ETA: {format_eta(eta)}")

        t1.join()
        t2.join()
        t3.join()
        pipe.wait()

    def clean_and_translate_srt(self, video_path, srt_content, target_lang="vi", progress_callback=None):
        """Standard subtitle translation step."""
        if progress_callback: progress_callback("V4: Translating subtitles...")
        translator = AITranslator()
        return translator.translate_srt_content(srt_content, target_lang, progress_callback)

    def run(self, video_path, output_path, target_lang="vi", progress_callback=None):
        """Full execution flow."""
        # 1. Detect
        frame_to_boxes, total_frames, fps, size = self.extract_metadata(video_path, progress_callback)
        
        # 2. Inpaint & Render
        temp_video = output_path.replace(".mp4", "_inpainted.mp4")
        self.inpaint_and_render(video_path, frame_to_boxes, total_frames, fps, size, temp_video, progress_callback)
        
        # 3. Clean OCR -> SRT
        # (This remains as in v2/v3, simplified here)
        if progress_callback: progress_callback("V4: Finalizing output...")
        
        # Rename temp to final
        if os.path.exists(temp_video):
            if os.path.exists(output_path): os.remove(output_path)
            os.rename(temp_video, output_path)
        
        return output_path


def run_v4(video_path, output_path, target_lang="vi", progress_callback=None):
    pipe = SelectiveInpaintPipe()
    return pipe.run(video_path, output_path, target_lang, progress_callback)
