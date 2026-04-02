"""
pipeline_watermark.py — High-Quality Watermark Removal Pipeline

Steps:
  1. Detect watermark regions via WatermarkDetector (temporal variance analysis)
  2. Per-frame chunk: call AIInpainter (ProPainter) on buffered frame sequences
  3. Encode result with FFmpeg NVENC (GPU) for maximum speed.
"""

import cv2
import os
import time
import queue
import threading
import subprocess
import numpy as np

from typing import Optional, List, Callable

from watermark_detector import WatermarkDetector, WatermarkRegion
from inpainter import AIInpainter


def format_eta(seconds: float) -> str:
    if seconds > 3600:
        return f"{int(seconds // 3600)}:{int(seconds % 3600) // 60:02}:{int(seconds % 60):02}"
    return f"{int(seconds // 60):02}:{int(seconds % 60):02}"


class WatermarkRemovalPipeline:
    """
    End-to-end pipeline:
      detect → AI inpaint (ProPainter) → NVENC encode
    """

    def __init__(self):
        self.detector = WatermarkDetector(
            sample_rate=0.06,
            min_area=250,
            n_segments=6,
            margin=8,
        )
        self.inpainter = AIInpainter(use_propainter=True)

    def run(
        self,
        video_path: str,
        progress_callback=None,
        output_dir: Optional[str] = None,
        regions: Optional[List[WatermarkRegion]] = None,
    ) -> str:
        """
        Detect and remove watermarks from *video_path*.
        Returns path to the cleaned output video.
        """
        def _log(msg):
            if progress_callback:
                progress_callback(msg)

        _log("Watermark Removal Pipeline started")

        # Step 1: Detect 
        if regions is None:
            regions = self.detector.detect(video_path, progress_callback=progress_callback)
        else:
            _log(f"Using {len(regions)} pre-determined watermark region(s)")

        if not regions:
            _log("No watermarks detected -- returning original video unchanged.")
            return video_path

        _log(f"Detected {len(regions)} watermark region(s) -- starting ProPainter inpainting")
            
        output_path = self._build_output_path(video_path, output_dir)
        self._inpaint_and_encode(video_path, regions, output_path, _log)

        _log(f"Saved: {os.path.basename(output_path)}")
        return output_path

    @staticmethod
    def _build_output_path(video_path: str, output_dir: Optional[str]) -> str:
        base      = os.path.basename(video_path)
        stem, _   = os.path.splitext(base)
        out_name  = f"{stem}_watermark_removed.mp4"
        if output_dir:
            return os.path.join(output_dir, out_name)
        return os.path.join(os.path.dirname(video_path), out_name)

    def _frame_regions(self, frame_idx: int, regions: list) -> list:
        """Return the list of WatermarkRegion active at *frame_idx*."""
        return [r for r in regions if r.start_frame <= frame_idx <= r.end_frame]

    def _regions_to_boxes(self, active: list):
        """Convert active WatermarkRegion list to inpainter boxes format."""
        return [r.box_points for r in active]

    def _inpaint_and_encode(
        self,
        video_path: str,
        regions: list,
        output_path: str,
        _log,
    ):
        """
        3-tier threaded pipeline:
          Producer → Processor(ProPainter) → Consumer(FFmpeg NVENC)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        _log(f"Video: {w}x{h} | {total_frames} frames @ {fps:.2f} fps")

        # FFmpeg setup
        temp_video_path = output_path.replace(".mp4", "_temp.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "ultrafast",
            "-crf", "18",
            temp_video_path,
        ]

        read_queue  = queue.Queue(maxsize=120)
        write_queue = queue.Queue(maxsize=120)
        stop_event  = threading.Event()
        frames_done = [0]
        done_lock   = threading.Lock()

        def producer():
            idx = 0
            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret: break
                read_queue.put((idx, frame))
                idx += 1
            read_queue.put(None)

        def processor():
            chunk_size = 40
            buffer = []

            def process_buffer(buf):
                if not buf: return
                
                frames = [item[1] for item in buf]
                boxes_list = []
                for idx, frame in buf:
                    active = self._frame_regions(idx, regions)
                    boxes_list.append(self._regions_to_boxes(active))
                
                if any(boxes_list):
                    try:
                        inpainted_frames = self.inpainter.inpaint_sequence(frames, boxes_list)
                        for i, (idx, _) in enumerate(buf):
                            write_queue.put(inpainted_frames[i].tobytes())
                            with done_lock:
                                frames_done[0] += 1
                    except Exception as e:
                        _log(f"Inpaint sequence error: {e}")
                        for i, (idx, frame) in enumerate(buf):
                            write_queue.put(frame.tobytes())
                            with done_lock:
                                frames_done[0] += 1
                else:
                    for idx, frame in buf:
                        write_queue.put(frame.tobytes())
                        with done_lock:
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

        def consumer(pipe):
            try:
                while not stop_event.is_set():
                    data = write_queue.get()
                    if data is None: break
                    pipe.stdin.write(data)
            finally:
                pipe.stdin.close()

        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        t1 = threading.Thread(target=producer, daemon=True)
        t2 = threading.Thread(target=processor, daemon=True)
        t3 = threading.Thread(target=consumer, args=(pipe,), daemon=True)
        
        t1.start(); t2.start(); t3.start()

        start_time = time.time()
        while t3.is_alive():
            with done_lock:
                done = frames_done[0]
            if done > 0:
                pct     = int((done / total_frames) * 100)
                elapsed = time.time() - start_time
                fps_now = done / elapsed if elapsed > 0 else 0
                eta_str = f" | ETA: {format_eta((total_frames - done) / fps_now)}" if fps_now > 0 else ""
                _log(f"Inpainting & Rendering: {pct}% ({done}/{total_frames}) | {fps_now:.1f} FPS{eta_str}")
            time.sleep(1.0)

        t1.join(); t2.join(); t3.join()
        pipe.wait()
        cap.release()
        
        # Step 2: Mux audio back in
        _log("Muxing audio...")
        audio_cmd = [
            'ffmpeg', '-y', '-i', temp_video_path, '-i', video_path,
            '-map', '0:v', '-map', '1:a?', '-c', 'copy', output_path
        ]
        try:
            subprocess.run(audio_cmd, check=True, capture_output=True)
            if os.path.exists(temp_video_path): os.remove(temp_video_path)
        except Exception as e:
            _log(f"Audio mux error: {e}")


def run_watermark_pipeline(video_path, progress_callback=None, output_dir=None, regions=None):
    pipe = WatermarkRemovalPipeline()
    return pipe.run(video_path, progress_callback, output_dir, regions)
