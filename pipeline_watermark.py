"""
pipeline_watermark.py  High-Quality Watermark Removal Pipeline

Steps:
  1. Detect watermark regions via WatermarkDetector (temporal variance analysis)
  2. Per-frame: call AIInpainter (LaMa) only on frames that fall within a watermark's
     temporal range  skipping frames where no watermark is present.
  3. Encode result with FFmpeg NVENC (GPU) for maximum speed.

Multiple watermarks at different times are handled: for any given frame, only
the watermarks that are active during that frame are inpainted.
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
# from inpainter import AIInpainter


def format_eta(seconds: float) -> str:
    if seconds > 3600:
        return f"{int(seconds // 3600)}:{int(seconds % 3600) // 60:02}:{int(seconds % 60):02}"
    return f"{int(seconds // 60):02}:{int(seconds % 60):02}"


class WatermarkRemovalPipeline:
    """
    End-to-end pipeline:
      detect  AI inpaint (LaMa)  NVENC encode
    """

    def __init__(self):
        self.detector = WatermarkDetector(
            sample_rate=0.06,
            min_area=250,
            n_segments=6,
            margin=8,
        )
        self.inpainter = None  # Deferred initialization

    # 
    # Public entry point
    # 

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

        #  Step 1: Detect 
        if regions is None:
            regions = self.detector.detect(video_path, progress_callback=progress_callback)
        else:
            _log(f"  Using {len(regions)} pre-determined watermark region(s)")

        if not regions:
            _log("No watermarks detected -- returning original video unchanged.")
            return video_path

        _log(f"Detected {len(regions)} watermark region(s) -- starting AI inpainting")

        #  Step 2 + 3: Inpaint + Encode 
        if self.inpainter is None:
            from inpainter import AIInpainter
            self.inpainter = AIInpainter()
            
        output_path = self._build_output_path(video_path, output_dir)
        self._inpaint_and_encode(video_path, regions, output_path, _log)

        _log(f" Saved: {os.path.basename(output_path)}")
        return output_path

    # 
    # Internals
    # 

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
        """Convert active WatermarkRegion list  inpainter boxes format."""
        return [r.box_points for r in active]

    def _inpaint_and_encode(
        self,
        video_path: str,
        regions: list,
        output_path: str,
        _log,
    ):
        """
        3-tier threaded pipeline (same architecture as pipeline_v4):
          Producer  Processor(LaMa)  Consumer(FFmpeg NVENC)
        """
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w            = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h            = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        _log(f" Video: {w}{h} | {total_frames} frames @ {fps:.2f} fps")

        #  FFmpeg NVENC encoder 
        temp_video_path = output_path.replace(".mp4", "_temp.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-f", "rawvideo", "-vcodec", "rawvideo",
            "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
            "-i", "-",
            # High quality NVENC
            "-c:v", "h264_nvenc",
            "-preset", "p5",          # Slow preset  better quality
            "-cq", "25",
            "-b:v", "5M",
            "-maxrate", "20M",
            "-bufsize", "40M",
            temp_video_path,
        ]

        #  Shared state 
        read_queue  = queue.Queue(maxsize=32)
        write_queue = queue.Queue(maxsize=32)
        stop_event  = threading.Event()
        frames_done = [0]
        done_lock   = threading.Lock()

        #  Producer 
        def producer():
            idx = 0
            while cap.isOpened() and not stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                read_queue.put((idx, frame))
                idx += 1
            read_queue.put(None)  # sentinel

        #  Processor (AI inpainting, GPU) 
        def processor():
            while not stop_event.is_set():
                item = read_queue.get()
                if item is None:
                    write_queue.put(None)
                    break

                idx, frame = item
                active = self._frame_regions(idx, regions)

                if active:
                    boxes = self._regions_to_boxes(active)
                    try:
                        frame = self.inpainter.inpaint_frame(frame, boxes)
                    except Exception as e:
                        # Fall through  original frame is used
                        _log(f"   Inpaint error on frame {idx}: {e}")
                        pass

                write_queue.put(frame.tobytes())
                with done_lock:
                    frames_done[0] = frames_done[0] + 1

        #  Consumer (write to FFmpeg stdin) 
        def consumer(pipe):
            try:
                while not stop_event.is_set():
                    data = write_queue.get()
                    if data is None:
                        break
                    pipe.stdin.write(data)
            finally:
                pipe.stdin.close()

        # Launch
        pipe = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
        t1 = threading.Thread(target=producer,            daemon=True)
        t2 = threading.Thread(target=processor,           daemon=True)
        t3 = threading.Thread(target=consumer, args=(pipe,), daemon=True)
        
        t1.start(); t2.start(); t3.start()

        start_time = time.time()

        # Progress monitor
        while t3.is_alive():
            with done_lock:
                done = frames_done[0]
            if done > 0:
                pct     = int((done / total_frames) * 100)
                elapsed = time.time() - start_time
                fps_now = done / elapsed if elapsed > 0 else 0
                eta_str = ""
                if fps_now > 0:
                    eta_str = f" | ETA: {format_eta((total_frames - done) / fps_now)}"
                _log(f"Inpainting & Rendering: {pct}% ({done}/{total_frames}) | {fps_now:.1f} FPS{eta_str}")
            time.sleep(1.0)

        t1.join(); t2.join(); t3.join()
        pipe.wait()
        cap.release()
        
        # Step 2: Mux audio back in
        _log("Muxing audio...")
        audio_cmd = [
            'ffmpeg', '-y',
            '-i', temp_video_path,
            '-i', video_path,
            '-map', '0:v', '-map', '1:a?',
            '-c', 'copy',
            output_path
        ]
        try:
            subprocess.run(audio_cmd, check=True, capture_output=True)
            if os.path.exists(temp_video_path):
                os.remove(temp_video_path)
            # Make sure we don't accidentally leave anything corrupted
        except Exception as e:
            _log(f"Audio mux error: {e}")


def run_watermark_pipeline(
    video_path: str,
    progress_callback=None,
    output_dir: Optional[str] = None,
    regions: Optional[List[WatermarkRegion]] = None,
) -> str:
    """Convenience wrapper for main.py."""
    pipe = WatermarkRemovalPipeline()
    return pipe.run(video_path, progress_callback=progress_callback, output_dir=output_dir, regions=regions)
