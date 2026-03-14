import os
import time
import subprocess
import whisper
import cv2
from ai_translator import AITranslator


def format_eta(seconds):
    if seconds > 3600:
        return f"{int(seconds // 3600)}:{(int(seconds % 3600) // 60):02}:{(int(seconds % 60)):02}"
    else:
        return f"{(int(seconds // 60)):02}:{(int(seconds % 60)):02}"


def get_video_duration(video_path):
    """Get video duration in seconds using ffprobe (primary) or OpenCV (fallback)."""
    try:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path
        ]
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=5)
        if result.returncode == 0:
            return float(result.stdout.strip())
    except Exception:
        pass

    # Fallback to OpenCV
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return 0
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        return duration
    except Exception:
        return 0


def format_timestamp_srt(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def extract_audio(video_path, progress_callback=None):
    """Extract audio from video to a temporary WAV file using FFmpeg."""
    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    audio_path = video_path.rsplit(".", 1)[0] + "_temp_audio.wav"
    _log("Extracting audio from video...")

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                      # No video
        "-acodec", "pcm_s16le",     # 16-bit PCM (Whisper expects this)
        "-ar", "16000",             # 16kHz sample rate (Whisper optimal)
        "-ac", "1",                 # Mono
        audio_path
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, encoding="utf-8", errors="replace", timeout=300)
        if result.returncode != 0:
            _log(f"FFmpeg audio extraction error: {result.stderr[:200]}")
            return None
    except subprocess.TimeoutExpired:
        _log("Audio extraction timed out (>5 min)")
        return None

    if os.path.exists(audio_path):
        size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        _log(f"Audio extracted: {size_mb:.1f} MB")
        return audio_path
    return None


def transcribe_audio(audio_path, whisper_model_name="base", progress_callback=None):
    """Transcribe audio using Whisper and return SRT-formatted content.
    
    Uses word-level timestamps for precise timing and applies a small delay
    offset to ensure subtitles don't appear before the speech starts.
    """
    # Delay offset in seconds — shift subtitles forward to sync with audio.
    # Whisper timestamps tend to start ~0.2-0.5s before actual speech.
    SUBTITLE_DELAY_OFFSET = 0.3

    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    _log(f"Loading Whisper model: {whisper_model_name}...")
    start_time = time.time()
    model = whisper.load_model(whisper_model_name)
    load_time = time.time() - start_time
    _log(f"Whisper model loaded in {load_time:.1f}s")

    _log("Transcribing audio (this may take a while)...")
    start_time = time.time()
    result = model.transcribe(
        audio_path,
        verbose=False,
        word_timestamps=True,           # More precise timing per word
        no_speech_threshold=0.6,        # Better silence filtering
        condition_on_previous_text=True, # Better context continuity
    )
    transcribe_time = time.time() - start_time

    segments = result.get("segments", [])
    detected_lang = result.get("language", "unknown")
    _log(f"Transcription complete in {transcribe_time:.1f}s | "
         f"Language: {detected_lang} | {len(segments)} segments detected")

    if not segments:
        _log("No speech detected in audio!")
        return "", detected_lang

    # Build SRT content from Whisper segments with timing correction
    srt_lines = []
    srt_idx = 0
    for seg in segments:
        text = seg["text"].strip()
        if not text:
            continue

        # Use the first word's timestamp as the real start of speech
        # This is more accurate than the segment-level start time
        words = seg.get("words", [])
        if words:
            # First word start = when speech actually begins
            seg_start = words[0]["start"]
            # Last word end = when speech actually ends
            seg_end = words[-1]["end"]
        else:
            seg_start = seg["start"]
            seg_end = seg["end"]

        # Apply delay offset to push subtitles later (sync with audio)
        seg_start = max(0, seg_start + SUBTITLE_DELAY_OFFSET)
        seg_end = seg_end + SUBTITLE_DELAY_OFFSET

        # Ensure minimum subtitle duration of 0.5s
        if seg_end - seg_start < 0.5:
            seg_end = seg_start + 0.5

        srt_idx += 1
        start = format_timestamp_srt(seg_start)
        end = format_timestamp_srt(seg_end)
        srt_lines.append(f"{srt_idx}")
        srt_lines.append(f"{start} --> {end}")
        srt_lines.append(f"{text}\n")

    srt_content = "\n".join(srt_lines)
    _log(f"Generated SRT with {srt_idx} blocks (delay offset: +{SUBTITLE_DELAY_OFFSET}s)")

    # Unload model from GPU to free VRAM for other tasks
    del model
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    return srt_content, detected_lang


def render_subtitles(video_path, srt_content, progress_callback=None):
    """Render translated subtitles onto the video using FFmpeg (no inpainting needed)."""
    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    if not srt_content or not srt_content.strip():
        _log("No subtitles to render!")
        return video_path

    output_path = video_path.rsplit(".", 1)[0] + "_audio_translated.mp4"
    temp_srt_path = video_path.rsplit(".", 1)[0] + "_audio_translated.srt"

    # Write SRT with UTF-8 BOM for FFmpeg compatibility
    with open(temp_srt_path, "w", encoding="utf-8-sig") as f:
        f.write(srt_content)

    _log("Rendering subtitles onto video...")

    # Get duration for progress calculation
    total_duration = get_video_duration(video_path)
    if total_duration <= 0:
        _log("   ⚠ Could not determine video duration, progress will be limited.")

    # Escape path for FFmpeg subtitles filter (Windows needs special handling)
    srt_abs = os.path.abspath(temp_srt_path).replace("\\", "/").replace(":", "\\:")

    style = "FontSize=22,PrimaryColour=&H00FFFFFF,Outline=1.2,OutlineColour=&H00000000,BorderStyle=1,Shadow=1,Alignment=2,MarginV=15"

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vf", f"subtitles='{srt_abs}':force_style='{style}'",
        "-c:v", "h264_nvenc", "-preset", "p5", "-cq", "32",
        "-b:v", "5M", "-maxrate", "8M", "-bufsize", "16M",
        "-c:a", "copy",     # Passthrough audio
        "-progress", "pipe:1",  # Output progress to stdout
        output_path
    ]

    try:
        # Redirect stderr to stdout to avoid deadlock if stderr buffer fills up
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding="utf-8", errors="replace", bufsize=1
        )

        if total_duration > 0:
            _log(f"   Video duration: {total_duration:.1f}s")

        # Monitor FFmpeg progress from the combined stream
        _log("   FFmpeg: Starting subprocess...")
        start_time = time.time()
        last_log_time = 0
        
        while True:
            line = process.stdout.readline()
            if not line:
                if process.poll() is not None:
                    break
                time.sleep(0.1)
                continue
            
            line = line.strip()
            if "out_time_ms=" in line:
                try:
                    # Line looks like: out_time_ms=1234567
                    time_val = line.split("=")[1].strip()
                    if time_val.isdigit():
                        time_ms = int(time_val)
                        curr_sec = time_ms / 1000000.0
                        
                        if time.time() - last_log_time > 1.0:
                            if total_duration > 0:
                                pct = min(100, int((curr_sec / total_duration) * 100))
                                elapsed = time.time() - start_time
                                speed = curr_sec / elapsed if elapsed > 0 else 0
                                
                                eta_str = ""
                                if speed > 0:
                                    eta_sec = (total_duration - curr_sec) / speed
                                    eta_str = f" | ETA: {format_eta(eta_sec)}"
                                
                                _log(f"Rendering subtitles onto video: {pct}% | {curr_sec:.1f}s / {total_duration:.1f}s{eta_str}")
                            else:
                                _log(f"Rendering subtitles onto video: {curr_sec:.1f}s processed")
                            
                            last_log_time = time.time()
                except (ValueError, IndexError):
                    pass
            
            # Watch for obvious errors in the combined log stream
            if "Error" in line or "failed" in line.lower():
                _log(f"   FFmpeg: {line}")

        returncode = process.wait()

        if returncode != 0:
            _log(f"FFmpeg render failed with exit code {returncode}")
            return None

    except Exception as e:
        _log(f"FFmpeg render exception: {e}")
        return None

    # Clean up temp SRT file
    if os.path.exists(temp_srt_path):
        os.remove(temp_srt_path)

    if os.path.exists(output_path):
        _log(f"Output saved: {os.path.basename(output_path)}")
        return output_path

    return None


def run_audio_pipeline(video_path, target_lang, translator_model="google",
                       whisper_model="base", progress_callback=None):
    """
    Full audio transcription + translation pipeline:
    1. Extract audio from video
    2. Transcribe with Whisper
    3. Translate SRT content
    4. Render subtitles onto video
    """
    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    _log("Audio Pipeline: Starting...")
    overall_start = time.time()

    # --- Step 1: Extract audio ---
    _log("Step 1/4: Extracting audio...")
    audio_path = extract_audio(video_path, progress_callback)
    if not audio_path:
        _log("Failed to extract audio from video!")
        return None

    # --- Step 2: Transcribe with Whisper ---
    _log(f"Step 2/4: Transcribing with Whisper ({whisper_model})...")
    srt_content, detected_lang = transcribe_audio(audio_path, whisper_model, progress_callback)

    # Clean up temp audio file
    if os.path.exists(audio_path):
        os.remove(audio_path)

    if not srt_content.strip():
        _log("No speech detected. Skipping translation and rendering.")
        return None

    # Save original transcription SRT for debugging
    base = video_path.rsplit(".", 1)[0]
    debug_original = base + "_debug_whisper_original.srt"
    with open(debug_original, "w", encoding="utf-8") as f:
        f.write(srt_content)
    _log(f"   Debug: Whisper SRT saved as {os.path.basename(debug_original)}")

    # --- Step 3: Translate ---
    _log(f"Step 3/4: Translating to {target_lang}...")
    translator = AITranslator(model=translator_model)
    translated_srt = translator.translate_srt_content(srt_content, target_lang, progress_callback=progress_callback)
    translator.unload()

    if not translated_srt or not translated_srt.strip():
        _log("Translation returned empty result. Using original transcription.")
        translated_srt = srt_content

    # Save translated SRT for debugging
    debug_translated = base + "_debug_whisper_translated.srt"
    with open(debug_translated, "w", encoding="utf-8") as f:
        f.write(translated_srt)
    _log(f"   Debug: Translated SRT saved as {os.path.basename(debug_translated)}")

    # --- Step 4: Render subtitles ---
    _log("Step 4/4: Rendering subtitles onto video...")
    result = render_subtitles(video_path, translated_srt, progress_callback)

    elapsed = time.time() - overall_start
    if result:
        _log(f"Audio Pipeline complete in {format_eta(elapsed)}")
    else:
        _log(f"Audio Pipeline failed after {format_eta(elapsed)}")

    return result
