"""
Replace Subs (Full) Pipeline
─────────────────────────────
Hybrid pipeline that:
  1. OCR-detects existing hardcoded subtitles
  2. Inpaints (removes) them from the video
  3. Extracts audio from the *original* video
  4. Transcribes with Whisper (medium model)
  5. Translates the transcription
  6. Renders new translated subtitles onto the clean video

Reuses existing modules — no code duplication.
"""

import os
import time
import subprocess

from pipeline_v4 import SelectiveInpaintPipe, format_eta
from pipeline_audio import extract_audio, transcribe_audio, render_subtitles
from ai_translator import AITranslator


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_video_duration_ffprobe(video_path: str) -> float:
    """Fast duration query via ffprobe (seconds). Returns 0 on failure."""
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]
        out = subprocess.run(cmd, capture_output=True, encoding="utf-8",
                             errors="replace", timeout=5)
        if out.returncode == 0:
            return float(out.stdout.strip())
    except Exception:
        pass
    return 0.0


def _free_gpu_memory():
    """Aggressively free VRAM so the next model can load."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


# ─── Core pipeline ────────────────────────────────────────────────────────────

def run_replace_subs_pipeline(
    video_path: str,
    target_lang: str,
    translator_model: str = "google",
    progress_callback=None,
    output_dir=None,
) -> str | None:
    """
    Full "Replace Subs" pipeline.

    Returns the path to the final output video, or *None* on failure.
    """
    def _log(msg: str):
        if progress_callback:
            progress_callback(msg)

    _log("Replace Subs Pipeline: Starting...")
    overall_start = time.time()

    # ── Step 1/6: OCR detect subtitle regions ────────────────────────────
    _log("Step 1/6: Detecting existing subtitles (OCR)...")
    pipe = SelectiveInpaintPipe()
    ocr_history, fps = pipe.extract_metadata(video_path, progress_callback)

    from srt_utils import get_stabilized_segments
    segments = get_stabilized_segments(ocr_history, fps)
    _log(f"   OCR detected text on {len(ocr_history)} frames → "
         f"{len(segments)} subtitle segments")

    if not segments:
        _log("   ⚠ No existing subtitles detected — skipping inpainting, "
             "falling back to audio-only flow.")
        # Fast-path: behave like audio-only pipeline
        return _audio_only_fallback(
            video_path, target_lang,
            translator_model, progress_callback,
            output_dir=output_dir
        )

    # ── Step 2/6: Inpaint (remove old subs) ──────────────────────────────
    _log("Step 2/6: Removing old subtitles (AI Inpainting)...")
    # Pass empty SRT so no new subs are overlaid during inpainting
    clean_video = pipe.inpaint_and_render(
        video_path, segments, translated_srt="", progress_callback=progress_callback,
    )

    # Free inpainting model from GPU before loading Whisper
    del pipe
    _free_gpu_memory()

    if not clean_video or not os.path.exists(clean_video):
        _log("❌ Inpainting failed — cannot continue.")
        return None

    _log(f"   Clean video: {os.path.basename(clean_video)}")

    # ── Step 3/6: Extract audio (from *original* — cleaner than re-encode) ─
    _log("Step 3/6: Extracting audio...")
    audio_path = extract_audio(video_path, progress_callback)
    if not audio_path:
        _log("❌ Audio extraction failed!")
        return None

    # ── Step 4/6: Transcribe with Whisper medium ─────────────────────────
    _log("Step 4/6: Transcribing audio (Whisper medium)...")
    srt_content, detected_lang = transcribe_audio(
        audio_path, whisper_model_name="medium", progress_callback=progress_callback,
    )

    # Clean up temp audio
    _safe_remove(audio_path)

    if not srt_content or not srt_content.strip():
        _log("⚠ No speech detected — output will be the clean (sub-free) video.")
        return clean_video

    # Debug: save raw Whisper SRT
    base = video_path.rsplit(".", 1)[0]
    _save_debug_srt(base + "_debug_replacesubs_whisper.srt", srt_content, _log)

    # ── Step 5/6: Translate ──────────────────────────────────────────────
    _log(f"Step 5/6: Translating to {target_lang}...")
    translator = AITranslator(model=translator_model)
    translated_srt = translator.translate_srt_content(
        srt_content, target_lang, progress_callback=progress_callback
    )
    translator.unload()

    if not translated_srt or not translated_srt.strip():
        _log("⚠ Translation returned empty — using original transcription.")
        translated_srt = srt_content

    # Debug: save translated SRT
    _save_debug_srt(base + "_debug_replacesubs_translated.srt", translated_srt, _log)

    # ── Step 6/6: Render new subs onto clean video ───────────────────────
    _log("Step 6/6: Rendering new subtitles onto clean video...")
    final_output = render_subtitles(clean_video, translated_srt, progress_callback, output_dir=output_dir)

    # Clean up the intermediate inpainted video (keep only final)
    if final_output and os.path.exists(final_output) and final_output != clean_video:
        _safe_remove(clean_video)

    elapsed = time.time() - overall_start
    if final_output:
        _log(f"Replace Subs Pipeline complete in {format_eta(elapsed)}")
    else:
        _log(f"Replace Subs Pipeline failed after {format_eta(elapsed)}")

    return final_output


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _audio_only_fallback(video_path, target_lang, translator_model, progress_callback, output_dir=None):
    """When no existing subs are detected, just run the audio pipeline."""
    from pipeline_audio import run_audio_pipeline
    return run_audio_pipeline(
        video_path, target_lang,
        translator_model=translator_model,
        whisper_model="medium",
        progress_callback=progress_callback,
        output_dir=output_dir,
    )


def _save_debug_srt(path: str, content: str, _log):
    """Write an SRT file for debugging/inspection."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        _log(f"   Debug: saved {os.path.basename(path)}")
    except Exception as exc:
        _log(f"   ⚠ Could not save debug SRT: {exc}")


def _safe_remove(path: str):
    """Delete a file if it exists, silently ignoring errors."""
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
