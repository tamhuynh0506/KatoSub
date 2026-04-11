import torch
from pyannote.audio import Pipeline

def run_diarization(audio_path, hf_token, progress_callback=None):
    """
    Run pyannote.audio speaker diarization on the given audio.
    Returns a pyannote Annotation object containing speaker segments.
    """
    def _log(msg):
        if progress_callback:
            progress_callback(msg)

    _log("Loading Diarization Model (pyannote/speaker-diarization-3.1)...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
    except Exception as e:
        _log(f"   ❌ Diarization Init Error: {e}")
        return None

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipeline.to(device)
    _log(f"   Diarization model loaded on {device}.")

    _log("Running speaker diarization (this may take a while)...")
    try:
        diarization = pipeline(audio_path)
        _log("   ✅ Diarization complete.")
        return diarization
    except Exception as e:
        _log(f"   ❌ Diarization Execution Error: {e}")
        return None

def merge_whisper_speakers(whisper_segments, diarization):
    """
    Given a list of Whisper segments and a pyannote diarization object,
    assigns the most dominant 'speaker' label to each segment.
    """
    if not diarization:
        return whisper_segments

    # Convert pyannote annotation into a flat list of (start, end, speaker)
    turns = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        turns.append((turn.start, turn.end, speaker))

    for segment in whisper_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        
        # Calculate overlap for each speaker
        speaker_duration = {}
        for d_start, d_end, speaker in turns:
            # Overlap calculation
            overlap_start = max(seg_start, d_start)
            overlap_end = min(seg_end, d_end)
            if overlap_start < overlap_end:
                overlap_time = overlap_end - overlap_start
                speaker_duration[speaker] = speaker_duration.get(speaker, 0) + overlap_time

        if speaker_duration:
            # Assign the speaker with maximum overlap
            best_speaker = max(speaker_duration.items(), key=lambda x: x[1])[0]
            segment["speaker"] = best_speaker
        else:
            segment["speaker"] = "Unknown"

    return whisper_segments
