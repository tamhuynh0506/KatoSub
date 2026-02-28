import datetime
import difflib

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format: HH:MM:SS,mmm"""
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def is_similar(text1, text2, threshold=0.8):
    """Check if two strings are similar using SequenceMatcher."""
    if not text1 or not text2: return False
    return difflib.SequenceMatcher(None, text1, text2).ratio() > threshold

def get_stabilized_segments(ocr_history, fps):
    """
    Group OCR history into stabilized segments with timing and bounding boxes.
    Returns a list of dicts: {'start_frame', 'last_frame', 'text', 'boxes'}
    """
    if not ocr_history:
        return []

    segments = []
    current_segment = None
    
    # Filter out empty or whitespace-only results first
    clean_history = [e for e in ocr_history if e['text'].strip()]
    if not clean_history: return []

    for entry in clean_history:
        text = entry['text'].strip()
        frame = entry['frame']
        boxes = entry['boxes']
        
        if current_segment is None:
            current_segment = {
                'text': text,
                'start_frame': frame,
                'last_frame': frame,
                'boxes': boxes[:]
            }
        else:
            time_gap = (frame - current_segment['last_frame']) / fps
            
            # If text is similar OR the gap is small and text isn't wildly different
            if is_similar(text, current_segment['text']) or (time_gap < 0.4 and len(text) > 0):
                # Update text to the "longest" version
                if len(text) > len(current_segment['text']):
                    current_segment['text'] = text
                
                current_segment['last_frame'] = frame
                # Accrue boxes to ensure we cover the entire area over the segment's life
                current_segment['boxes'].extend(boxes)
            else:
                segments.append(current_segment)
                current_segment = {
                    'text': text,
                    'start_frame': frame,
                    'last_frame': frame,
                    'boxes': boxes[:]
                }
    
    if current_segment:
        segments.append(current_segment)

    # Post-process: add timing padding (e.g. 0.2s) to prevent flicker at start/end
    padding_frames = int(0.2 * fps)
    for seg in segments:
        seg['start_frame'] = max(0, seg['start_frame'] - padding_frames)
        # We don't necessarily know the total frames here, but inpaint_and_render will handle it
        seg['last_frame'] = seg['last_frame'] + padding_frames

    return segments

def frames_to_srt(ocr_history, fps):
    """
    Convert OCR history into stabilized SRT entries using segments.
    """
    segments = get_stabilized_segments(ocr_history, fps)
    if not segments:
        return ""

    output = []
    idx = 1
    for seg in segments:
        start_sec = seg['start_frame'] / fps
        end_sec = (seg['last_frame'] + 1) / fps
        duration = end_sec - start_sec
        
        if duration < 0.1: continue
        
        start = format_timestamp(start_sec)
        end = format_timestamp(end_sec)
        output.append(f"{idx}")
        output.append(f"{start} --> {end}")
        output.append(f"{seg['text']}\n")
        idx += 1
        
    return "\n".join(output)
