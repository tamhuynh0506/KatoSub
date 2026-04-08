import subprocess
import json
import sys
import os

# Ensure UTF-8 for output
# sys.stdout.reconfigure(encoding='utf-8')

def run_split_pipeline():
    video = r'C:\lil_summer\code\KatoSub\test\Blue Fight- Aoki Wakamonotachi no Breaking Down (2025) Episode 1 English SUB - Dramacool - Trim.mp4'
    
    # Stage 1: Detection (Subprocess to avoid Paddle/Torch conflict)
    print("--- Stage 1: Watermark Detection (PaddleOCR) ---")
    
    # We use a standalone detection script
    detect_script = f"""
import sys
import json
from watermark_detector import WatermarkDetector
from dataclasses import asdict

# Ensure UTF-8 for internal stdout
# sys.stdout.reconfigure(encoding='utf-8')

try:
    detector = WatermarkDetector()
    regions = detector.detect(r'{video}')
    # Convert to dict for JSON serialization
    data = [asdict(r) for r in regions]
    print('JSON_START' + json.dumps(data) + 'JSON_END')
except Exception as e:
    print('ERROR:' + str(e))
    sys.exit(1)
"""
    result = subprocess.run([sys.executable, "-c", detect_script], capture_output=True, text=True, encoding='utf-8')
    
    if result.returncode != 0:
        print("Detection failed:")
        print(result.stdout)
        print(result.stderr)
        return

    output = result.stdout
    if 'JSON_START' not in output:
        print("Could not find detection results in output.")
        print(output)
        return
        
    json_str = output.split('JSON_START')[1].split('JSON_END')[0]
    regions_data = json.loads(json_str)
    
    from watermark_detector import WatermarkRegion
    regions = [WatermarkRegion(**r) for r in regions_data]
    print(f"--- Stage 1 Complete: Detected {len(regions)} regions ---")

    # Stage 2: Inpainting (Fresh process)
    print("--- Stage 2: AI Inpainting (PyTorch) ---")
    from pipeline_watermark import run_watermark_pipeline
    try:
        run_watermark_pipeline(video, progress_callback=lambda msg: print(msg), output_dir='test', regions=regions)
        print("✅ Pipeline finished successfully.")
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_split_pipeline()
