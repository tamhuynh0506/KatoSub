import os
import time
import sys
from ai_translator import AITranslator

# Ensure stdout and stderr handle UTF-8 (important for Windows consoles)
if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

def test_srt_translation():
    input_file = r"C:\Users\betam\Downloads\1_test_debug_original.srt"
    output_file = "1_test_debug_translated.srt"
    target_language = "Vietnamese" # Based on LANG_MAP logic in ai_translator.py

    print("--- Ollama SRT Translation Test ---")
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        return

    with open(input_file, "r", encoding="utf-8") as f:
        srt_content = f.read()

    print(f"Read {input_file} ({len(srt_content)} bytes)")
    
    # Initialize translator with Ollama Gemma 3 12B
    print("Initializing AITranslator with model='ollama:gemma3:12b'...")
    translator = AITranslator(model="ollama:gemma3:12b")
    
    if not translator.model.startswith("ollama"):
        print(f"Warning: Translator fell back to {translator.model}. Check if Ollama is running.")

    start_time = time.time()
    print(f"Starting translation to {target_language}...")
    
    try:
        translated_content = translator.translate_srt_content(srt_content, target_language)
        
        duration = time.time() - start_time
        print(f"\nTranslation completed in {duration:.2f} seconds.")
        
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(translated_content)
            
        print(f"Saved translated content to {output_file}")
        
        print("\n--- Preview of Translation ---")
        # Ensure we don't crash on preview either
        preview = translated_content[:500]
        print(preview + ("..." if len(translated_content) > 500 else ""))
        
    except Exception as e:
        import traceback
        print(f"An error occurred during translation: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_srt_translation()
