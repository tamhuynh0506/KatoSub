import re
import time
import json
from deep_translator import GoogleTranslator
try:
    import config
except ImportError:
    config = None

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Language name -> Google Translate language code
LANG_MAP = {
    "en": "en", "ja": "ja", "ko": "ko", "zh-cn": "zh-CN", "zh-tw": "zh-TW",
    "fr": "fr", "de": "de", "es": "es", "pt": "pt", "ru": "ru", "vi": "vi",
    "it": "it", "id": "id", "th": "th", "hi": "hi", "ar": "ar",
}

class AITranslator:
    def __init__(self, model="google"):
        """Translator supporting Google Translate and ChatGPT."""
        self.model = model
        if self.model == "chatgpt":
            if not OpenAI:
                print("DEBUG: OpenAI library not installed. Falling back to Google.")
                self.model = "google"
            elif not config or not hasattr(config, "OPENAI_API_KEY") or not config.OPENAI_API_KEY:
                print("DEBUG: OPENAI_API_KEY not found in config. Falling back to Google.")
                self.model = "google"
            else:
                print("DEBUG: Initializing ChatGPT Translator")
                self.client = OpenAI(api_key=config.OPENAI_API_KEY)
        
        if self.model == "google":
            print("DEBUG: Initializing Google Translate (Unlimited, Free)")

    def _translate_with_retry(self, text, src, tgt, max_retries=5):
        """Translate a single text with retries and exponential backoff."""
        for attempt in range(max_retries):
            try:
                result = GoogleTranslator(source=src, target=tgt).translate(text)
                if result:
                    return result
            except Exception as e:
                err = str(e)
                if attempt < max_retries - 1:
                    wait = 2 ** attempt  # 1s, 2s, 4s, 8s, 16s
                    print(f"DEBUG: Translate retry {attempt+1}/{max_retries} (wait {wait}s): {err[:80]}")
                    time.sleep(wait)
                else:
                    print(f"DEBUG: Translation failed after {max_retries} retries: {err[:80]}")
        return None

    def _translate_batch_chatgpt(self, texts, target_lang):
        prompt = f"Translate the following JSON array of subtitle texts to {target_lang}. Return ONLY a valid JSON array of strings containing the translations in the exact same order. Do not include any explanations, markdown formatting like ```json, or other text.\n\n"
        prompt += json.dumps(texts, ensure_ascii=False)
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional subtitle translator. You ONLY return a JSON array string."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                content = response.choices[0].message.content.strip()
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()
                
                result = json.loads(content)
                if isinstance(result, list) and len(result) == len(texts):
                    return result
                else:
                    print(f"DEBUG: ChatGPT length mismatch ({len(result)} vs {len(texts)}), retrying...")
            except Exception as e:
                print(f"DEBUG: ChatGPT translation error on attempt {attempt+1}: {e}")
                time.sleep(1)
        return None

    def translate_srt_content(self, srt_content, target_lang):
        if not srt_content.strip():
            return srt_content

        # Always auto-detect source language — OCR text may differ from the user's selection
        # (e.g., user selects "English" but video has Spanish/Japanese burned-in subs)
        tgt_code = LANG_MAP.get(target_lang, "vi")

        print(f"DEBUG: Translating to {target_lang} ({tgt_code}) with auto-detection")

        # Parse SRT into blocks and extract text
        blocks = re.split(r'\n\n+', srt_content.strip())
        
        headers = []
        texts = []
        block_map = []
        
        for block in blocks:
            lines = block.split('\n')
            if len(lines) >= 3:
                header = lines[:2]
                text = " ".join(lines[2:]).strip()
                if text:
                    headers.append(header)
                    texts.append(text)
                    block_map.append(('translate', len(headers) - 1))
                else:
                    block_map.append(('passthrough', block))
            else:
                block_map.append(('passthrough', block))

        if not texts:
            return srt_content

        total = len(texts)
        print(f"DEBUG: Found {total} subtitle blocks to translate")

        # Translate in small batches using newline as separator
        # Google Translate preserves newlines reliably
        BATCH_SIZE = 30 if self.model == "chatgpt" else 10
        translated_texts = []
        
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = texts[batch_start:batch_end]
            
            if self.model == "chatgpt":
                parts = self._translate_batch_chatgpt(batch, target_lang)
                if parts:
                    translated_texts.extend(parts)
                else:
                    print(f"DEBUG: Batch {batch_start}-{batch_end} failed with ChatGPT, falling back to Google individually")
                    for text in batch:
                        individual = self._translate_with_retry(text, 'auto', tgt_code)
                        translated_texts.append(individual if individual else text)
                        time.sleep(1.0)
            else:
                # Join with newlines - Google Translate preserves these
                joined = "\n".join(batch)
                
                result = self._translate_with_retry(joined, 'auto', tgt_code)
                
                if result:
                    parts = result.split("\n")
                    # Clean empty parts that might come from extra newlines
                    parts = [p.strip() for p in parts if p.strip()]
                    
                    if len(parts) == len(batch):
                        translated_texts.extend(parts)
                    else:
                        # Newline split didn't match - translate individually
                        print(f"DEBUG: Batch {batch_start}-{batch_end} split mismatch ({len(parts)} vs {len(batch)}), translating individually")
                        for text in batch:
                            individual = self._translate_with_retry(text, 'auto', tgt_code)
                            translated_texts.append(individual if individual else text)
                            time.sleep(1.0)
                else:
                    # Batch failed entirely - translate individually
                    print(f"DEBUG: Batch {batch_start}-{batch_end} failed, translating individually")
                    for text in batch:
                        individual = self._translate_with_retry(text, 'auto', tgt_code)
                        translated_texts.append(individual if individual else text)
                        time.sleep(1.0)
            
            print(f"DEBUG: Translated {min(batch_end, total)}/{total} blocks")
            time.sleep(0.5)  # Delay between batches

        # Reassemble SRT
        translated_blocks = []
        for entry in block_map:
            if entry[0] == 'translate':
                idx = entry[1]
                header = headers[idx]
                translated = translated_texts[idx] if idx < len(translated_texts) else texts[idx]
                translated_blocks.append("\n".join(header + [translated]))
            else:
                translated_blocks.append(entry[1])

        success = sum(1 for i in range(len(translated_texts)) if i < len(texts) and translated_texts[i] != texts[i])
        print(f"DEBUG: ✅ Translation complete: {success}/{total} blocks translated")

        return "\n\n".join(translated_blocks)

    def unload(self):
        """No-op for Google Translate (no model to unload)."""
        pass
