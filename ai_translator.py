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

def format_eta(seconds):
    if seconds > 3600:
        return f"{int(seconds // 3600)}:{(int(seconds % 3600) // 60):02}:{(int(seconds % 60)):02}"
    else:
        return f"{(int(seconds // 60)):02}:{(int(seconds % 60)):02}"


# Language name -> Google Translate language code
LANG_MAP = {
    "en": "en", "ja": "ja", "ko": "ko", "zh-cn": "zh-CN", "zh-tw": "zh-TW",
    "fr": "fr", "de": "de", "es": "es", "pt": "pt", "ru": "ru", "vi": "vi",
    "it": "it", "id": "id", "th": "th", "hi": "hi", "ar": "ar",
}

class AITranslator:
    def __init__(self, model="google"):
        """Translator supporting Google Translate, ChatGPT, and Ollama (Gemma 3)."""
        self.model = model
        self.client = None
        self.ollama_model = None

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

        elif self.model.startswith("ollama:"):
            # e.g. "ollama:gemma3:12b" -> ollama_model = "gemma3:12b"
            self.ollama_model = self.model.split("ollama:", 1)[1]
            if not OpenAI:
                print("DEBUG: OpenAI library not installed (needed for Ollama client). Falling back to Google.")
                self.model = "google"
            else:
                print(f"DEBUG: Initializing Ollama Translator with model: {self.ollama_model}")
                import requests
                found = False
                for host in ["localhost", "127.0.0.1"]:
                    try:
                        test_resp = requests.get(f"http://{host}:11434/api/tags", timeout=3)
                        if test_resp.status_code == 200:
                            print(f"DEBUG: Ollama server found on {host}")
                            self.client = OpenAI(base_url=f"http://{host}:11434/v1", api_key="ollama", timeout=120.0)
                            found = True
                            break
                    except Exception:
                        continue
                if not found:
                    print("DEBUG: Ollama server NOT reachable. Falling back to Google.")
                    self.model = "google"

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

    def extract_character_context(self, texts, target_lang):
        """Run one focused AI call on a sample of the full transcript to extract
        a compact Character Sheet: names, relationships, and show tone.
        Returns a plain-text string, or None if unavailable / failed."""
        if not texts or len(texts) < 3:
            return None

        # Sample up to 200 lines spread across beginning / middle / end
        # to stay within token limits while covering the full story arc.
        max_sample = 200
        if len(texts) <= max_sample:
            sample = texts
        else:
            step = len(texts) / max_sample
            sample = [texts[int(i * step)] for i in range(max_sample)]

        transcript_snippet = " ".join([t.replace("\n", " ") for t in sample])

        prompt = (
            f"Read this TV/movie subtitle transcript excerpt and identify:\n"
            f"1. All character names (with gender, age/role hints if discernible)\n"
            f"2. Key relationships between characters (e.g. husband/wife, boss/employee)\n"
            f"3. Overall show tone (e.g. romantic comedy, action thriller, historical drama)\n"
            f"4. Any important recurring terms, titles, or honorifics\n\n"
            f"Be concise. Use plain text only — no markdown, no JSON.\n"
            f"If you cannot determine something with reasonable confidence, omit it.\n\n"
            f"Target translation language: {target_lang}\n"
            f"(Hint: correct pronoun/honorific choices in {target_lang} depend heavily on "
            f"the speakers' ages and relationships, so be as specific as possible.)\n\n"
            f"--- TRANSCRIPT SAMPLE ---\n{transcript_snippet}\n--- END SAMPLE ---"
        )

        system_msg = (
            "You are an expert script analyst for TV and film. "
            "Your sole task is to produce a concise, accurate Character Sheet "
            "from the provided subtitle transcript."
        )

        try:
            print("DEBUG: Extracting character context...")
            response = self.client.chat.completions.create(
                model=self.ollama_model if self.ollama_model else "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            result = response.choices[0].message.content.strip()
            if result:
                print(f"DEBUG: Character context extracted ({len(result)} chars)")
                return result
        except Exception as e:
            print(f"DEBUG: Character extraction failed (non-fatal): {e}")
        return None

    def _translate_batch_chatgpt(self, texts, target_lang, all_texts=None, prev_translations=None, character_context=None):
        prompt = f"""Translate the following JSON array of subtitle texts to {target_lang}. Return ONLY a valid JSON array of strings containing the translations in the exact same order.
            Guidelines:
            - Correct translation mistakes and improve grammar.
            - Rewrite sentences to sound natural, conversational, and fluent.
            - Maintain consistent terminology for character names, titles, locations, skills/abilities, organizations, and important objects or concepts.
            - Keep subtitles concise and readable. Remove redundant words if necessary, but do not remove important meaning.
            - The source text is generated by an OCR engine and may contain typos, severe misspellings, or merged characters (e.g. "genilal" instead of "genial", "rn" instead of "m"). If a word is clearly incorrect or meaningless, you MUST infer the intended word based on context and translate that intended word.
            - Translate ALL non-name words to the target language. Do not leave misspelled source words in the translated text.
            - CRITICALLY IMPORTANT: If a subtitle text contains multiple lines (separated by \n) or dialogue markers (like "- "), YOU MUST PRESERVE the exact same number of lines and markers in your translation. Do NOT merge them into a single line.
            - CRITICALLY IMPORTANT: Return ONLY a valid JSON array of strings. Do not include any explanations, markdown formatting like ```json, or other text.
        """
        
        if all_texts and len(all_texts) > 0:
            prompt += "--- REFERENCE CONTEXT (Full Transcript) ---\n"
            prompt += "The following is the full video transcript to help you establish a 'Terminology Memory'. Use this context to ensure the same translations for names and special terms are used consistently, and to understand the story, character relationships, and tone:\n"
            prompt += " ".join([t.replace('\n', ' ') for t in all_texts])
            prompt += "\n--------------------------\n\n"

        if character_context:
            prompt += "--- CHARACTER SHEET (auto-extracted) ---\n"
            prompt += "Use the character names, relationships, and tone below to choose the correct pronouns, honorifics, and register for every translated line:\n"
            prompt += character_context
            prompt += "\n--------------------------\n\n"

        if prev_translations and len(prev_translations) > 0:
            prompt += "--- RECENTLY TRANSLATED (for continuity) ---\n"
            prompt += "These are the most recent lines you already translated. Continue with the SAME style, character names, and terminology:\n"
            for pair in prev_translations:
                prompt += f"  Source: {pair['source']}\n  Translation: {pair['translated']}\n\n"
            prompt += "--------------------------\n\n"
            
        prompt += "--- TEXTS TO TRANSLATE NOW ---\n"
        prompt += json.dumps(texts, ensure_ascii=False)
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a professional subtitle translator and editor working on TV, Anime, and Drama series. Your task is to carefully review and improve subtitles while maintaining absolute consistency for character names, locations, and special terms across the entire file. You ONLY return a JSON array string of the translations."},
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

    def _translate_batch_ollama(self, texts, target_lang, all_texts=None, prev_translations=None, character_context=None):
        """Translate a batch of subtitle texts using the local Ollama model (Gemma 3)."""
        system_msg = (
            "You are a professional subtitle translator and editor working on TV, Anime, and Drama series. "
            "Your task is to carefully review and improve subtitles while maintaining absolute consistency "
            "for character names, locations, and special terms across the entire file. "
            "You ONLY return a JSON array of strings containing the translations."
        )

        prompt = f"""Translate the following JSON array of subtitle texts to {target_lang}. Return ONLY a valid JSON array of strings containing the translations in the exact same order.
Guidelines:
- Correct translation mistakes and improve grammar.
- Rewrite sentences to sound natural, conversational, and fluent — like a real TV drama subtitle.
- Maintain consistent terminology for character names, titles, locations, and important terms.
- Keep subtitles concise and readable.
- The source text is generated by an OCR engine and may contain typos, severe misspellings, or merged characters (e.g. "genilal" instead of "genial", "rn" instead of "m"). If a word is clearly incorrect or meaningless, you MUST infer the intended word based on context and translate that intended word.
- Translate ALL non-name words to the target language. Do not leave misspelled source words in the translated text.
- CRITICALLY IMPORTANT: If a subtitle text contains multiple lines (separated by \n) or dialogue markers (like "- "), YOU MUST PRESERVE the exact same number of lines and markers in your translation. Do NOT merge them into a single line.
- CRITICALLY IMPORTANT: Return ONLY a valid JSON array of {len(texts)} strings. No explanations, no markdown, no extra text.
"""

        if all_texts and len(all_texts) > 0:
            prompt += "--- REFERENCE CONTEXT (Full Transcript) ---\n"
            prompt += "The following is the full video transcript to help you establish a 'Terminology Memory'. Use this context to ensure the same translations for names and special terms are used consistently, and to understand the story, character relationships, and tone:\n"
            prompt += " ".join([t.replace('\n', ' ') for t in all_texts])
            prompt += "\n--------------------------\n\n"

        if character_context:
            prompt += "--- CHARACTER SHEET (auto-extracted) ---\n"
            prompt += "Use the character names, relationships, and tone below to choose the correct pronouns, honorifics, and register for every translated line:\n"
            prompt += character_context
            prompt += "\n--------------------------\n\n"

        if prev_translations and len(prev_translations) > 0:
            prompt += "--- RECENTLY TRANSLATED (for continuity) ---\n"
            prompt += "These are the most recent lines you already translated. Continue with the SAME style, character names, and terminology:\n"
            for pair in prev_translations:
                prompt += f"  Source: {pair['source']}\n  Translation: {pair['translated']}\n\n"
            prompt += "--------------------------\n\n"

        prompt += f"--- TEXTS TO TRANSLATE NOW (return exactly {len(texts)} strings) ---\n"
        prompt += json.dumps(texts, ensure_ascii=False)

        for attempt in range(3):
            try:
                print(f"DEBUG: Waiting for Ollama ({self.ollama_model}) response (Batch {len(texts)} lines)...")
                response = self.client.chat.completions.create(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                content = response.choices[0].message.content.strip()

                # Strip markdown fences if present
                if content.startswith("```json"):
                    content = content[7:-3].strip()
                elif content.startswith("```"):
                    content = content[3:-3].strip()

                # Try to extract JSON array from the response
                # Sometimes models wrap the array in extra text
                json_match = re.search(r'\[.*\]', content, re.DOTALL)
                if json_match:
                    content = json_match.group(0)

                result = json.loads(content)
                if isinstance(result, list) and len(result) == len(texts):
                    return result
                else:
                    print(f"DEBUG: Ollama length mismatch ({len(result)} vs {len(texts)}), retrying...")
            except json.JSONDecodeError as e:
                print(f"DEBUG: Ollama JSON parse error on attempt {attempt+1}: {e}")
                print(f"DEBUG: Raw response: {content[:200]}...")
                time.sleep(1)
            except Exception as e:
                err_msg = str(e)
                print(f"DEBUG: Ollama translation error on attempt {attempt+1}: {err_msg}")
                # Check for OOM / Memory limit errors specifically
                if "more system memory" in err_msg.lower() or "not enough memory" in err_msg.lower():
                    print("🚨 ERROR: Model too large for your RAM/VRAM. Switching to individual Google fallback.")
                    print("💡 TIP: Try choosing the smaller 'Ollama (gemma3:4b)' model in Project Settings.")
                    break # Stop retrying, memory won't magically appear
                time.sleep(2)
        return None


    def translate_srt_content(self, srt_content, target_lang, progress_callback=None):
        if not srt_content.strip():
            return srt_content

        def _log(msg):
            if progress_callback:
                progress_callback(msg)

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
                text = "\n".join(lines[2:]).strip()  # Preserve multi-line dialogue
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

        # ── Step 0: Extract character context (one AI call before any batches) ──
        character_context = None
        if self.model != "google":
            _log("Analyzing characters and relationships...")
            character_context = self.extract_character_context(texts, target_lang)
            if character_context:
                _log(f"✓ Character sheet extracted ({len(character_context)} chars)")
                print(f"DEBUG: Character sheet:\n{character_context}")
            else:
                _log("⚠ Character extraction skipped or failed — continuing without it")

        start_time = time.time()

        # Sliding window context sizes
        if self.model == "chatgpt":
            BATCH_SIZE = 30
            CONTEXT_WINDOW = 8  # Last 8 translated pairs as context for next batch
        elif self.model.startswith("ollama:"):
            BATCH_SIZE = 5
            CONTEXT_WINDOW = 3  # Smaller window for local models
        else:
            BATCH_SIZE = 10
            CONTEXT_WINDOW = 0  # Google Translate doesn't use context
        translated_texts = []
        
        for batch_start in range(0, total, BATCH_SIZE):
            batch_end = min(batch_start + BATCH_SIZE, total)
            batch = texts[batch_start:batch_end]

            # Build sliding window from previously translated lines
            prev_translations = None
            if CONTEXT_WINDOW > 0 and len(translated_texts) > 0:
                window_start = max(0, len(translated_texts) - CONTEXT_WINDOW)
                prev_translations = []
                for i in range(window_start, len(translated_texts)):
                    prev_translations.append({
                        "source": texts[i],
                        "translated": translated_texts[i]
                    })
            
            if self.model == "chatgpt":
                parts = self._translate_batch_chatgpt(batch, target_lang, all_texts=texts, prev_translations=prev_translations, character_context=character_context)
                if parts:
                    translated_texts.extend(parts)
                else:
                    print(f"DEBUG: Batch {batch_start}-{batch_end} failed with ChatGPT, falling back to Google individually")
                    for text in batch:
                        individual = self._translate_with_retry(text, 'auto', tgt_code)
                        translated_texts.append(individual if individual else text)
                        time.sleep(1.0)
            elif self.model.startswith("ollama:"):
                parts = self._translate_batch_ollama(batch, target_lang, all_texts=texts, prev_translations=prev_translations, character_context=character_context)
                if parts:
                    translated_texts.extend(parts)
                else:
                    print(f"DEBUG: Batch {batch_start}-{batch_end} failed with Ollama, falling back to Google individually")
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
            

            elapsed = time.time() - start_time
            speed = batch_end / elapsed if elapsed > 0 else 0
            eta_str = "Calculating..."
            if speed > 0:
                eta_sec = (total - batch_end) / speed
                eta_str = format_eta(eta_sec)

            _log(f"Translating to {target_lang}: {int(batch_end/total*100)}% ({batch_end}/{total}) | ETA: {eta_str}")
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
        _log(f"Translation complete: {success}/{total} blocks translated")

        return "\n\n".join(translated_blocks)

    def unload(self):
        """No-op for Google Translate (no model to unload)."""
        pass
