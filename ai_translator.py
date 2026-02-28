from google import genai
from google.genai import types

class AITranslator:
    def __init__(self, api_key):
        if not api_key:
            raise ValueError("Gemini API Key is required for AI translation.")
        self.client = genai.Client(api_key=api_key)
        self.model = 'gemini-2.0-flash'
        print(f"DEBUG: Using Google GenAI SDK with model: {self.model}")

    def translate_srt_content(self, srt_content, source_lang, target_lang, custom_prompt=None):
        if not self.client:
            return srt_content

        prompt = custom_prompt if custom_prompt else f"""
        Translate the following subtitles from {source_lang} to {target_lang}.
        Maintain the SRT structure (timing and indices) perfectly. 
        Only return the translated SRT content.
        Subtitles:
        {srt_content}
        """
        
        try:
            # safety settings to avoid filtering
            safety_settings = [
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.BLOCK_NONE,
                ),
            ]
            
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=safety_settings,
                )
            )
            
            if response and hasattr(response, 'text') and response.text:
                return response.text.strip()
            
            return srt_content
        except Exception as e:
            print(f"DEBUG: AI Call failed ({self.model}): {e}")
            return srt_content
