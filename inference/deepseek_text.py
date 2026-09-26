"""Send text to DeepSeek (OpenAI-compatible chat API) and get text back.

Same request as telegram-server/deepseek_client.py `_pedir`. The API key comes from the
DEEPSEEK_API_KEY environment variable, or from DEEPSEEK_API_KEY=... in this pack's .env file.

Replies are cached on disk by (model, instruction, text, temperature), so re-running a workflow
does not pay for the same answer twice.
"""
import hashlib
import json
import os
import re
import urllib.error
import urllib.request

import folder_paths

API_URL = "https://api.deepseek.com/chat/completions"
MODELS = ["deepseek-v4-flash", "deepseek-v4-pro"]
PACK_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INSTRUCTION = ("coloque essa letra com tags de seção[verse] [chorus] [outro] [bridge] etc. "
                       "plain text e nada mais")
SECTION_TAG = re.compile(r"^\s*\[[^\]]+\]\s*$", re.M)


def _api_key():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if key:
        return key.strip()
    try:
        with open(os.path.join(PACK_DIR, ".env"), encoding="utf-8") as f:
            for line in f:
                name, _, value = line.partition("=")
                if name.strip() == "DEEPSEEK_API_KEY":
                    return value.strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def _cache_file(model, instruction, text, temperature):
    digest = hashlib.sha256(json.dumps([model, instruction, text, temperature]).encode("utf-8")).hexdigest()
    return os.path.join(folder_paths.get_user_directory(), "deepseek_cache", digest[:32] + ".txt")


def _strip_fences(text):
    """Drop a ``` code fence the model sometimes wraps plain text in."""
    m = re.fullmatch(r"\s*```[^\n]*\n(.*?)\n```\s*", text, re.S)
    return m.group(1) if m else text


def ask_deepseek(model, instruction, text, temperature):
    key = _api_key()
    if not key:
        raise RuntimeError(f"DEEPSEEK_API_KEY not found in the environment or in {os.path.join(PACK_DIR, '.env')}.")
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{instruction}\n\n{text}"}],
        "temperature": temperature,
        "max_tokens": 8192,
        "stream": False,
        # deepseek-v4 reasons by default and the reasoning counts against max_tokens: a long
        # deliberation can use it all and return 200 with empty content. Only this exact form disables it.
        "thinking": {"type": "disabled"},
    }
    req = urllib.request.Request(API_URL, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"DeepSeek HTTP {e.code}: {e.read().decode('utf-8', errors='ignore')[:500]}") from e
    except urllib.error.URLError as e:
        raise RuntimeError(f"Could not reach DeepSeek: {e.reason}") from e
    choice = (data.get("choices") or [{}])[0]
    content = (choice.get("message", {}).get("content") or "").strip()
    if not content:
        raise RuntimeError(f"DeepSeek returned no text (finish_reason={choice.get('finish_reason')!r}).")
    return _strip_fences(content).strip()


def cached_ask(model, instruction, text, temperature):
    """ask_deepseek, answered from the disk cache when the same request was made before."""
    cache = _cache_file(model, instruction, text, temperature)
    if os.path.exists(cache):
        with open(cache, encoding="utf-8") as f:
            return f.read()
    reply = ask_deepseek(model, instruction, text, temperature)
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    with open(cache, "w", encoding="utf-8") as f:
        f.write(reply)
    return reply


def has_section_tags(text):
    return bool(SECTION_TAG.search(text))


class DeepSeekText:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "instruction": ("STRING", {"multiline": True, "default": DEFAULT_INSTRUCTION,
                                           "tooltip": "Sent before the text, in the same message."}),
                "model": (MODELS, {"default": MODELS[0]}),
                "temperature": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05}),
                "skip_if_tagged": ("BOOLEAN", {"default": True,
                                               "tooltip": "Pass the text through unchanged when it already has "
                                                          "section tags like [Verse] or [Chorus] on their own line."}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "run"
    CATEGORY = "AnotherUtils/inference"

    def run(self, text, instruction, model, temperature, skip_if_tagged):
        if not text.strip() or (skip_if_tagged and has_section_tags(text)):
            return (text,)
        return (cached_ask(model, instruction, text, temperature),)
