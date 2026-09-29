"""
Qwen2.5-VL Vision-Language Node for ComfyUI

Delegates inference to an isolated subprocess worker (qwen_vl_worker.py).
Killing the worker fully returns RAM/VRAM to the OS, which the in-process
path could never reliably do on Windows.
"""

import os
import sys
import io
import json
import math
import atexit
import base64
import logging
import subprocess

import numpy as np
from PIL import Image

import comfy.model_management as mm

logger = logging.getLogger(__name__)

DEFAULT_HF_CACHE = r"F:\Models\HuggingFace\hub"

AVAILABLE_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
]

_WORKER_SCRIPT = os.path.join(os.path.dirname(__file__), "qwen_vl_worker.py")


# ── Image helpers ────────────────────────────────────────────────────────────

def _resize_for_inference(image: Image.Image, target_pixels: int = 250_000) -> Image.Image:
    w, h = image.size
    current = w * h
    if current <= target_pixels:
        return image
    scale = math.sqrt(target_pixels / current)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _comfy_tensor_to_pil(image_tensor) -> Image.Image:
    img_np = image_tensor[0].cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np, mode="RGB")


def _pil_to_b64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Worker subprocess management ─────────────────────────────────────────────

_worker: subprocess.Popen | None = None
_worker_model: str | None = None


def _worker_alive() -> bool:
    return _worker is not None and _worker.poll() is None


def _kill_worker():
    global _worker, _worker_model
    if _worker is None:
        return

    logger.info("[QwenVLNode] Killing worker subprocess")
    try:
        if _worker.poll() is None and _worker.stdin is not None:
            try:
                _worker.stdin.write(json.dumps({"cmd": "exit"}) + "\n")
                _worker.stdin.flush()
            except Exception:
                pass
            try:
                _worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
        if _worker.poll() is None:
            _worker.kill()
            try:
                _worker.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    except Exception as e:
        logger.warning(f"[QwenVLNode] kill_worker error: {e}")
    finally:
        _worker = None
        _worker_model = None


atexit.register(_kill_worker)


def _spawn_worker(model_name: str, hf_cache: str):
    global _worker, _worker_model

    # Evict ComfyUI-managed models so the subprocess has VRAM headroom.
    # ComfyUI cannot see a foreign subprocess requesting memory, so we have
    # to ask for eviction explicitly here, in the parent process.
    logger.info("[QwenVLNode] Unloading ComfyUI models before spawning worker")
    try:
        mm.unload_all_models()
    except Exception as e:
        logger.warning(f"[QwenVLNode] unload_all_models failed: {e}")
    try:
        mm.soft_empty_cache()
    except Exception:
        pass

    logger.info(f"[QwenVLNode] Spawning worker for {model_name}")
    _worker = subprocess.Popen(
        [sys.executable, "-u", _WORKER_SCRIPT],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,  # inherit parent stderr -> progress bars / logs visible
        text=True,
        bufsize=1,
        encoding="utf-8",
    )

    init = json.dumps({
        "cmd": "init",
        "model_name": model_name,
        "hf_cache": hf_cache,
    })
    try:
        _worker.stdin.write(init + "\n")
        _worker.stdin.flush()
    except Exception as e:
        _kill_worker()
        raise RuntimeError(f"Failed to send init to worker: {e}")

    line = _worker.stdout.readline()
    if not line:
        rc = _worker.poll()
        _kill_worker()
        raise RuntimeError(f"Worker died during init (exit={rc})")

    try:
        msg = json.loads(line.strip())
    except Exception:
        _kill_worker()
        raise RuntimeError(f"Worker sent non-JSON on init: {line!r}")

    if msg.get("error"):
        err = msg["error"]
        _kill_worker()
        raise RuntimeError(f"Worker init error: {err}")
    if msg.get("status") != "ready":
        _kill_worker()
        raise RuntimeError(f"Unexpected worker init reply: {msg}")

    _worker_model = model_name
    logger.info("[QwenVLNode] Worker ready")


def _ensure_worker(model_name: str, hf_cache: str):
    if _worker_alive() and _worker_model == model_name:
        return
    if _worker is not None:
        _kill_worker()
    _spawn_worker(model_name, hf_cache)


def _inference_via_worker(
    pil_image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    max_new_tokens: int,
) -> str:
    if not _worker_alive():
        raise RuntimeError("Worker is not alive")

    req = json.dumps({
        "cmd": "infer",
        "image_b64": _pil_to_b64(pil_image),
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_new_tokens": int(max_new_tokens),
    })
    try:
        _worker.stdin.write(req + "\n")
        _worker.stdin.flush()
    except Exception as e:
        rc = _worker.poll()
        _kill_worker()
        raise RuntimeError(f"Worker pipe broken (exit={rc}): {e}")

    line = _worker.stdout.readline()
    if not line:
        rc = _worker.poll()
        _kill_worker()
        raise RuntimeError(f"Worker died during inference (exit={rc})")

    try:
        msg = json.loads(line.strip())
    except Exception:
        raise RuntimeError(f"Worker sent non-JSON: {line!r}")

    if msg.get("error"):
        raise RuntimeError(f"Worker error: {msg['error']}")
    return msg["response"]


# ── ComfyUI Node ─────────────────────────────────────────────────────────────

class QwenVLNode:
    """
    Runs Qwen2.5-VL multimodal inference inside ComfyUI.

    Inputs:
      - image       : IMAGE (ComfyUI tensor)
      - system_prompt: STRING (editable) — role/context for the LLM
      - user_prompt : STRING (editable) — the actual question/instruction
      - model_name  : choice of available local models
      - hf_cache    : path to your HuggingFace hub cache folder
      - max_new_tokens: INT — how many tokens to generate
      - resize_image: BOOLEAN — resize to ~0.25MP before inference (faster)
      - unload_model: BOOLEAN — kill the worker subprocess after inference

    Output:
      - response    : STRING — the LLM's generated text
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "system_prompt": (
                    "STRING",
                    {
                        "default": (
                            "You are an expert comic book analyst. "
                            "Your job is to extract and describe the content of a single comic panel "
                            "in a concise, structured way that will later be used to generate a video prompt."
                        ),
                        "multiline": True,
                    },
                ),
                "user_prompt": (
                    "STRING",
                    {
                        "default": (
                            "Analyze this comic panel and provide:\n"
                            "1. DIALOG: Transcribe all speech bubble / caption text verbatim (write NONE if absent).\n"
                            "2. ACTION: Describe what is happening visually (characters, expressions, movement).\n"
                            "3. MOOD: One or two words capturing the emotional tone.\n"
                            "Keep the answer short and factual."
                        ),
                        "multiline": True,
                    },
                ),
                "model_name": (AVAILABLE_MODELS, {"default": AVAILABLE_MODELS[0]}),
                "max_new_tokens": (
                    "INT",
                    {"default": 300, "min": 32, "max": 2048, "step": 32},
                ),
            },
            "optional": {
                "hf_cache_override": (
                    "STRING",
                    {
                        "default": DEFAULT_HF_CACHE,
                        "multiline": False,
                    },
                ),
                "resize_image": ("BOOLEAN", {"default": True}),
                "unload_model": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("response",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "run_inference"
    CATEGORY = "AnotherUtils/inference"

    def run_inference(
        self,
        image,
        system_prompt,
        user_prompt,
        model_name,
        max_new_tokens,
        hf_cache_override=DEFAULT_HF_CACHE,
        resize_image=True,
        unload_model=False,
    ):
        hf_cache = hf_cache_override.strip() if hf_cache_override.strip() else DEFAULT_HF_CACHE

        pil_image = _comfy_tensor_to_pil(image)
        if resize_image:
            pil_image = _resize_for_inference(pil_image)

        logger.info(
            f"[QwenVLNode] Running inference | model={model_name} "
            f"| image={pil_image.size} | max_new_tokens={max_new_tokens}"
        )

        _ensure_worker(model_name, hf_cache)
        response = _inference_via_worker(
            pil_image, system_prompt, user_prompt, int(max_new_tokens)
        )

        if unload_model:
            logger.info("[QwenVLNode] unload_model=True -> killing worker")
            _kill_worker()

        logger.info(f"[QwenVLNode] Response ({len(response)} chars): {response[:120]}...")
        return (response,)
