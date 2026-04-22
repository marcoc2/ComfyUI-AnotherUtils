"""
Qwen2.5-VL Vision-Language Node for ComfyUI

Receives an IMAGE and a text prompt, runs Qwen2.5-VL inference,
and returns the generated string. 

Supports any Qwen2.5-VL model available in the local HuggingFace cache.
"""

import os
import gc
import math
import logging
import torch
import numpy as np
from PIL import Image

import comfy.model_management as mm

logger = logging.getLogger(__name__)

# Default HuggingFace cache location (matches your setup)
DEFAULT_HF_CACHE = r"F:\Models\HuggingFace\hub"

# Available models (those already in your local cache)
AVAILABLE_MODELS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "huihui-ai/Qwen2.5-VL-3B-Instruct-abliterated",
]

# ── Image helpers ────────────────────────────────────────────────────────────

def _resize_for_inference(image: Image.Image, target_pixels: int = 250_000) -> Image.Image:
    """Resize image to ~target_pixels while maintaining aspect ratio."""
    w, h = image.size
    current = w * h
    if current <= target_pixels:
        return image
    scale = math.sqrt(target_pixels / current)
    return image.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)


def _comfy_tensor_to_pil(image_tensor) -> Image.Image:
    """Convert ComfyUI IMAGE tensor [B,H,W,C] float32 → PIL RGB (first image of batch)."""
    img_np = image_tensor[0].cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img_np, mode="RGB")


# ── Model cache (singleton per model_name) ───────────────────────────────────

_loaded_model_name: str | None = None
_model = None
_processor = None


def _load_model(model_name: str, hf_cache: str):
    """Lazy-load Qwen2.5-VL model + processor. Reuses cache between calls."""
    global _loaded_model_name, _model, _processor

    if _model is not None and _loaded_model_name == model_name:
        return _model, _processor

    # Unload previous model if switching
    if _model is not None:
        logger.info(f"[QwenVLNode] Unloading previous model: {_loaded_model_name}")
        try:
            _model.to("cpu")
        except:
            pass
        del _model, _processor
        _model = None
        _processor = None
        _loaded_model_name = None
        
        gc.collect()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    try:
        from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    except ImportError as e:
        raise ImportError(
            "transformers package required. Run: pip install transformers qwen-vl-utils\n"
            f"Original error: {e}"
        )

    # Point HuggingFace to your local cache
    os.environ["HF_HOME"] = hf_cache
    os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(hf_cache, "hub") if not hf_cache.endswith("hub") else hf_cache

    logger.info(f"[QwenVLNode] Loading model: {model_name}")
    logger.info(f"[QwenVLNode] HF cache: {hf_cache}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    _processor = AutoProcessor.from_pretrained(model_name, cache_dir=hf_cache)

    # Force ComfyUI to unload other models to free up VRAM
    mm.soft_empty_cache()
    
    # Try flash attention first, fall back to standard
    device_map = "auto"
    
    try:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            attn_implementation="flash_attention_2",
            cache_dir=hf_cache,
        )
        logger.info("[QwenVLNode] Loaded with Flash Attention 2")
    except Exception:
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto",
            cache_dir=hf_cache,
        )
        logger.info("[QwenVLNode] Loaded with standard attention")

    _model.eval()
    _loaded_model_name = model_name
    logger.info(f"[QwenVLNode] Model ready: {model_name}")

    return _model, _processor


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
        # Determine cache path
        hf_cache = hf_cache_override.strip() if hf_cache_override.strip() else DEFAULT_HF_CACHE

        # Convert ComfyUI tensor → PIL
        pil_image = _comfy_tensor_to_pil(image)
        if resize_image:
            pil_image = _resize_for_inference(pil_image)

        logger.info(
            f"[QwenVLNode] Running inference | model={model_name} "
            f"| image={pil_image.size} | max_new_tokens={max_new_tokens}"
        )
        
        # Free ComfyUI VRAM BEFORE loading Qwen
        mm.soft_empty_cache()

        model, processor = _load_model(model_name, hf_cache)

        # Build message in Qwen2.5-VL format
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": user_prompt},
                ],
            },
        ]

        # Tokenize
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=[pil_image],
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
            )

        # Decode only the newly generated tokens
        generated_ids = [
            out[len(inp):]
            for inp, out in zip(inputs.input_ids, output_ids)
        ]
        response = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0].strip()

        # Cleanup
        del inputs, output_ids, generated_ids
        
        # Unload model entirely if requested
        if unload_model:
            global _model, _processor, _loaded_model_name
            logger.info(f"[QwenVLNode] Unloading model {model_name} to free VRAM...")
            
            try:
                _model.to("cpu")
            except:
                pass
                
            del _model, _processor
            _model = None
            _processor = None
            _loaded_model_name = None
            
        # Always run garbage collection BEFORE emptying CUDA cache
        gc.collect()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        logger.info(f"[QwenVLNode] Response ({len(response)} chars): {response[:120]}...")
        return (response,)
