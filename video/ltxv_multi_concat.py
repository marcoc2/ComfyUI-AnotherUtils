import torch
from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask
from typing import Dict, Tuple, Any
import logging
from .ltxv_utils import resolve_frame_indices, parse_strengths, flatten_images

class LTXVMultiConcat:
    """
    Injects N reference images as latents into an LTX Video latent (Inpainting style).
    This method is faster than Guiding because it doesn't add tokens to the attention context.
    The model 'sees' the frames via the latent channels and noise mask.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "images": ("IMAGE",),
                "mode": (["frames", "seconds", "percentage"],),
                "fps": ("FLOAT", {
                    "default": 24.0,
                    "min": 1.0,
                    "max": 120.0,
                    "step": 0.1,
                    "tooltip": "Video FPS. Used to convert seconds/percentage to frame indices.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Inpaint strength (conceptually). Fixed areas will have noise_mask=0.0.",
                }),
            },
            "optional": {
                "indices": ("INT", {
                    "tooltip": "Frame indices from ImageListSampler or similar. Overrides positions string.",
                }),
                "positions": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Comma-separated positions for each image. "
                        "Only used if indices is not connected. "
                        "Use -1 for last frame (frames mode only)."
                    ),
                }),
                "strengths": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional comma-separated per-image strengths (0.0-1.0). "
                        "If empty, uses the default strength for all."
                    ),
                }),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "conditioning/video_models"

    def execute(self, positive, negative, vae, latent, images, mode, fps, strength,
                indices=None, positions=None, strengths=None):
        # Unwrap list inputs
        positive = positive[0] if isinstance(positive, list) else positive
        negative = negative[0] if isinstance(negative, list) else negative
        vae = vae[0] if isinstance(vae, list) else vae
        latent = latent[0] if isinstance(latent, list) else latent
        mode = mode[0] if isinstance(mode, list) else mode
        fps = fps[0] if isinstance(fps, list) else fps
        strength = strength[0] if isinstance(strength, list) else strength
        positions_str = positions[0] if isinstance(positions, list) else (positions or "")
        strengths_str = strengths[0] if isinstance(strengths, list) else (strengths or "")

        # Flatten images
        images_tensor = flatten_images(images)
        num_images = images_tensor.shape[0]

        idx_list = indices if indices is not None and isinstance(indices, list) else None

        scale_factors = vae.downscale_index_formula
        time_scale_factor = scale_factors[0]

        # Handle NestedTensor from audio-enabled latents
        try:
            from comfy.nested_tensor import NestedTensor
            is_nested = isinstance(latent["samples"], NestedTensor)
        except ImportError:
            is_nested = False

        if is_nested:
            latent_samples = latent["samples"].tensors[0].clone()
            audio_samples = latent["samples"].tensors[1]
        else:
            latent_samples = latent["samples"].clone()
            
        latent_length = latent_samples.shape[2]
        total_pixel_frames = (latent_length - 1) * time_scale_factor + 1

        frame_indices = resolve_frame_indices(
            idx_list, positions_str, mode, fps, total_pixel_frames, num_images
        )
        strength_list = parse_strengths(strengths_str, num_images, strength)

        n = min(num_images, len(frame_indices))
        
        has_nested_mask = False
        noise_mask_obj = get_noise_mask(latent)
        if is_nested:
            from comfy.nested_tensor import NestedTensor
            if isinstance(noise_mask_obj, NestedTensor):
                noise_mask = noise_mask_obj.tensors[0].clone()
            else:
                noise_mask = noise_mask_obj.clone()
        else:
            noise_mask = noise_mask_obj.clone()
            
        _, _, lat_len, lat_h, lat_w = latent_samples.shape

        for i in range(n):
            single_image = images_tensor[i:i+1]
            frame_idx = frame_indices[i]
            # img_strength = strength_list[i] # In concat mode, we use strength to determine mask?
            
            # Encode image to latent at video resolution
            _, t = LTXVAddGuide.encode(vae, lat_w, lat_h, single_image, scale_factors)
            # Find the correct latent index
            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                positive, latent_samples.shape[2], len(single_image), frame_idx, scale_factors
            )

            # Insert encoded frames into the main latent
            # LTXV latents are [B, C, T, H, W]
            # t is usually [1, C, 1, H, W] for a single frame
            if latent_idx < latent_samples.shape[2]:
                latent_samples[:, :, latent_idx:latent_idx+t.shape[2], :, :] = t
                # Set mask to 1.0 - strength (0.0 = fully known/fixed, 1.0 = fully noise/rendered)
                # noise_mask is [B, 1, T, 1, 1] typically
                noise_mask[:, :, latent_idx:latent_idx+t.shape[2], :, :] = 1.0 - strength

        new_latent = latent.copy()

        if is_nested:
            from comfy.nested_tensor import NestedTensor
            new_latent["samples"] = NestedTensor((latent_samples, audio_samples))
            # Re-wrap noise_mask as NestedTensor to match samples structure.
            # The audio mask stays unchanged (all ones = fully denoised).
            noise_mask_raw = latent.get("noise_mask", None)
            if noise_mask_raw is not None and isinstance(noise_mask_raw, NestedTensor):
                audio_noise_mask = noise_mask_raw.tensors[1]
            else:
                audio_noise_mask = torch.ones_like(audio_samples)
            new_latent["noise_mask"] = NestedTensor((noise_mask, audio_noise_mask))
        else:
            new_latent["samples"] = latent_samples
            new_latent["noise_mask"] = noise_mask

        # In Concat mode, we DON'T modify conditioning with attention entries.
        # We just return the modified latent with frames and mask.
        return (positive, negative, new_latent)
