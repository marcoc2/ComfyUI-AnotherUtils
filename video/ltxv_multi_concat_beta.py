import torch
from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask
from typing import Dict, Tuple, Any
import logging
from .ltxv_utils import resolve_frame_indices, parse_strengths, flatten_images

class LTXVMultiConcatBeta:
    """
    Injects N reference images as latents into an LTX Video latent (Inpainting style).
    Includes experimental 'smooth_strength' to interpolate latents and mask strength between keyframes.
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
                "smooth_strength": ("BOOLEAN", {"default": True, "tooltip": "Interpolates latents and decays strength between keyframes to prevent harsh slideshows."}),
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
                    "tooltip": "Inpaint strength at anchor frames. Fixed areas will have noise_mask=0.0.",
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

    def execute(self, positive, negative, vae, latent, images, smooth_strength, mode, fps, strength,
                indices=None, positions=None, strengths=None):
        # Unwrap list inputs
        positive = positive[0] if isinstance(positive, list) else positive
        negative = negative[0] if isinstance(negative, list) else negative
        vae = vae[0] if isinstance(vae, list) else vae
        latent = latent[0] if isinstance(latent, list) else latent
        smooth_strength = smooth_strength[0] if isinstance(smooth_strength, list) else smooth_strength
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

        if not smooth_strength:
            # Original behavior
            for i in range(n):
                single_image = images_tensor[i:i+1]
                frame_idx = frame_indices[i]
                _, t = LTXVAddGuide.encode(vae, lat_w, lat_h, single_image, scale_factors)
                frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                    positive, latent_samples.shape[2], len(single_image), frame_idx, scale_factors
                )
                if latent_idx < latent_samples.shape[2]:
                    latent_samples[:, :, latent_idx:latent_idx+t.shape[2], :, :] = t
                    noise_mask[:, :, latent_idx:latent_idx+t.shape[2], :, :] = 1.0 - strength_list[i]
        else:
            # Smooth Strength / Interpolation behavior
            anchors = {}
            for i in range(n):
                single_image = images_tensor[i:i+1]
                frame_idx = frame_indices[i]
                _, t = LTXVAddGuide.encode(vae, lat_w, lat_h, single_image, scale_factors)
                frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                    positive, latent_length, len(single_image), frame_idx, scale_factors
                )
                if latent_idx < latent_length:
                    anchors[latent_idx] = (t, strength_list[i])
            
            if not anchors:
                if is_nested:
                    from comfy.nested_tensor import NestedTensor
                    early_latent = latent.copy()
                    early_latent["samples"] = NestedTensor((latent_samples, audio_samples))
                    early_latent["noise_mask"] = noise_mask
                else:
                    early_latent = {"samples": latent_samples, "noise_mask": noise_mask}
                return (positive, negative, early_latent)

            sorted_idx = sorted(anchors.keys())

            for i in range(latent_length):
                # Find bounding anchors
                left_idx = None
                right_idx = None
                
                for idx in sorted_idx:
                    if idx <= i:
                        left_idx = idx
                    if idx >= i and right_idx is None:
                        right_idx = idx
                        break
                
                t_interp = None
                s_interp = 0.0

                if left_idx == i or right_idx == i:
                    # Exactly on anchor
                    anchor_idx = left_idx if left_idx == i else right_idx
                    t_interp, s_interp = anchors[anchor_idx]
                
                elif left_idx is not None and right_idx is not None:
                    # Between two anchors
                    dist = right_idx - left_idx
                    alpha = (i - left_idx) / dist
                    t_L, s_L = anchors[left_idx]
                    t_R, s_R = anchors[right_idx]
                    t_interp = t_L * (1.0 - alpha) + t_R * alpha
                    
                    # Triangle strength: peaks at anchors, 0 at midpoint
                    dist_to_nearest = min(i - left_idx, right_idx - i)
                    half_dist = dist / 2.0
                    peak_strength = s_L if (i - left_idx) < (right_idx - i) else s_R
                    s_interp = peak_strength * (1.0 - (dist_to_nearest / half_dist))
                
                elif left_idx is not None and right_idx is None:
                    # After last anchor
                    t_L, s_L = anchors[left_idx]
                    t_interp = t_L
                    dist_from_L = i - left_idx
                    max_dist = max(1, latent_length - 1 - left_idx)
                    s_interp = s_L * max(0.0, 1.0 - (dist_from_L / max_dist))
                
                elif left_idx is None and right_idx is not None:
                    # Before first anchor
                    t_R, s_R = anchors[right_idx]
                    t_interp = t_R
                    dist_from_R = right_idx - i
                    max_dist = max(1, right_idx)
                    s_interp = s_R * max(0.0, 1.0 - (dist_from_R / max_dist))

                # Apply to latent and mask
                s_interp = max(0.0, min(1.0, float(s_interp)))
                if t_interp is not None:
                    latent_samples[:, :, i:i+1, :, :] = t_interp
                    noise_mask[:, :, i:i+1, :, :] = 1.0 - s_interp

        if is_nested:
            from comfy.nested_tensor import NestedTensor
            # For AV latents, preserve audio metadata
            new_latent = latent.copy()
            new_latent["samples"] = NestedTensor((latent_samples, audio_samples))
            noise_mask_raw = latent.get("noise_mask", None)
            if noise_mask_raw is not None and isinstance(noise_mask_raw, NestedTensor):
                audio_noise_mask = noise_mask_raw.tensors[1]
            else:
                audio_noise_mask = torch.ones_like(audio_samples)
            new_latent["noise_mask"] = NestedTensor((noise_mask, audio_noise_mask))
        else:
            # Clean dict — no metadata carryover
            new_latent = {"samples": latent_samples, "noise_mask": noise_mask}

        return (positive, negative, new_latent)
