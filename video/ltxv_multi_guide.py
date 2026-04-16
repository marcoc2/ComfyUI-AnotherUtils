"""
LTXVMultiGuide - Dynamic N-frame guide injection for LTX Video
Pure mirror of LTXSequencer (WhatDreamsCost) behavior:
- Uses ONLY append_keyframe (no manual attention entries)
- Returns a CLEAN latent dict (no metadata carryover)
"""

import torch
from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask

from .ltxv_utils import resolve_frame_indices, parse_strengths, flatten_images


class LTXVMultiGuide:
    """
    Injects N reference images as guides into an LTX Video latent.
    Each image is positioned at a specific point in the video timeline.
    Wraps the core LTXVAddGuide.append_keyframe in a loop for dynamic N.
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
                    "tooltip": "Default strength for all guides. Overridden by per-image strengths if provided.",
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
        # Unwrap list inputs (INPUT_IS_LIST=True sends everything as lists)
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

        # Unwrap indices list
        idx_list = indices if indices is not None and isinstance(indices, list) else None

        scale_factors = vae.downscale_index_formula
        time_scale_factor = scale_factors[0]

        # --- Mirror of LTXSequencer reference ---
        # Clone to avoid mutating upstream latent
        latent_image = latent["samples"].clone()

        # Fetch or generate noise mask
        if "noise_mask" in latent:
            noise_mask = latent["noise_mask"].clone()
        else:
            batch, _, latent_frames, latent_height, latent_width = latent_image.shape
            noise_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=latent_image.device,
            )

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        total_pixel_frames = (latent_length - 1) * time_scale_factor + 1

        frame_indices = resolve_frame_indices(
            idx_list, positions_str, mode, fps, total_pixel_frames, num_images
        )
        strength_list = parse_strengths(strengths_str, num_images, strength)

        n = min(num_images, len(frame_indices))
        if num_images != len(frame_indices):
            print(f"[LTXVMultiGuide] Warning: {num_images} images vs {len(frame_indices)} positions. Using first {n}.")

        # Process guide images — pure append_keyframe loop, NO attention entries
        for i in range(n):
            single_image = images_tensor[i:i+1]
            frame_idx = frame_indices[i]
            img_strength = strength_list[i]

            _, t = LTXVAddGuide.encode(vae, latent_width, latent_height, single_image, scale_factors)

            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                positive, latent_length, len(single_image), frame_idx, scale_factors
            )

            # append_keyframe only — NO _append_guide_attention_entry
            positive, negative, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                positive, negative,
                frame_idx,
                latent_image, noise_mask,
                t, img_strength,
                scale_factors,
            )

        # Return CLEAN dict — no metadata carryover
        return (positive, negative, {"samples": latent_image, "noise_mask": noise_mask})
