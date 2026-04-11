"""
LTXVMultiGuide - Dynamic N-frame guide injection for LTX Video
Uses the core LTXVAddGuide API (Technique B: append + RoPE repositioning)
"""

import torch
from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask, _append_guide_attention_entry


class LTXVMultiGuide:
    """
    Injects N reference images as guides into an LTX Video latent.
    Each image is positioned at a specific point in the video timeline.
    Wraps the core LTXVAddGuide internal methods in a loop for dynamic N.
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

    def _resolve_frame_indices(self, indices, positions_str, mode, fps, total_pixel_frames, num_images):
        """Resolve frame indices from either INT list or positions string."""
        # Priority 1: indices input (from ImageListSampler etc.)
        if indices is not None and len(indices) > 0:
            if mode == "frames":
                return [int(i) for i in indices]
            elif mode == "seconds":
                return [
                    -1 if i < 0 else min(round(float(i) / fps * fps), total_pixel_frames - 1)
                    if False else int(i)  # indices are already frame numbers
                    for i in indices
                ]
            elif mode == "percentage":
                return [
                    -1 if i >= total_pixel_frames - 1
                    else int(i)
                    for i in indices
                ]
            # Default: treat indices as frame numbers directly
            return [int(i) for i in indices]

        # Priority 2: positions string
        if positions_str and positions_str.strip():
            return self._parse_positions(positions_str, mode, fps, total_pixel_frames)

        # Fallback: distribute evenly
        if num_images == 1:
            return [0]
        step = (total_pixel_frames - 1) / (num_images - 1)
        return [round(i * step) for i in range(num_images)]

    def _parse_positions(self, positions_str, mode, fps, total_pixel_frames):
        """Parse position string into frame indices."""
        raw = [s.strip() for s in positions_str.split(",") if s.strip()]
        frame_indices = []

        for val_str in raw:
            val = float(val_str)

            if mode == "frames":
                frame_indices.append(int(val))
            elif mode == "seconds":
                if val < 0:
                    frame_indices.append(-1)
                else:
                    frame_idx = round(val * fps)
                    frame_idx = min(frame_idx, total_pixel_frames - 1)
                    frame_indices.append(frame_idx)
            elif mode == "percentage":
                if val < 0 or val >= 100.0:
                    frame_indices.append(-1)
                else:
                    frame_idx = round(val / 100.0 * (total_pixel_frames - 1))
                    frame_indices.append(frame_idx)

        return frame_indices

    def _parse_strengths(self, strengths_str, num_images, default_strength):
        """Parse per-image strengths or fill with default."""
        if not strengths_str or not strengths_str.strip():
            return [default_strength] * num_images

        raw = [s.strip() for s in strengths_str.split(",") if s.strip()]
        parsed = [max(0.0, min(1.0, float(s))) for s in raw]

        while len(parsed) < num_images:
            parsed.append(default_strength)

        return parsed[:num_images]

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

        # Flatten images into individual frames
        if isinstance(images, list):
            frames = []
            for item in images:
                if isinstance(item, torch.Tensor):
                    if len(item.shape) == 4:
                        for j in range(item.shape[0]):
                            frames.append(item[j:j+1])
                    else:
                        frames.append(item.unsqueeze(0) if len(item.shape) == 3 else item)
            images_tensor = torch.cat(frames, dim=0) if frames else images[0]
        else:
            images_tensor = images

        num_images = images_tensor.shape[0]

        # Unwrap indices list
        if indices is not None and isinstance(indices, list):
            idx_list = indices  # already a list of ints
        else:
            idx_list = None

        scale_factors = vae.downscale_index_formula
        time_scale_factor = scale_factors[0]
        latent_length = latent["samples"].shape[2]
        total_pixel_frames = (latent_length - 1) * time_scale_factor + 1

        frame_indices = self._resolve_frame_indices(
            idx_list, positions_str, mode, fps, total_pixel_frames, num_images
        )
        strength_list = self._parse_strengths(strengths_str, num_images, strength)

        n = min(num_images, len(frame_indices))
        if num_images != len(frame_indices):
            print(f"[LTXVMultiGuide] Warning: {num_images} images vs {len(frame_indices)} positions. Using first {n}.")

        latent_image = latent["samples"]
        noise_mask = get_noise_mask(latent)
        _, _, lat_len, lat_h, lat_w = latent_image.shape

        cur_pos = positive
        cur_neg = negative

        for i in range(n):
            single_image = images_tensor[i:i+1]
            frame_idx = frame_indices[i]
            img_strength = strength_list[i]

            _, t = LTXVAddGuide.encode(vae, lat_w, lat_h, single_image, scale_factors)

            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                cur_pos, latent_image.shape[2], len(single_image), frame_idx, scale_factors
            )

            cur_pos, cur_neg, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                cur_pos, cur_neg,
                frame_idx,
                latent_image, noise_mask,
                t, img_strength,
                scale_factors,
            )

            pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
            guide_latent_shape = list(t.shape[2:])
            cur_pos, cur_neg = _append_guide_attention_entry(
                cur_pos, cur_neg, pre_filter_count, guide_latent_shape, strength=img_strength,
            )

        return (cur_pos, cur_neg, {"samples": latent_image, "noise_mask": noise_mask})
