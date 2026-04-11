"""
LTXVVid2Vid - Encode a video into LTX Video latent space for vid2vid workflows.
The output latent replaces EmptyLTXVLatentVideo in the pipeline.
Denoise level is controlled by the scheduler (e.g. BasicScheduler denoise=0.3).
"""

import torch
import comfy.utils
import comfy.model_management


class LTXVVid2Vid:
    """
    Encodes a source video into the LTX Video latent space.
    Use as a drop-in replacement for EmptyLTXVLatentVideo in vid2vid workflows.
    Frames are cropped to 8n+1 and resized to the target dimensions.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_frames": ("IMAGE",),
                "vae": ("VAE",),
                "width": ("INT", {
                    "default": 768, "min": 64, "max": 4096, "step": 32,
                    "tooltip": "Target width. Must be divisible by 32. Set to 0 to use source width (rounded to nearest 32).",
                }),
                "height": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 32,
                    "tooltip": "Target height. Must be divisible by 32. Set to 0 to use source height (rounded to nearest 32).",
                }),
            },
        }

    RETURN_TYPES = ("LATENT", "INT")
    RETURN_NAMES = ("latent", "frame_count")
    FUNCTION = "execute"
    CATEGORY = "latent/video/ltxv"

    def execute(self, video_frames, vae, width, height):
        num_frames = video_frames.shape[0]
        src_h = video_frames.shape[1]
        src_w = video_frames.shape[2]

        # Auto-detect dimensions from source if 0
        if width == 0:
            width = (src_w // 32) * 32
        if height == 0:
            height = (src_h // 32) * 32

        # Crop frame count to 8n+1
        valid_length = ((num_frames - 1) // 8) * 8 + 1
        if valid_length > num_frames:
            valid_length = max(num_frames - 8, 1)
            valid_length = ((valid_length - 1) // 8) * 8 + 1

        if valid_length != num_frames:
            print(f"[LTXVVid2Vid] Cropping {num_frames} frames to {valid_length} (8n+1 requirement)")

        pixels = video_frames[:valid_length]

        # Resize to target dimensions
        if pixels.shape[1] != height or pixels.shape[2] != width:
            pixels = comfy.utils.common_upscale(
                pixels.movedim(-1, 1), width, height, "bilinear", "center"
            ).movedim(1, -1)

        # Keep only RGB
        encode_pixels = pixels[:, :, :, :3]

        # VAE encode
        t = vae.encode(encode_pixels)

        return ({"samples": t}, valid_length)
