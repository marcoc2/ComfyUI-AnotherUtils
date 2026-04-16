import torch
import logging

logger = logging.getLogger(__name__)

class AnotherLTXSequencer:
    """
    Automated version of LTXSequencer (Guide mode).
    Pure mirror of the WhatDreamsCost LTXSequencer behavior:
    - Uses ONLY append_keyframe (no manual attention entries)
    - Returns a CLEAN latent dict (no metadata carryover)
    - Automatically detects number of images from the index list
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
                "latent": ("LATENT",),
                "multi_input": ("IMAGE",),
                "indices": ("INT",),
                "num_images": ("INT", {"default": 1, "min": 0, "max": 50, "step": 1, "tooltip": "Number of images to process from the batch."}),
                "insert_mode": (["frames", "seconds"], {"default": "frames", "tooltip": "Select the method for determining insertion points."}),
                "frame_rate": ("INT", {"default": 24, "min": 1, "max": 120, "step": 1, "tooltip": "Video FPS (used for calculating second insertions)."}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Global strength for all guide images."}),
            }
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT")
    RETURN_NAMES = ("positive", "negative", "latent")
    FUNCTION = "execute"
    CATEGORY = "AnotherUtils/video"

    def execute(self, positive, negative, vae, latent, multi_input, indices, num_images, insert_mode, frame_rate, strength):
        from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask

        # Unwrap list inputs because INPUT_IS_LIST = True
        pos = positive[0] if isinstance(positive, list) else positive
        neg = negative[0] if isinstance(negative, list) else negative
        v = vae[0] if isinstance(vae, list) else vae
        lat = latent[0] if isinstance(latent, list) else latent
        m_input = multi_input[0] if isinstance(multi_input, list) else multi_input
        n_imgs = num_images[0] if isinstance(num_images, list) else num_images
        i_mode = insert_mode[0] if isinstance(insert_mode, list) else insert_mode
        f_rate = frame_rate[0] if isinstance(frame_rate, list) else frame_rate
        str_val = strength[0] if isinstance(strength, list) else strength

        # Indices list automation
        idx_list = indices if isinstance(indices, list) else [indices]
        if len(idx_list) > 0 and isinstance(idx_list[0], list):
            idx_list = idx_list[0]

        scale_factors = v.downscale_index_formula

        # Clone to avoid mutating upstream latent
        latent_image = lat["samples"].clone()

        # Fetch or generate noise mask
        if "noise_mask" in lat:
            noise_mask = lat["noise_mask"].clone()
        else:
            batch, _, latent_frames, latent_height, latent_width = latent_image.shape
            noise_mask = torch.ones(
                (batch, 1, latent_frames, 1, 1),
                dtype=torch.float32,
                device=latent_image.device,
            )

        _, _, latent_length, latent_height, latent_width = latent_image.shape
        batch_size = m_input.shape[0] if m_input is not None else 0

        # Automation: Use indices count if provided
        effective_num_images = n_imgs
        if len(idx_list) > 1:
            effective_num_images = min(len(idx_list), batch_size)

        # Initialize current conditioning
        cur_pos = pos
        cur_neg = neg

        # Process guide images — pure append_keyframe loop, no attention entries
        for i in range(1, effective_num_images + 1):
            if i > batch_size:
                continue

            img = m_input[i-1:i]
            if img is None:
                continue

            if (i-1) >= len(idx_list):
                continue

            # Calculate frame index
            f_idx = idx_list[i-1]
            if i_mode == "seconds":
                f_idx = int(f_idx * f_rate)

            # Encode image
            image_1, t = LTXVAddGuide.encode(v, latent_width, latent_height, img, scale_factors)

            # Get latent index using current conditioning
            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(
                cur_pos, latent_length, len(image_1), f_idx, scale_factors
            )

            # append_keyframe only
            cur_pos, cur_neg, latent_image, noise_mask = LTXVAddGuide.append_keyframe(
                cur_pos,
                cur_neg,
                frame_idx,
                latent_image,
                noise_mask,
                t,
                str_val,
                scale_factors,
            )

        # Return CLEAN dict
        return (cur_pos, cur_neg, {"samples": latent_image, "noise_mask": noise_mask})
