import torch
import logging

logger = logging.getLogger(__name__)


class AnotherLTXSequencer:
    """
    Automated version of LTXSequencer (Guide mode).
    Takes 'multi_input' (batched images) and 'indices' (list of frame positions).

    IMPORTANT: This node uses LTXVAddGuide.append_keyframe which EXTENDS the video
    latent along the time dimension. For LTX 2.3 AV workflows, this node should be
    placed BEFORE LTXVConcatAVLatent (i.e., it should receive a plain video latent,
    NOT a NestedTensor). The audio latent is handled separately by ConcatAVLatent.
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
        from comfy_extras.nodes_lt import LTXVAddGuide, get_noise_mask, _append_guide_attention_entry, get_keyframe_idxs

        # Unwrap list inputs because INPUT_IS_LIST = True
        positive = positive[0] if isinstance(positive, list) else positive
        negative = negative[0] if isinstance(negative, list) else negative
        vae = vae[0] if isinstance(vae, list) else vae
        latent = latent[0] if isinstance(latent, list) else latent
        multi_input = multi_input[0] if isinstance(multi_input, list) else multi_input
        num_images = num_images[0] if isinstance(num_images, list) else num_images
        insert_mode = insert_mode[0] if isinstance(insert_mode, list) else insert_mode
        frame_rate = frame_rate[0] if isinstance(frame_rate, list) else frame_rate
        strength = strength[0] if isinstance(strength, list) else strength

        # Indices list
        idx_list = indices if isinstance(indices, list) else [indices]
        if len(idx_list) > 0 and isinstance(idx_list[0], list):
            idx_list = idx_list[0]

        scale_factors = vae.downscale_index_formula
        
        # Pure Video approach
        latent_samples = latent["samples"]
        noise_mask = get_noise_mask(latent).clone()
        
        if len(latent_samples.shape) != 5:
            # Check for NestedTensor (Audio+Video)
            try:
                from comfy.nested_tensor import NestedTensor
                if isinstance(latent_samples, NestedTensor):
                    raise ValueError("[AnotherLTXSequencer] Este nó agora opera em modo modular (Apenas Vídeo). Use o nó 'Separate AV Latent' antes de conectar aqui, como no workflow oficial.")
            except ImportError:
                pass

        _, _, latent_length, latent_height, latent_width = latent_samples.shape
        batch_size = multi_input.shape[0] if multi_input is not None else 0

        cur_pos = positive
        cur_neg = negative

        # Process guide images
        for i in range(1, num_images + 1):
            if i > batch_size:
                continue

            img = multi_input[i-1:i]
            if img is None:
                continue

            if (i-1) >= len(idx_list):
                continue

            # Calculate frame index
            f_idx = idx_list[i-1]
            if insert_mode == "seconds":
                f_idx = int(f_idx * frame_rate)

            # Encode and get latent index
            image_1, t = LTXVAddGuide.encode(vae, latent_width, latent_height, img, scale_factors)
            frame_idx, latent_idx = LTXVAddGuide.get_latent_index(cur_pos, latent_length, len(image_1), f_idx, scale_factors)

            delta_t = t.shape[2]

            # In modular mode, we just append guides as instructed.
            # If the user wants to CROP old guides, they use the modular LTXVCropGuides node.
            cur_pos, cur_neg, latent_samples, noise_mask = LTXVAddGuide.append_keyframe(
                cur_pos, cur_neg,
                frame_idx,
                latent_samples,
                noise_mask,
                t,
                strength,
                scale_factors,
            )

            # Add attention entry for the guide
            try:
                pre_filter_count = t.shape[2] * t.shape[3] * t.shape[4]
                guide_latent_shape = list(t.shape[2:])
                cur_pos, cur_neg = _append_guide_attention_entry(
                    cur_pos, cur_neg, pre_filter_count, guide_latent_shape, strength=strength
                )
            except Exception as e:
                logger.error(f"[AnotherLTXSequencer] Failed to append attention entry: {e}")

        # Build output latent
        new_latent = latent.copy()
        new_latent["samples"] = latent_samples
        new_latent["noise_mask"] = noise_mask

        return (cur_pos, cur_neg, new_latent)
