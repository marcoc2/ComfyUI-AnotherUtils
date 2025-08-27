# rgb_noise_gegl_like.py
"""
ComfyUI Custom Node — RGB Noise (GEGL-like)
Replicates GIMP/GEGL "gegl:noise-rgb" with support for:
- correlated (multiplicative) vs additive noise
- independent RGB channels
- Gaussian or uniform distribution
- Linear RGB processing toggle
- Per-channel amounts (red/green/blue) and optional alpha amount
- Deterministic seeding

Install: save this file to ComfyUI/custom_nodes/rgb_noise_gegl_like.py and restart ComfyUI.
"""

from typing import Tuple
import torch

# --- sRGB <-> Linear helpers (torch, vectorized) ---

def srgb_to_linear_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    a = 0.055
    # piecewise per IEC 61966-2-1
    low = x / 12.92
    high = ((x + a) / (1.0 + a)).pow(2.4)
    return torch.where(x <= 0.04045, low, high)


def linear_to_srgb_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    a = 0.055
    low = 12.92 * x
    high = (1.0 + a) * x.clamp(min=0.0).pow(1.0 / 2.4) - a
    return torch.where(x <= 0.0031308, low, high)


class RGBNoiseGEGLLike:
    """Apply RGB noise similarly to GEGL's gegl:noise-rgb."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Amounts (0..1), per GEGL semantics
                "red": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.001}),
                "green": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.001}),
                "blue": ("FLOAT", {"default": 0.20, "min": 0.0, "max": 1.0, "step": 0.001}),
                # Alpha amount is supported only if the input has an alpha channel
                "alpha": ("FLOAT", {"default": 0.00, "min": 0.0, "max": 1.0, "step": 0.001}),
                # Toggles mirroring GEGL options
                "correlated": ("BOOLEAN", {"default": True}),   # multiplicative when True (GEGL correlated)
                "independent": ("BOOLEAN", {"default": True}),  # per-channel independent noise
                "linear": ("BOOLEAN", {"default": True}),       # process in linear RGB
                "gaussian": ("BOOLEAN", {"default": True}),     # Gaussian vs Uniform
                # Deterministic RNG seed; set any integer. If you want different noise each call, vary the seed.
                "seed": ("INT", {"default": 0, "min": -2**31, "max": 2**31 - 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "image/noise"

    def _make_generator(self, device: torch.device, seed: int) -> torch.Generator:
        gen = torch.Generator(device=device)
        gen.manual_seed(int(seed))
        return gen

    def _prep(self, image: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, bool]:
        """Ensure BHWC float32 in [0,1]; split rgb and optional alpha; returns (rgb, alpha, has_alpha)."""
        # Expect image as [B, H, W, C] float, C=3 or 4
        assert image.dim() == 4, "Expected image tensor with shape [B,H,W,C]"
        b, h, w, c = image.shape
        assert c in (3, 4), "IMAGE must have 3 (RGB) or 4 (RGBA) channels"
        img = image.to(dtype=torch.float32)
        if c == 3:
            return img, None, False
        else:
            return img[..., :3], img[..., 3:4], True

    def apply(self, image: torch.Tensor,
              red: float, green: float, blue: float, alpha: float,
              correlated: bool, independent: bool, linear: bool, gaussian: bool,
              seed: int):
        device = image.device
        dtype = torch.float32
        gen = self._make_generator(device, seed)

        rgb, a, has_alpha = self._prep(image)
        b, h, w, _ = rgb.shape

        # Convert to linear if requested
        work = srgb_to_linear_torch(rgb) if linear else rgb

        # Build noise tensors
        if gaussian:
            if independent:
                n_rgb = torch.randn((b, h, w, 3), generator=gen, device=device, dtype=dtype)
            else:
                base = torch.randn((b, h, w, 1), generator=gen, device=device, dtype=dtype)
                n_rgb = base.expand(-1, -1, -1, 3)
            n_a = torch.randn((b, h, w, 1), generator=gen, device=device, dtype=dtype)
        else:
            if independent:
                n_rgb = torch.rand((b, h, w, 3), generator=gen, device=device, dtype=dtype) * 2.0 - 1.0
            else:
                base = torch.rand((b, h, w, 1), generator=gen, device=device, dtype=dtype) * 2.0 - 1.0
                n_rgb = base.expand(-1, -1, -1, 3)
            n_a = torch.rand((b, h, w, 1), generator=gen, device=device, dtype=dtype) * 2.0 - 1.0

        amt_rgb = torch.tensor([red, green, blue], device=device, dtype=dtype).view(1, 1, 1, 3)
        amt_a = torch.tensor([alpha], device=device, dtype=dtype).view(1, 1, 1, 1)

        if correlated:
            # multiplicative: out = in * (1 + amount * n)
            out_rgb = work * (1.0 + amt_rgb * n_rgb)
            if has_alpha:
                out_a = a * (1.0 + amt_a * n_a)
        else:
            # additive: out = in + 0.5 * amount * n  (0.5 factor matches GEGL behavior)
            out_rgb = work + 0.5 * amt_rgb * n_rgb
            if has_alpha:
                out_a = a + 0.5 * amt_a * n_a

        out_rgb = out_rgb.clamp(0.0, 1.0)
        if has_alpha:
            out_a = out_a.clamp(0.0, 1.0)

        if linear:
            out_rgb = linear_to_srgb_torch(out_rgb)

        if has_alpha:
            out = torch.cat([out_rgb, out_a], dim=3)
        else:
            out = out_rgb

        out = out.clamp(0.0, 1.0)
        return (out,)


NODE_CLASS_MAPPINGS = {
    "RGBNoiseGEGLLike": RGBNoiseGEGLLike,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "RGBNoiseGEGLLike": "RGB Noise (GEGL-like)",
}

