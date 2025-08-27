# mean_curvature_blur_gegl_like.py
"""
ComfyUI Custom Node — Mean Curvature Blur (GEGL-like)
Approximate GEGL's `gegl:mean-curvature-blur` using mean curvature flow.

Inputs:
  - image (IMAGE)
  - iterations (INT, default 20, range 0..500)
  - linear (BOOLEAN, default True) — process in linear RGB for better radiometric behavior

Alpha channel is preserved unchanged.

Install: save to ComfyUI/custom_nodes/mean_curvature_blur_gegl_like.py and restart ComfyUI.
"""

from typing import Tuple
import torch

# --------- sRGB <-> Linear (torch) ---------

def srgb_to_linear_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    a = 0.055
    low = x / 12.92
    high = ((x + a) / (1.0 + a)).pow(2.4)
    return torch.where(x <= 0.04045, low, high)


def linear_to_srgb_torch(x: torch.Tensor) -> torch.Tensor:
    x = x.clamp(0.0, 1.0)
    a = 0.055
    low = 12.92 * x
    high = (1.0 + a) * x.clamp(min=0.0).pow(1.0 / 2.4) - a
    return torch.where(x <= 0.0031308, low, high)


# --------- Mean Curvature Flow step ---------

def _central_diff_xy(img_nchw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central differences with reflect padding along H (dim=2) and W (dim=3).
    img_nchw: [B,C,H,W]
    Returns (dx, dy) each [B,C,H,W]
    """
    pad = torch.nn.functional.pad(img_nchw, (1,1,1,1), mode="reflect")
    # central difference: (f(x+1) - f(x-1)) / 2
    dx = (pad[:, :, 1:-1, 2:] - pad[:, :, 1:-1, :-2]) * 0.5
    dy = (pad[:, :, 2:, 1:-1] - pad[:, :, :-2, 1:-1]) * 0.5
    return dx, dy


def _divergence(nx: torch.Tensor, ny: torch.Tensor) -> torch.Tensor:
    """Compute divergence d(nx)/dx + d(ny)/dy using central differences with reflect pad.
    nx, ny: [B,C,H,W]
    Returns [B,C,H,W]
    """
    padx = torch.nn.functional.pad(nx, (1,1,1,1), mode="reflect")
    pdy = torch.nn.functional.pad(ny, (1,1,1,1), mode="reflect")
    dnx_dx = (padx[:, :, 1:-1, 2:] - padx[:, :, 1:-1, :-2]) * 0.5
    dny_dy = (pdy[:, :, 2:, 1:-1] - pdy[:, :, :-2, 1:-1]) * 0.5
    return dnx_dx + dny_dy


class MeanCurvatureBlurGEGLLike:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "iterations": ("INT", {"default": 20, "min": 0, "max": 500}),
                "linear": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "image/blur"

    def _prep(self, image: torch.Tensor):
        assert image.dim() == 4 and image.shape[-1] in (3,4), "Expected [B,H,W,3|4]"
        if image.shape[-1] == 3:
            return image.to(torch.float32), None, False
        return image[..., :3].to(torch.float32), image[..., 3:4].to(torch.float32), True

    def apply(self, image: torch.Tensor, iterations: int, linear: bool):
        device = image.device
        rgb, a, has_alpha = self._prep(image)

        # move to NCHW for finite differences
        x = (srgb_to_linear_torch(rgb) if linear else rgb).permute(0,3,1,2).contiguous()

        # Explicit Euler with small stable step
        # Empirically, dt=0.2 is stable for central-diff scheme used here
        dt = 0.2
        eps = 1e-6

        for _ in range(int(iterations)):
            ux, uy = _central_diff_xy(x)
            grad_norm = torch.sqrt(ux*ux + uy*uy + eps)
            nx = ux / grad_norm
            ny = uy / grad_norm
            curv = _divergence(nx, ny)
            x = x + dt * grad_norm * curv
            x = x.clamp(0.0, 1.0)

        # back to BHWC and sRGB
        x = x.permute(0,2,3,1).contiguous()
        out_rgb = linear_to_srgb_torch(x) if linear else x
        out_rgb = out_rgb.clamp(0.0, 1.0)

        if has_alpha:
            out = torch.cat([out_rgb, a], dim=-1)
        else:
            out = out_rgb
        return (out,)


NODE_CLASS_MAPPINGS = {
    "MeanCurvatureBlurGEGLLike": MeanCurvatureBlurGEGLLike,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MeanCurvatureBlurGEGLLike": "Mean Curvature Blur (GEGL-like)",
}
