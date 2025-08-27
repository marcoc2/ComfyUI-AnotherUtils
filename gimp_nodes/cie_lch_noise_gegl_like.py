# cie_lch_noise_gegl_like.py
"""
ComfyUI Custom Node — CIE LCh Noise (GEGL-like)
Replicates GIMP/GEGL "gegl:noise-cie-lch" behavior.

Inputs:
  - image (IMAGE)
  - lightness_distance (FLOAT 0..100, default 40.0)
  - chroma_distance    (FLOAT 0..100, default 40.0)
  - hue_distance       (FLOAT 0..180, default 3.0)
  - holdness           (INT 1..8,    default 2)  # "Dulling" no GEGL
  - seed               (INT, default 0)

Alpha é preservado.
Instalação: salve como ComfyUI/custom_nodes/cie_lch_noise_gegl_like.py e reinicie o ComfyUI.
"""

import math
import torch

# ---------- sRGB <-> Linear (torch) ----------

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

# ---------- Linear RGB <-> XYZ (D65) ----------

def rgb_to_xyz_d65_torch(rgb_lin: torch.Tensor) -> torch.Tensor:
    """Convert linear sRGB to XYZ (D65). Input [B,H,W,3]."""
    M = torch.tensor([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ], device=rgb_lin.device, dtype=rgb_lin.dtype)
    x = rgb_lin.reshape(-1, 3) @ M.T
    return x.reshape_as(rgb_lin)

def xyz_to_rgb_d65_torch(xyz: torch.Tensor) -> torch.Tensor:
    """Convert XYZ (D65) to linear sRGB. Input [B,H,W,3]."""
    M_inv = torch.tensor([
        [ 3.2404542, -1.5371385, -0.4985314],
        [-0.9692660,  1.8760108,  0.0415560],
        [ 0.0556434, -0.2040259,  1.0572252],
    ], device=xyz.device, dtype=xyz.dtype)
    x = xyz.reshape(-1, 3) @ M_inv.T
    return x.reshape_as(xyz)

# ---------- XYZ (D65) <-> Lab ----------

def _f_lab_torch(t: torch.Tensor) -> torch.Tensor:
    delta = 6/29
    return torch.where(t > delta**3, t.pow(1/3), t/(3*delta**2) + 4/29)

def _finv_lab_torch(ft: torch.Tensor) -> torch.Tensor:
    delta = 6/29
    return torch.where(ft > delta, ft**3, 3*delta**2*(ft - 4/29))

def xyz_to_lab_d65_torch(xyz: torch.Tensor) -> torch.Tensor:
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = xyz[...,0] / Xn
    y = xyz[...,1] / Yn
    z = xyz[...,2] / Zn
    fx = _f_lab_torch(x)
    fy = _f_lab_torch(y)
    fz = _f_lab_torch(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.stack([L, a, b], dim=-1)

def lab_to_xyz_d65_torch(lab: torch.Tensor) -> torch.Tensor:
    L, a, b = lab[...,0], lab[...,1], lab[...,2]
    fy = (L + 16) / 116
    fx = fy + (a / 500)
    fz = fy - (b / 200)
    Xn, Yn, Zn = 0.95047, 1.00000, 1.08883
    x = _finv_lab_torch(fx) * Xn
    y = _finv_lab_torch(fy) * Yn
    z = _finv_lab_torch(fz) * Zn
    return torch.stack([x, y, z], dim=-1)

# ---------- Lab <-> LCh ----------

def lab_to_lch_torch(lab: torch.Tensor) -> torch.Tensor:
    L = lab[...,0]
    a = lab[...,1]
    b = lab[...,2]
    C = torch.sqrt(a*a + b*b)
    h = torch.atan2(b, a)  # radians
    h_deg = (h * (180.0/math.pi)) % 360.0
    return torch.stack([L, C, h_deg], dim=-1)

def lch_to_lab_torch(lch: torch.Tensor) -> torch.Tensor:
    L = lch[...,0]
    C = lch[...,1]
    h = lch[...,2] * (math.pi/180.0)
    a = C * torch.cos(h)
    b = C * torch.sin(h)
    return torch.stack([L, a, b], dim=-1)

# ---------- GEGL-like randomization helpers ----------

def _min_of_k_uniform(shape, k: int, generator: torch.Generator, device, dtype):
    v = torch.rand(shape, generator=generator, device=device, dtype=dtype)
    for _ in range(max(0, k-1)):
        v = torch.minimum(v, torch.rand(shape, generator=generator, device=device, dtype=dtype))
    return v

def _randomize_value(now: torch.Tensor, minv: float, maxv: float, wraps: bool,
                     rand_max: float, holdness: int, gen: torch.Generator) -> torch.Tensor:
    """Vectorized port of GEGL randomize_value():
    - min-of-k uniforms (k = holdness)
    - random +/- direction
    - wrapping for circular ranges (e.g., Hue)
    """
    device, dtype = now.device, now.dtype
    steps = (maxv - minv) + 0.5
    rand_val = _min_of_k_uniform(now.shape, int(holdness), gen, device, dtype)
    flag = torch.where(torch.rand(now.shape, generator=gen, device=device, dtype=dtype) < 0.5,
                       torch.tensor(-1.0, device=device, dtype=dtype),
                       torch.tensor( 1.0, device=device, dtype=dtype))
    delta = (rand_max * rand_val) % steps
    newv = now + flag * delta
    if wraps:
        newv = torch.where(newv < minv, newv + steps, newv)
        newv = torch.where(newv > maxv, newv - steps, newv)
    else:
        newv = torch.clamp(newv, minv, maxv)
    return newv

# ---------- ComfyUI Node ----------

class CIELChNoiseGEGLLike:
    """Apply CIE LCh noise similarly to GEGL's gegl:noise-cie-lch.

    Inputs:
      - lightness_distance: max delta L (0..100)
      - chroma_distance:    max delta C (0..100)
      - hue_distance:       max delta H degrees (0..180)
      - holdness:           (aka Dulling) integer 1..8
      - seed:               RNG seed for determinism
    Alpha is preserved.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lightness_distance": ("FLOAT", {"default": 40.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "chroma_distance":    ("FLOAT", {"default": 40.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "hue_distance":       ("FLOAT", {"default": 3.0,  "min": 0.0, "max": 180.0, "step": 0.1}),
                "holdness":           ("INT",   {"default": 2,    "min": 1,   "max": 8}),
                "seed":               ("INT",   {"default": 0,    "min": -2**31, "max": 2**31 - 1}),
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

    def _prep(self, image: torch.Tensor):
        assert image.dim() == 4 and image.shape[-1] in (3,4), "Expected [B,H,W,3|4]"
        if image.shape[-1] == 3:
            return image.to(torch.float32), None, False
        return image[..., :3].to(torch.float32), image[..., 3:4].to(torch.float32), True

    def apply(self, image: torch.Tensor,
              lightness_distance: float, chroma_distance: float, hue_distance: float,
              holdness: int, seed: int):
        device = image.device
        dtype = torch.float32
        gen = self._make_generator(device, seed)

        rgb, a, has_alpha = self._prep(image)

        # sRGB -> linear -> XYZ -> Lab -> LCh
        rgb_lin = srgb_to_linear_torch(rgb)
        xyz = rgb_to_xyz_d65_torch(rgb_lin)
        lab = xyz_to_lab_d65_torch(xyz)
        lch = lab_to_lch_torch(lab)

        L = lch[..., 0]
        C = lch[..., 1]
        H = lch[..., 2]

        # Hue (wraps 0..360) only if C>0  [fix: mask assignment to avoid shape mismatch]
        if hue_distance > 0:
            mask_Cpos = (C > 0.0)
            if mask_Cpos.any():
                H_new = _randomize_value(H[mask_Cpos], 0.0, 359.0, True, hue_distance, holdness, gen)
                H = H.clone()
                H[mask_Cpos] = H_new

        # Chroma (clamped 0..100); for C==0 assign random hue first  [fix: mask assignment]
        if chroma_distance > 0:
            mask_Czero = (C == 0)
            if mask_Czero.any():
                H_rand = torch.rand(H[mask_Czero].shape, generator=gen, device=device, dtype=dtype) * 360.0
                H = H.clone()
                H[mask_Czero] = H_rand
            C = _randomize_value(C, 0.0, 100.0, False, chroma_distance, holdness, gen)

        # Lightness (0..100)
        if lightness_distance > 0:
            L = _randomize_value(L, 0.0, 100.0, False, lightness_distance, holdness, gen)

        # Back to sRGB
        lch_noisy = torch.stack([L, C, H], dim=-1)
        lab_noisy = lch_to_lab_torch(lch_noisy)
        xyz_noisy = lab_to_xyz_d65_torch(lab_noisy)
        rgb_lin_noisy = xyz_to_rgb_d65_torch(xyz_noisy)
        rgb_noisy = linear_to_srgb_torch(rgb_lin_noisy).clamp(0.0, 1.0)

        out = torch.cat([rgb_noisy, a], dim=-1) if has_alpha else rgb_noisy
        return (out,)

NODE_CLASS_MAPPINGS = {
    "CIELChNoiseGEGLLike": CIELChNoiseGEGLLike,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CIELChNoiseGEGLLike": "CIE LCh Noise (GEGL-like)",
}

