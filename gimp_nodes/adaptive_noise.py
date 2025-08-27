# adaptive_noise.py
"""
ComfyUI Custom Node — Adaptive Noise Filter
Applies more noise to high-frequency regions (edges, details) and less noise to low-frequency regions (flat areas).

Uses edge detection and frequency analysis to create an adaptive noise mask that preserves smooth areas
while adding texture to detailed regions.

Inputs:
  - image (IMAGE)
  - noise_strength (FLOAT 0..1, default 0.3) - Overall noise intensity
  - adaptation_strength (FLOAT 0..2, default 1.0) - How much to adapt based on frequency (0=uniform, 2=very adaptive)
  - edge_threshold (FLOAT 0..1, default 0.1) - Sensitivity for edge detection
  - blur_radius (FLOAT 0.5..10, default 2.0) - Smoothing radius for frequency map
  - noise_type (combo) - Gaussian or Uniform noise
  - seed (INT) - Random seed for deterministic results
"""

import torch
import torch.nn.functional as F
import math

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

def rgb_to_luminance(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to luminance using standard weights."""
    weights = torch.tensor([0.299, 0.587, 0.114], device=rgb.device, dtype=rgb.dtype)
    weights = weights.view(1, 1, 1, 3)
    return torch.sum(rgb * weights, dim=-1, keepdim=True)

def sobel_edge_detection(img: torch.Tensor) -> torch.Tensor:
    """
    Apply Sobel edge detection to single-channel image.
    Input: [B,H,W,1] 
    Output: [B,H,W,1] edge magnitude
    """
    # Convert to [B,C,H,W] for conv2d
    img_nchw = img.permute(0, 3, 1, 2)
    
    # Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=img.dtype, device=img.device).view(1, 1, 3, 3)
    
    # Apply convolution with padding
    grad_x = F.conv2d(img_nchw, sobel_x, padding=1)
    grad_y = F.conv2d(img_nchw, sobel_y, padding=1)
    
    # Compute magnitude
    magnitude = torch.sqrt(grad_x**2 + grad_y**2)
    
    # Convert back to [B,H,W,1]
    return magnitude.permute(0, 2, 3, 1)

def gaussian_blur(img: torch.Tensor, radius: float) -> torch.Tensor:
    """
    Apply Gaussian blur to image.
    Input: [B,H,W,C]
    """
    if radius <= 0.5:
        return img
        
    # Convert to [B,C,H,W]
    img_nchw = img.permute(0, 3, 1, 2)
    
    # Create Gaussian kernel
    kernel_size = int(2 * math.ceil(2 * radius) + 1)
    sigma = radius / 3.0
    
    # 1D Gaussian kernel
    x = torch.arange(kernel_size, dtype=img.dtype, device=img.device) - kernel_size // 2
    kernel_1d = torch.exp(-x**2 / (2 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    
    # Separate horizontal and vertical convolution for efficiency
    kernel_h = kernel_1d.view(1, 1, 1, -1).expand(img_nchw.shape[1], 1, 1, -1)
    kernel_v = kernel_1d.view(1, 1, -1, 1).expand(img_nchw.shape[1], 1, -1, 1)
    
    # Apply separable convolution
    padding = kernel_size // 2
    blurred = F.conv2d(img_nchw, kernel_h, padding=(0, padding), groups=img_nchw.shape[1])
    blurred = F.conv2d(blurred, kernel_v, padding=(padding, 0), groups=img_nchw.shape[1])
    
    return blurred.permute(0, 2, 3, 1)

class AdaptiveNoise:
    """
    Apply noise adaptively based on local image frequency content.
    More noise in high-frequency areas (edges, textures), less in smooth areas.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "noise_strength": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "adaptation_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "edge_threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blur_radius": ("FLOAT", {"default": 2.0, "min": 0.5, "max": 10.0, "step": 0.1}),
                "noise_type": (["gaussian", "uniform"], {"default": "gaussian"}),
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

    def _prep(self, image: torch.Tensor):
        assert image.dim() == 4 and image.shape[-1] in (3,4), "Expected [B,H,W,3|4]"
        if image.shape[-1] == 3:
            return image.to(torch.float32), None, False
        return image[..., :3].to(torch.float32), image[..., 3:4].to(torch.float32), True

    def apply(self, image: torch.Tensor, noise_strength: float, adaptation_strength: float, 
              edge_threshold: float, blur_radius: float, noise_type: str, seed: int):
        
        device = image.device
        dtype = torch.float32
        gen = self._make_generator(device, seed)
        
        rgb, alpha, has_alpha = self._prep(image)
        b, h, w, c = rgb.shape
        
        # Convert to linear RGB for better frequency analysis
        rgb_linear = srgb_to_linear_torch(rgb)
        
        # Convert to luminance for edge detection
        luminance = rgb_to_luminance(rgb_linear)
        
        # Detect edges/high-frequency regions
        edge_map = sobel_edge_detection(luminance)
        
        # Normalize edge map
        edge_map = edge_map / (edge_map.max() + 1e-8)
        
        # Apply threshold to focus on significant edges
        edge_map = torch.where(edge_map > edge_threshold, edge_map, torch.zeros_like(edge_map))
        
        # Create adaptive noise mask
        # Base mask: more noise where there are edges
        noise_mask = edge_threshold + (1.0 - edge_threshold) * edge_map
        
        # Apply adaptation strength
        if adaptation_strength > 0:
            # Blend between uniform (adaptation_strength=0) and fully adaptive (adaptation_strength=1+)
            uniform_mask = torch.ones_like(noise_mask)
            noise_mask = torch.lerp(uniform_mask, noise_mask, adaptation_strength)
        
        # Smooth the noise mask to avoid artifacts
        noise_mask = gaussian_blur(noise_mask, blur_radius)
        
        # Expand noise mask to RGB channels
        noise_mask_rgb = noise_mask.expand(-1, -1, -1, 3)
        
        # Generate noise
        if noise_type == "gaussian":
            noise = torch.randn((b, h, w, 3), generator=gen, device=device, dtype=dtype)
        else:  # uniform
            noise = torch.rand((b, h, w, 3), generator=gen, device=device, dtype=dtype) * 2.0 - 1.0
        
        # Apply adaptive noise
        adaptive_noise = noise * noise_mask_rgb * noise_strength
        
        # Add noise to image
        noisy_rgb = rgb_linear + adaptive_noise
        noisy_rgb = noisy_rgb.clamp(0.0, 1.0)
        
        # Convert back to sRGB
        noisy_rgb_srgb = linear_to_srgb_torch(noisy_rgb)
        
        # Combine with alpha if present
        if has_alpha:
            result = torch.cat([noisy_rgb_srgb, alpha], dim=-1)
        else:
            result = noisy_rgb_srgb
            
        return (result.clamp(0.0, 1.0),)

NODE_CLASS_MAPPINGS = {
    "AdaptiveNoise": AdaptiveNoise,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AdaptiveNoise": "Adaptive Noise",
}