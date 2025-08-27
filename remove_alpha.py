# remove_alpha.py
import torch
import numpy as np

class RemoveAlphaNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "remove_alpha"
    CATEGORY = "image/postprocessing"

    def remove_alpha(self, images):
        # Debug: print shape and value ranges
        print(f"Input shape: {images.shape}")
        print(f"Input value range: {images.min().item():.3f} to {images.max().item():.3f}")
        
        # Se não tem canal alpha (RGBA), retorna a imagem como está
        if images.shape[3] != 4:
            return (images,)
            
        # Separa os canais RGB e alpha
        rgb = images[..., :3]
        alpha = images[..., 3:]
        
        print(f"RGB value range: {rgb.min().item():.3f} to {rgb.max().item():.3f}")
        print(f"Alpha value range: {alpha.min().item():.3f} to {alpha.max().item():.3f}")
        
        # Cria o background branco no mesmo formato que o RGB
        white_bg = torch.ones_like(rgb)
        
        # Expande alpha para ter 3 canais como RGB
        alpha = alpha.expand(-1, -1, -1, 3)
        
        # Combina imagem com fundo branco usando alpha
        result = rgb * alpha + white_bg * (1 - alpha)
        
        print(f"Output value range: {result.min().item():.3f} to {result.max().item():.3f}")
        
        return (result,)

    @classmethod
    def IS_CHANGED(cls):
        return False