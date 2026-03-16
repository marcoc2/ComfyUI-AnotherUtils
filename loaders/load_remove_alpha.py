# load_remove_alpha.py
import torch
import os
from PIL import Image
import numpy as np

class LoadImageRemoveAlpha:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("STRING", {"default": ""}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "load_and_remove_alpha"
    CATEGORY = "image/loaders"
    TITLE = "Load Image (Remove Alpha)"

    def load_and_remove_alpha(self, image):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image not found: {image}")
            
        i = Image.open(image)
        i = i.convert('RGBA')  # Converte para RGBA para garantir
        
        # Cria background branco
        background = Image.new('RGB', i.size, (255, 255, 255))
        # Combina com a imagem usando alpha
        background.paste(i, mask=i.split()[3])
        
        # Converte para o formato do ComfyUI
        image = np.array(background).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]
        
        return (image,)

    @classmethod
    def IS_CHANGED(cls):
        return False

    @classmethod
    def VALIDATE_INPUTS(cls, image):
        if not os.path.exists(image):
            return "Image path does not exist"
        return True