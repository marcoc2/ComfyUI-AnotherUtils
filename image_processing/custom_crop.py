# custom_nodes/image_processing/custom_crop.py
import torch
import numpy as np
from PIL import Image

class CustomCropNode:
    def __init__(self):
        self.crop_modes = ["center", "left", "right", "top", "bottom"]
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "crop_width": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "crop_height": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "crop_mode": (["center", "left", "right", "top", "bottom"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "crop_image"
    CATEGORY = "image/processing"

    def crop_image(self, image, crop_width, crop_height, crop_mode):
        # Converter o tensor para PIL Image para facilitar o cropping
        if isinstance(image, torch.Tensor):
            image_np = image[0].cpu().numpy()
            image_pil = Image.fromarray((image_np * 255).astype(np.uint8))
        else:
            image_pil = image

        # Pegar dimensões originais
        orig_width, orig_height = image_pil.size

        # Calcular coordenadas de crop baseado no modo
        if crop_mode == "center":
            left = (orig_width - crop_width) // 2
            top = (orig_height - crop_height) // 2
        elif crop_mode == "left":
            left = 0
            top = (orig_height - crop_height) // 2
        elif crop_mode == "right":
            left = orig_width - crop_width
            top = (orig_height - crop_height) // 2
        elif crop_mode == "top":
            left = (orig_width - crop_width) // 2
            top = 0
        else:  # bottom
            left = (orig_width - crop_width) // 2
            top = orig_height - crop_height

        # Ajustar coordenadas se necessário para evitar crops fora da imagem
        left = max(0, min(left, orig_width - crop_width))
        top = max(0, min(top, orig_height - crop_height))
        
        # Realizar o crop
        cropped_image = image_pil.crop((left, top, left + crop_width, top + crop_height))
        
        # Converter de volta para tensor
        cropped_np = np.array(cropped_image).astype(np.float32) / 255.0
        cropped_tensor = torch.from_numpy(cropped_np).unsqueeze(0)

        return (cropped_tensor,)