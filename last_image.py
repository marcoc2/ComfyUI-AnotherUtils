import torch
import numpy as np
from PIL import Image

class LastImage:
    """
    A custom node for ComfyUI that takes a list of images as input and returns only the last image from the list.
    """
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get_last_image"
    CATEGORY = "image/utils"

    def get_last_image(self, images):
        # If images is a batch, just take the last one
        if len(images.shape) == 4:  # [batch, height, width, channels]
            last_image = images[-1:].clone()  # Clone to avoid modifying the original tensor
            return (last_image,)
        else:
            # Handle the case where only one image is provided
            return (images.unsqueeze(0),)  # Return as a batch of one image
