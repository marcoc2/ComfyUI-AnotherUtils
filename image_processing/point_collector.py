import torch
import numpy as np
import json
import io
import base64
from PIL import Image

class PointCollectorSAM2:
    """
    Interactive Point Collector for SAM 2. 
    Outputs pixel coordinates in JSON format: [{"x": 1, "y": 2}]
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                # Hidden widgets populated by JS
                "coordinates": ("STRING", {"multiline": False, "default": "[]"}),
                "neg_coordinates": ("STRING", {"multiline": False, "default": "[]"}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("pos_points_json", "neg_points_json")
    FUNCTION = "collect"
    CATEGORY = "AnotherUtils/sam2"
    OUTPUT_NODE = True

    def collect(self, image, coordinates, neg_coordinates):
        # Coordinates from JS are already in pixel units relative to the image
        # We just need to ensure they are valid JSON strings for our SAM2 nodes
        
        pos_json = coordinates if coordinates and coordinates.strip() else "[]"
        neg_json = neg_coordinates if neg_coordinates and neg_coordinates.strip() else "[]"

        # Send image to the JS widget via a UI message
        img_base64 = self.tensor_to_base64(image)
        
        return {
            "ui": {"bg_image": [img_base64]},
            "result": (pos_json, neg_json)
        }

    def tensor_to_base64(self, tensor):
        # Convert from [B, H, W, C] to PIL Image (first frame)
        img_array = tensor[0].cpu().numpy()
        img_array = (img_array * 255).astype(np.uint8)
        pil_img = Image.fromarray(img_array)

        buffered = io.BytesIO()
        pil_img.save(buffered, format="JPEG", quality=75)
        img_bytes = buffered.getvalue()
        img_base64 = base64.b64encode(img_bytes).decode('utf-8')

        return img_base64

    @classmethod
    def IS_CHANGED(cls, image, coordinates, neg_coordinates):
        return float("nan") # Always update UI when points change
