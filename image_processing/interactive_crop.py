import torch
import numpy as np
import os
import folder_paths
from PIL import Image, ImageOps

class InteractiveCropNode:
    """
    A node that loads an image and allows interactive cropping via a specific ROI size.
    Behaves like LoadImage but with cropping capabilities.
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        files = sorted(files)
        
        return {
            "required": {
                "image": (files, {"image_upload": True}),
                "roi_width": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "roi_height": ("INT", {"default": 512, "min": 1, "max": 16384}),
                "crop_x": ("INT", {"default": 0, "min": 0, "max": 16384}),
                "crop_y": ("INT", {"default": 0, "min": 0, "max": 16384}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "x", "y", "width", "height")
    FUNCTION = "load_and_crop"
    OUTPUT_NODE = True 
    CATEGORY = "image/processing"

    def load_and_crop(self, image, roi_width, roi_height, crop_x=0, crop_y=0):
        x = crop_x
        y = crop_y

        image_path = folder_paths.get_annotated_filepath(image)
        
        i = Image.open(image_path)
        i = ImageOps.exif_transpose(i)
        
        if i.mode == 'I':
            i = i.point(lambda i: i * (1 / 255))
        
        image = i.convert("RGB")
        
        # Dimensions
        img_w, img_h = image.size
        
        # Validate ROI
        roi_w = min(roi_width, img_w)
        roi_h = min(roi_height, img_h)
        
        # Validate Coordinates
        final_x = max(0, min(x, img_w - roi_w))
        final_y = max(0, min(y, img_h - roi_h))
        
        # Crop
        crop = image.crop((final_x, final_y, final_x + roi_w, final_y + roi_h))
        
        # Convert to Tensor
        image_np = np.array(crop).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)[None,]
        
        # Mask handling
        if 'A' in i.getbands():
            mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - mask 
            mask_crop = Image.fromarray((mask * 255).astype(np.uint8)).crop((final_x, final_y, final_x + roi_w, final_y + roi_h))
            mask_tensor = torch.from_numpy(np.array(mask_crop).astype(np.float32) / 255.0)[None,]
        else:
            mask_tensor = torch.zeros((1, roi_h, roi_w), dtype=torch.float32)

        return (image_tensor, mask_tensor, final_x, final_y, roi_w, roi_h)

    @classmethod
    def IS_CHANGED(cls, image, roi_width, roi_height, crop_x, crop_y):
        image_path = folder_paths.get_annotated_filepath(image)
        m = os.path.getmtime(image_path)
        return f"{image_path}_{m}_{roi_width}_{roi_height}_{crop_x}_{crop_y}"

    @classmethod
    def VALIDATE_INPUTS(cls, image, **kwargs):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
