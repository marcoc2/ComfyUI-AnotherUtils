# image_type_detector.py
"""
ComfyUI Custom Node — Image Type Detector
Detects if input is single image or batch and provides outputs for both cases.

This node always succeeds and provides valid outputs for both single and batch workflows.
Use the batch_count output to manually route in your workflow or for debugging.
"""

import torch

class ImageTypeDetector:
    """
    Detects image type and provides appropriate outputs for workflow routing.
    
    Always provides valid outputs:
    - single_image: First image from input (works for both single and batch)
    - batch_images: Full batch (works for both single and batch)
    - batch_count: Number of images in batch
    - is_single: True if single image, False if batch
    
    For single images: both outputs contain the same single image
    For batches: single_image contains first frame, batch_images contains all frames
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "BOOLEAN")
    RETURN_NAMES = ("single_image", "batch_images", "batch_count", "is_single")
    FUNCTION = "detect"
    CATEGORY = "image/utility"

    def detect(self, image: torch.Tensor):
        """
        Analyze image and provide appropriate outputs.
        
        Returns:
            - single_image: First image [1,H,W,C]
            - batch_images: Full batch [B,H,W,C] 
            - batch_count: Number of images (int)
            - is_single: True if single image (bool)
        """
        if image.dim() != 4:
            raise ValueError(f"Expected 4D tensor [B,H,W,C], got {image.dim()}D")
        
        batch_size = image.shape[0]
        
        # Always provide valid outputs
        single_image = image[0:1]  # First image as [1,H,W,C]
        batch_images = image       # Full batch [B,H,W,C]
        batch_count = batch_size
        is_single = batch_size == 1
        
        return (single_image, batch_images, batch_count, is_single)

NODE_CLASS_MAPPINGS = {
    "ImageTypeDetector": ImageTypeDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageTypeDetector": "Image Type Detector",
}