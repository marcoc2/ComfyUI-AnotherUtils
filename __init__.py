# custom_nodes/image_processing/__init__.py
from .custom_crop import CustomCropNode
from .smart_resize import SmartResizeNode
from .nearest_upscale import NearestUpscaleNode
from .load_images import LoadImagesOriginalSize
from .pixel_normalizer import PixelArtNormalizerNode

NODE_CLASS_MAPPINGS = {
    "CustomCrop": CustomCropNode,
    "SmartResize": SmartResizeNode,
    "NearestUpscale": NearestUpscaleNode,
    "LoadImagesOriginal": LoadImagesOriginalSize,
    "PixelArtNormalizer": PixelArtNormalizerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomCrop": "Custom Crop",
    "SmartResize": "Smart Resize with Border Fill",
    "NearestUpscale": "Nearest Neighbor Upscale",
    "LoadImagesOriginal": "Load Images (Original Size)",
    "PixelArtNormalizer": "Pixel Art Normalizer"
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]