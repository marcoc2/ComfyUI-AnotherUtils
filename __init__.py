"""
ComfyUI Animation Nodes
A collection of custom nodes for animation and character generation in ComfyUI
"""

__version__ = "1.0.0"
__author__ = "marcoags"
__package_name__ = "AnotherUtils"

from .custom_crop import CustomCropNode
from .smart_resize import SmartResizeNode
from .nearest_upscale import NearestUpscaleNode
from .load_images import LoadImagesOriginalSize
from .pixel_normalizer import PixelArtNormalizerNode
from .fighting_game_character import FightingGameCharacter
from .walking_pose import WalkingPoseGenerator
from .load_remove_alpha import LoadImageRemoveAlpha
from .pixel_art_converter import PixelArtConverterNode
from .pixel_art_converter_parallel import PixelArtConverterNodeParallel
from .last_image import LastImage
from .character_constructor import CharacterConstructor
from .character_generator import CharacterRandomizer
from .remove_alpha import RemoveAlphaNode
from .gimp_nodes.adaptive_noise import AdaptiveNoise
from .gimp_nodes.cie_lch_noise_gegl_like import CIELChNoiseGEGLLike
from .gimp_nodes.image_type_detector import ImageTypeDetector
from .gimp_nodes.mean_curvature_blur_gegl_like import MeanCurvatureBlurGEGLLike
from .gimp_nodes.rgb_noise_gegl_like import RGBNoiseGEGLLike
from .csv_prompt_loader import CSVPromptLoader
from .comparison_swipe import ComparisonSwipeNode
from .folder_video_concatenator import FolderVideoConcatenator
from .interactive_crop import InteractiveCropNode

NODE_CLASS_MAPPINGS = {
    "CustomCrop": CustomCropNode,
    "SmartResize": SmartResizeNode,
    "NearestUpscale": NearestUpscaleNode,
    "LoadImagesOriginal": LoadImagesOriginalSize,
    "PixelArtNormalizer": PixelArtNormalizerNode,
    "FightingGameCharacter": FightingGameCharacter,
    "WalkingPoseGenerator": WalkingPoseGenerator,
    "LoadImageRemoveAlpha": LoadImageRemoveAlpha,
    "PixelArtConverter": PixelArtConverterNode,
    "PixelArtConverterParallel": PixelArtConverterNodeParallel,
    "LastImage": LastImage,
    "CharacterConstructor": CharacterConstructor,
    "CharacterRandomizer": CharacterRandomizer,
    "RemoveAlpha": RemoveAlphaNode,
    "AdaptiveNoise": AdaptiveNoise,
    "CIELChNoiseGEGLLike": CIELChNoiseGEGLLike,
    "ImageTypeDetector": ImageTypeDetector,
    "MeanCurvatureBlurGEGLLike": MeanCurvatureBlurGEGLLike,
    "RGBNoiseGEGLLike": RGBNoiseGEGLLike,
    "CSVPromptLoader": CSVPromptLoader,
    "ComparisonSwipe": ComparisonSwipeNode,
    "FolderVideoConcatenator": FolderVideoConcatenator,
    "InteractiveCrop": InteractiveCropNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomCrop": "Custom Crop",
    "SmartResize": "Smart Resize with Border Fill",
    "NearestUpscale": "Nearest Neighbor Upscale",
    "LoadImagesOriginal": "Load Images (Original Size)",
    "PixelArtNormalizer": "Pixel Art Normalizer",
    "FightingGameCharacter": "Fighting Game Character Generator",
    "WalkingPoseGenerator": "Walking Pose Generator",
    "LoadImageRemoveAlpha": "Load Image (Remove Alpha)",
    "PixelArtConverterParallel": "Pixel Art Converter Parallel",
    "LastImage": "Get Last Image",
    "CharacterConstructor": "Character Constructor",
    "CharacterRandomizer": "Character Randomizer",
    "RemoveAlpha": "Remove Alpha Channel",
    "AdaptiveNoise": "Adaptive Noise Filter",
    "CIELChNoiseGEGLLike": "CIE LCH Noise (GEGL-like)",
    "ImageTypeDetector": "Image Type Detector",
    "MeanCurvatureBlurGEGLLike": "Mean Curvature Blur (GEGL-like)",
    "RGBNoiseGEGLLike": "RGB Noise (GEGL-like)",
    "CSVPromptLoader": "CSV Prompt Loader",
    "ComparisonSwipe": "Comparison Swipe Video",
    "FolderVideoConcatenator": "Folder Video Concatenator (OpenCV)",
    "InteractiveCrop": "Interactive Crop"
}
WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]