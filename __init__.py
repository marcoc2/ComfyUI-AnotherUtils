"""
ComfyUI Animation Nodes
A collection of custom nodes for animation and character generation in ComfyUI
"""

__version__ = "1.0.0"
__author__ = "marcoags"
__package_name__ = "AnotherUtils"

from .image_processing.custom_crop import CustomCropNode
from .image_processing.smart_resize import SmartResizeNode
from .image_processing.nearest_upscale import NearestUpscaleNode
from .image_processing.image_grid_slicer import ImageGridSlicer
from .loaders.load_images import LoadImagesOriginalSize
from .pixel_art.pixel_normalizer import PixelArtNormalizerNode
from .characters.fighting_game_character import FightingGameCharacter
from .characters.walking_pose import WalkingPoseGenerator
from .loaders.load_remove_alpha import LoadImageRemoveAlpha
from .pixel_art.pixel_art_converter import PixelArtConverterNode
from .pixel_art.pixel_art_converter_parallel import PixelArtConverterNodeParallel
from .loaders.last_image import LastImage
from .characters.character_constructor import CharacterConstructor
from .characters.character_generator import CharacterRandomizer
from .image_processing.remove_alpha import RemoveAlphaNode
from .gimp_nodes.adaptive_noise import AdaptiveNoise
from .gimp_nodes.cie_lch_noise_gegl_like import CIELChNoiseGEGLLike
from .gimp_nodes.image_type_detector import ImageTypeDetector
from .gimp_nodes.mean_curvature_blur_gegl_like import MeanCurvatureBlurGEGLLike
from .gimp_nodes.rgb_noise_gegl_like import RGBNoiseGEGLLike
from .loaders.csv_prompt_loader import CSVPromptLoader
from .video.comparison_swipe import ComparisonSwipeNode
from .video.folder_video_concatenator import FolderVideoConcatenator
from .image_processing.interactive_crop import InteractiveCropNode
from .loaders.caption_image_loader import CaptionImageLoader
from .video.video_audio_combiner import (
    VideoAudioCombiner,
    VideoAudioCombinerSimple,
    HAS_NEW_VIDEO_API,
)

if HAS_NEW_VIDEO_API:
    from .video.video_audio_combiner import VideoAudioCombinerV3
from .audio.audio_waveform_slicer import AudioWaveformSlicer
from .audio.audio_slice_selector import AudioSliceSelector
from .audio.audio_concatenate import AudioConcatenate
from .loaders.load_gif_frames import LoadGifFrames, RemapGifFrames
from .loaders.batch_image_list import BatchToImageList
from .video.video_auto_sync_hstack import VideoAutoSyncHStack
from .video.ltxv_multi_guide import LTXVMultiGuide
from .video.ltxv_vid2vid import LTXVVid2Vid
from .loaders.folder_image_loader import FolderImageLoader
from .logic_management.dataset_loader import DatasetLoader
from .logic_management.image_list_sampler import ImageListSampler
from .image_processing.segs_adapter import SEGStoBBox, SEGStoSAM2Points, GetFirstFrame, ManualPointToSAM2, RefineMask
from .image_processing.point_collector import PointCollectorSAM2
from .inference_nodes import (
    AnotherLoadInferenceModel,
    AnotherYOLOInference,
    AnotherSAM2Inference,
    AnotherDepthInference,
    AnotherBBoxToPoints,
    AnotherPoseToPoints,
    AnotherImageToMask,
    AnotherMaskToImage,
    AnotherMaskMath,
    AnotherMaskBlur
)
from .core import server_routes  # Register Custom API Routes

NODE_CLASS_MAPPINGS = {
    "CustomCrop": CustomCropNode,
    "SmartResize": SmartResizeNode,
    "NearestUpscale": NearestUpscaleNode,
    "ImageGridSlicer": ImageGridSlicer,
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
    "InteractiveCrop": InteractiveCropNode,
    "CaptionImageLoader": CaptionImageLoader,
    "VideoAudioCombiner": VideoAudioCombiner,
    "VideoAudioCombinerSimple": VideoAudioCombinerSimple,
    "AudioWaveformSlicer": AudioWaveformSlicer,
    "AudioSliceSelector": AudioSliceSelector,
    "AudioConcatenate": AudioConcatenate,
    "LoadGifFrames": LoadGifFrames,
    "RemapGifFrames": RemapGifFrames,
    "BatchToImageList": BatchToImageList,
    "VideoAutoSyncHStack": VideoAutoSyncHStack,
    "FolderImageLoader": FolderImageLoader,
    "DatasetLoader": DatasetLoader,
    "ImageListSampler": ImageListSampler,
    "LTXVMultiGuide": LTXVMultiGuide,
    "LTXVVid2Vid": LTXVVid2Vid,
    "SEGStoBBox": SEGStoBBox,
    "SEGStoSAM2Points": SEGStoSAM2Points,
    "GetFirstFrame": GetFirstFrame,
    "ManualPointToSAM2": ManualPointToSAM2,
    "RefineMask": RefineMask,
    "PointCollectorSAM2": PointCollectorSAM2,
    "AnotherLoadInferenceModel": AnotherLoadInferenceModel,
    "AnotherYOLOInference": AnotherYOLOInference,
    "AnotherSAM2Inference": AnotherSAM2Inference,
    "AnotherDepthInference": AnotherDepthInference,
    "AnotherBBoxToPoints": AnotherBBoxToPoints,
    "AnotherPoseToPoints": AnotherPoseToPoints,
    "AnotherImageToMask": AnotherImageToMask,
    "AnotherMaskToImage": AnotherMaskToImage,
    "AnotherMaskMath": AnotherMaskMath,
    "AnotherMaskBlur": AnotherMaskBlur,
}

# Add V3 nodes if the new API is available
if HAS_NEW_VIDEO_API:
    NODE_CLASS_MAPPINGS["VideoAudioCombinerV3"] = VideoAudioCombinerV3

NODE_DISPLAY_NAME_MAPPINGS = {
    "CustomCrop": "Custom Crop",
    "SmartResize": "Smart Resize with Border Fill",
    "NearestUpscale": "Nearest Neighbor Upscale",
    "ImageGridSlicer": "Image Grid Slicer",
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
    "InteractiveCrop": "Interactive Crop",
    "CaptionImageLoader": "Caption Image Loader",
    "VideoAudioCombiner": "Video + Audio Combiner",
    "VideoAudioCombinerSimple": "Video + Audio Combiner (Simple)",
    "AudioWaveformSlicer": "Audio Waveform Slicer",
    "AudioSliceSelector": "Audio Slice Selector",
    "AudioConcatenate": "Audio Concatenate",
    "LoadGifFrames": "Load GIF Frames (Raw)",
    "RemapGifFrames": "Remap GIF Frames",
    "BatchToImageList": "Batch to Image List",
    "VideoAutoSyncHStack": "Video Auto Sync HStack",
    "FolderImageLoader": "Folder Image Loader",
    "DatasetLoader": "Dataset Loader (Images + Captions)",
    "ImageListSampler": "Image List Sampler",
    "LTXVMultiGuide": "LTXV Multi Guide (N Frames)",
    "LTXVVid2Vid": "LTXV Vid2Vid Encode",
    "SEGStoBBox": "SEGS to BBox",
    "SEGStoSAM2Points": "SEGS to SAM2 Points (JSON)",
    "GetFirstFrame": "Get First Frame (Batch to Single)",
    "ManualPointToSAM2": "Manual Point to SAM2 (JSON)",
    "RefineMask": "Refine Mask (Expand & Blur)",
    "PointCollectorSAM2": "Interactive Point Collector (SAM2)",
    "AnotherLoadInferenceModel": "Load Inference Model (AnotherUtils)",
    "AnotherYOLOInference": "YOLO/Pose Inference (AnotherUtils)",
    "AnotherSAM2Inference": "SAM2 Image Inference (AnotherUtils)",
    "AnotherDepthInference": "DepthAnything Inference (AnotherUtils)",
    "AnotherBBoxToPoints": "BBox to Central Point (JSON)",
    "AnotherPoseToPoints": "Pose Keypoints to Points (JSON)",
    "AnotherImageToMask": "Image to Mask (AnotherUtils)",
    "AnotherMaskToImage": "Mask to Image (AnotherUtils)",
    "AnotherMaskMath": "Mask Mathematics (AnotherUtils)",
    "AnotherMaskBlur": "Mask Gaussian Blur (AnotherUtils)",
}

# Add V3 display names if available
if HAS_NEW_VIDEO_API:
    NODE_DISPLAY_NAME_MAPPINGS["VideoAudioCombinerV3"] = "Video + Audio Combiner (V3)"

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "server_routes"]
