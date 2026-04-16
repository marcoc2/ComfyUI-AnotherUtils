"""
ComfyUI Animation Nodes
A collection of custom nodes for animation and character generation in ComfyUI
"""

__version__ = "1.0.0"
__author__ = "marcoags"
__package_name__ = "AnotherUtils"

import logging

# Basic Image Processing
from .image_processing.custom_crop import CustomCropNode
from .image_processing.smart_resize import SmartResizeNode
from .image_processing.nearest_upscale import NearestUpscaleNode
from .image_processing.image_grid_slicer import ImageGridSlicer
from .image_processing.remove_alpha import RemoveAlphaNode
from .image_processing.interactive_crop import InteractiveCropNode
from .image_processing.image_composite_masked import AnotherImageCompositeMasked
from .image_processing.segs_adapter import SEGStoBBox, SEGStoSAM2Points, GetFirstFrame, ManualPointToSAM2, RefineMask
from .image_processing.point_collector import PointCollectorSAM2

# Loaders
from .loaders.load_images import LoadImagesOriginalSize
from .loaders.load_remove_alpha import LoadImageRemoveAlpha
from .loaders.last_image import LastImage
from .loaders.csv_prompt_loader import CSVPromptLoader
from .loaders.trello_prompt_loader import TrelloPromptLoader
from .loaders.trello_browser import TrelloBrowser
from .loaders.caption_image_loader import CaptionImageLoader
from .loaders.load_image_metadata import LoadImageAndExtractPrompt
from .loaders.folder_image_metadata import FolderImageAndExtractPrompt
from .loaders.folder_image_metadata_by_name import FolderImageMetadataByName
from .loaders.save_image_with_tag import AnotherSaveImageWithTag
from .loaders.load_gif_frames import LoadGifFrames, RemapGifFrames
from .loaders.batch_image_list import BatchToImageList
from .loaders.folder_image_loader import FolderImageLoader

# Pixel Art
from .pixel_art.pixel_normalizer import PixelArtNormalizerNode
from .pixel_art.pixel_art_converter import PixelArtConverterNode
from .pixel_art.pixel_art_converter_parallel import PixelArtConverterNodeParallel

# Characters
from .characters.fighting_game_character import FightingGameCharacter
from .characters.walking_pose import WalkingPoseGenerator
from .characters.character_constructor import CharacterConstructor
from .characters.character_generator import CharacterRandomizer

# GIMP / GEGL Like
from .gimp_nodes.adaptive_noise import AdaptiveNoise
from .gimp_nodes.cie_lch_noise_gegl_like import CIELChNoiseGEGLLike
from .gimp_nodes.image_type_detector import ImageTypeDetector
from .gimp_nodes.mean_curvature_blur_gegl_like import MeanCurvatureBlurGEGLLike
from .gimp_nodes.rgb_noise_gegl_like import RGBNoiseGEGLLike

# Video General
from .video.comparison_swipe import ComparisonSwipeNode
from .video.folder_video_concatenator import FolderVideoConcatenator
from .video.animated_composite import AnotherTransformKeyframes, AnotherAnimatedCompositeMasked, AnotherTransformOrchestrator
from .video.camera_switcher import AnotherCameraSwitcher
from .video.video_auto_sync_hstack import VideoAutoSyncHStack
from .video.video_audio_combiner import (
    VideoAudioCombiner,
    VideoAudioCombinerSimple,
    HAS_NEW_VIDEO_API,
)

# Audio
from .audio.audio_waveform_slicer import AudioWaveformSlicer
from .audio.audio_slice_selector import AudioSliceSelector
from .audio.audio_concatenate import AudioConcatenate

# Logic & Management
from .logic_management.image_list_to_batch import ImageListToBatch
from .logic_management.indices_list_to_50 import IndicesListTo50
from .logic_management.dataset_loader import DatasetLoader
from .logic_management.image_list_sampler import ImageListSampler
from .logic_management.debug_list import AnotherShowList

# Inference
from .inference_nodes import (
    AnotherLoadYOLO,
    AnotherLoadSAM2,
    AnotherLoadDepth,
    AnotherYOLOInference,
    AnotherSAM2Inference,
    AnotherSAM2VideoAddPoints,
    AnotherSAM2VideoPropagate,
    AnotherDepthInference,
    AnotherBBoxToPoints,
    AnotherPoseToPoints,
    AnotherImageToMask,
    AnotherMaskToImage,
    AnotherMaskMath,
    AnotherMaskBlur
)

from .core import server_routes  # Register Custom API Routes

# Initial Mappings
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
    "AnotherShowList": AnotherShowList,
    "SEGStoBBox": SEGStoBBox,
    "SEGStoSAM2Points": SEGStoSAM2Points,
    "GetFirstFrame": GetFirstFrame,
    "ManualPointToSAM2": ManualPointToSAM2,
    "RefineMask": RefineMask,
    "PointCollectorSAM2": PointCollectorSAM2,
    "AnotherLoadYOLO": AnotherLoadYOLO,
    "AnotherLoadSAM2": AnotherLoadSAM2,
    "AnotherLoadDepth": AnotherLoadDepth,
    "AnotherYOLOInference": AnotherYOLOInference,
    "AnotherSAM2Inference": AnotherSAM2Inference,
    "AnotherSAM2VideoAddPoints": AnotherSAM2VideoAddPoints,
    "AnotherSAM2VideoPropagate": AnotherSAM2VideoPropagate,
    "AnotherDepthInference": AnotherDepthInference,
    "AnotherBBoxToPoints": AnotherBBoxToPoints,
    "AnotherPoseToPoints": AnotherPoseToPoints,
    "AnotherTransformKeyframes": AnotherTransformKeyframes,
    "AnotherAnimatedCompositeMasked": AnotherAnimatedCompositeMasked,
    "AnotherTransformOrchestrator": AnotherTransformOrchestrator,
    "AnotherCameraSwitcher": AnotherCameraSwitcher,
    "AnotherImageCompositeMasked": AnotherImageCompositeMasked,
    "AnotherImageToMask": AnotherImageToMask,
    "AnotherMaskToImage": AnotherMaskToImage,
    "AnotherMaskMath": AnotherMaskMath,
    "AnotherMaskBlur": AnotherMaskBlur,
    "TrelloPromptLoader": TrelloPromptLoader,
    "TrelloBrowser": TrelloBrowser,
    "LoadImageAndExtractPrompt": LoadImageAndExtractPrompt,
    "FolderImageAndExtractPrompt": FolderImageAndExtractPrompt,
    "FolderImageMetadataByName": FolderImageMetadataByName,
    "AnotherSaveImageWithTag": AnotherSaveImageWithTag,
    "FolderImageLoader": FolderImageLoader,
    "ImageListToBatch": ImageListToBatch,
    "IndicesListTo50": IndicesListTo50,
}

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
    "AnotherShowList": "Debug List (AnotherUtils)",
    "SEGStoBBox": "SEGS to BBox",
    "SEGStoSAM2Points": "SEGS to SAM2 Points (JSON)",
    "GetFirstFrame": "Get First Frame (Batch to Single)",
    "ManualPointToSAM2": "Manual Point to SAM2 (JSON)",
    "RefineMask": "Refine Mask (Expand & Blur)",
    "PointCollectorSAM2": "Interactive Point Collector (SAM2)",
    "AnotherLoadYOLO": "Load YOLO Model (AnotherUtils)",
    "AnotherLoadSAM2": "Load SAM2 Model (AnotherUtils)",
    "AnotherLoadDepth": "Load Depth Model (AnotherUtils)",
    "AnotherYOLOInference": "YOLO/Pose Inference (AnotherUtils)",
    "AnotherSAM2Inference": "SAM2 Image Inference (AnotherUtils)",
    "AnotherSAM2VideoAddPoints": "SAM2 Video Add Points (AnotherUtils)",
    "AnotherSAM2VideoPropagate": "SAM2 Video Propagate (AnotherUtils)",
    "AnotherDepthInference": "DepthAnything Inference (AnotherUtils)",
    "AnotherBBoxToPoints": "BBox to Central Point (JSON)",
    "AnotherPoseToPoints": "Pose Keypoints to Points (JSON)",
    "AnotherTransformKeyframes": "Transform Keyframes (Math)",
    "AnotherAnimatedCompositeMasked": "Animated Composite Masked (Batch)",
    "AnotherTransformOrchestrator": "Transform Orchestrator (Multi-Segment)",
    "AnotherCameraSwitcher": "Camera Switcher (Multi-Video Director)",
    "AnotherImageCompositeMasked": "ImageCompositeMasked (AnotherUtils Memory Safe)",
    "AnotherImageToMask": "Image to Mask (AnotherUtils)",
    "AnotherMaskToImage": "Mask to Image (AnotherUtils)",
    "AnotherMaskMath": "Mask Mathematics (AnotherUtils)",
    "AnotherMaskBlur": "Mask Gaussian Blur (AnotherUtils)",
    "TrelloPromptLoader": "Trello Prompt Loader",
    "TrelloBrowser": "Trello Browser (Advanced)",
    "LoadImageAndExtractPrompt": "Load Image and Extract Prompt",
    "FolderImageAndExtractPrompt": "Folder Image and Extract Prompt",
    "FolderImageMetadataByName": "Folder Metadata by Node Name",
    "AnotherSaveImageWithTag": "Another Save Image with Tag",
    "FolderImageLoader": "Folder Image Loader",
    "ImageListToBatch": "Image List To Multi Batch",
    "IndicesListTo 50": "Indices List To 50 Inputs",
}

# LTX Video Specific - Conditional Loading
if HAS_NEW_VIDEO_API:
    try:
        from .video.video_audio_combiner import VideoAudioCombinerV3
        from .video.ltxv_multi_guide import LTXVMultiGuide
        from .video.another_ltx_sequencer import AnotherLTXSequencer
        from .video.ltxv_multi_concat import LTXVMultiConcat
        from .video.ltxv_multi_concat_beta import LTXVMultiConcatBeta
        from .video.ltxv_vid2vid import LTXVVid2Vid
        from .video.ltxv_diagnostic import LTXVDiagnosticNode

        NODE_CLASS_MAPPINGS.update({
            "VideoAudioCombinerV3": VideoAudioCombinerV3,
            "LTXVMultiGuide": LTXVMultiGuide,
            "AnotherLTXSequencer": AnotherLTXSequencer,
            "LTXVMultiConcat": LTXVMultiConcat,
            "LTXVMultiConcatBeta": LTXVMultiConcatBeta,
            "LTXVVid2Vid": LTXVVid2Vid,
            "LTXVDiagnosticNode": LTXVDiagnosticNode,
        })

        NODE_DISPLAY_NAME_MAPPINGS.update({
            "VideoAudioCombinerV3": "Video + Audio Combiner (V3)",
            "LTXVMultiGuide": "LTXV Multi Guide (N Frames)",
            "AnotherLTXSequencer": "LTX Sequencer (Automated)",
            "LTXVMultiConcat": "LTXV Multi Concat (N Frames)",
            "LTXVMultiConcatBeta": "LTXV Multi Concat (N Frames) (beta)",
            "LTXVVid2Vid": "LTXV Vid2Vid Encode",
            "LTXVDiagnosticNode": "LTXV Diagnostic (Shape Checker)",
        })
    except Exception as e:
        logging.error(f"Failed to load LTX Video extension nodes: {e}")

WEB_DIRECTORY = "./js"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "server_routes"]
