import sys
import os
import types

# Mock folder_paths
mock_folder_paths = types.ModuleType("folder_paths")
sys.modules["folder_paths"] = mock_folder_paths

# Add project root to sys.path
sys.path.insert(0, r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\custom_nodes\ComfyUI-AnotherUtils")

from loaders.utils import PromptExtractor
from PIL import Image
import json

def test_extraction():
    image_path = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\custom_nodes\ComfyUI-AnotherUtils\reference\comfyui_output.png"
    img = Image.open(image_path)
    
    # Test shared utility directly
    extracted = PromptExtractor.extract_from_image(img)
    
    print(f"Refactored Extraction Test Results:")
    print(f"Target: enhance and restore it")
    print(f"Extracted: {extracted}")
    
    if extracted == "enhance and restore it":
        print("SUCCESS: Refactored logic works for reference image.")
    else:
        print("FAILURE: Refactored logic failed.")

if __name__ == "__main__":
    test_extraction()
