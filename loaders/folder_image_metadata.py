import os
import hashlib
import numpy as np
import torch
from PIL import Image
import folder_paths
from .utils import PromptExtractor

class FolderImageAndExtractPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "Path to directory of images"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "prompts")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_images"
    CATEGORY = "AnotherUtils/loaders"

    def load_images(self, directory: str):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        image_files = [f for f in os.listdir(directory) 
                      if f.lower().endswith(valid_extensions)]
        
        if len(image_files) == 0:
            raise FileNotFoundError(f"No valid image files found in directory '{directory}'.")
        
        image_files.sort()
        
        images_list = []
        prompts_list = []

        for filename in image_files:
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath)
                
                # Image processing
                image_tensor = PromptExtractor.preprocess_image(img)
                images_list.append(image_tensor)
                
                # Prompt extraction
                prompt = PromptExtractor.extract_from_image(img)
                prompts_list.append(prompt)
                
            except Exception as e:
                print(f"[AnotherUtils] Error loading image {filename}: {e}")
        
        return (images_list, prompts_list)
    
    @classmethod
    def IS_CHANGED(s, directory: str):
        if not os.path.isdir(directory):
            return False
            
        m = hashlib.sha256()
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(valid_extensions):
                filepath = os.path.join(directory, filename)
                m.update(str(os.path.getmtime(filepath)).encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, directory: str):
        if not directory.strip():
            return "Directory path cannot be empty"
            
        if not os.path.isdir(directory):
            return f"Directory '{directory}' cannot be found."
        return True
