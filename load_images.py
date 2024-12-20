import os
import hashlib
import numpy as np
import torch
from PIL import Image
import folder_paths

class LoadImagesOriginalSize:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "Path to directory of images"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_images"
    CATEGORY = "image/loading"

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
        for filename in image_files:
            filepath = os.path.join(directory, filename)
            
            img = Image.open(filepath)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            img_np = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_np).unsqueeze(0)
            images_list.append(img_tensor)
        
        return (images_list,)
    
    @classmethod
    def IS_CHANGED(cls, directory: str):
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
    def VALIDATE_INPUTS(cls, directory: str):
        if not directory.strip():
            return "Directory path cannot be empty"
            
        if not os.path.isdir(directory):
            return f"Directory '{directory}' cannot be found."
        return True