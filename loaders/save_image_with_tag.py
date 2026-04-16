import os
import json
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import folder_paths
import comfy.utils

class AnotherSaveImageWithTag:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.prefix_append = ""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "custom_tag": ("STRING", {"forceInput": True}),
                "filename_prefix": ("STRING", {"default": "AnotherTag"}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "AnotherUtils/loaders"

    def save_images(self, images, custom_tag, filename_prefix):
        filename_prefix = filename_prefix[0] if isinstance(filename_prefix, list) else filename_prefix
        
        # Flatten all images and tags to ensure we handle batches and lists correctly
        flat_images = []
        for img in images:
            if len(img.shape) == 4: # (B, H, W, C)
                for b in range(img.shape[0]):
                    flat_images.append(img[b])
            else:
                flat_images.append(img)
                
        flat_tags = []
        for tag in custom_tag:
            if isinstance(tag, list):
                flat_tags.extend(tag)
            else:
                flat_tags.append(tag)
        
        num_images = len(flat_images)
        num_tags = len(flat_tags)
        
        results = list()
        for i in range(num_images):
            img_tensor = flat_images[i]
            # Handle list mapping for tags
            tag_value = flat_tags[i] if i < num_tags else flat_tags[-1]
            
            # Convert tensor (H, W, C) to PIL
            # Ensure we remove any remaining leading 1s if they exist
            if len(img_tensor.shape) == 3 and img_tensor.shape[0] == 1: # (1, H, W) -> shouldn't happen but safe
                img_tensor = img_tensor.squeeze(0)
                
            i_numpy = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i_numpy, 0, 255).astype(np.uint8))
            
            # ComfyUI Filename generation
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, img.width, img.height)
            file = f"{filename}_{counter:05}_.png"
            
            # Prepare Metadata
            metadata = PngInfo()
            metadata.add_text("another_tag", str(tag_value))
            
            # Save
            save_path = os.path.join(full_output_folder, file)
            img.save(save_path, pnginfo=metadata, compress_level=4)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })

        return { "ui": { "images": results } }
