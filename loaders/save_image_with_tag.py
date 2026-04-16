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
        
        # Determine number of items to process
        num_images = len(images)
        num_tags = len(custom_tag)
        
        results = list()
        for i in range(num_images):
            img_tensor = images[i]
            # Handle list mapping for tags
            tag_value = custom_tag[i] if i < num_tags else custom_tag[-1]
            
            # Convert tensor to PIL
            i_numpy = 255. * img_tensor.cpu().numpy()
            img = Image.fromarray(np.clip(i_numpy, 0, 255).astype(np.uint8))
            
            # ComfyUI Filename generation
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, self.output_dir, img.width, img.height)
            file = f"{filename}_{counter:05}_.png"
            
            # Prepare Metadata
            metadata = PngInfo()
            # Standard ComfyUI stuff (not strictly required if we only want our tag, 
            # but usually good for compatibility)
            # However, we ONLY care about the custom tag for recovery.
            metadata.add_text("another_tag", str(tag_value))
            
            # Save
            save_path = os.path.join(full_output_folder, file)
            img.save(save_path, pnginfo=metadata, compress_level=4)
            
            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            })
            counter += 1

        return { "ui": { "images": results } }
