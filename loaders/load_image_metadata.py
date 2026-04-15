import os
import torch
from PIL import Image
import folder_paths
import hashlib
from .utils import PromptExtractor

class LoadImageAndExtractPrompt:
    @classmethod
    def INPUT_TYPES(s):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
        return {
            "required": {
                "image": (sorted(files), {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "prompt")
    FUNCTION = "load_image"
    CATEGORY = "AnotherUtils/loaders"

    def load_image(self, image):
        image_path = folder_paths.get_annotated_filepath(image)
        img = Image.open(image_path)
        
        # 1. Image processing
        image_tensor = PromptExtractor.preprocess_image(img)
        
        # 2. Mask processing
        if 'A' in img.getbands():
            import numpy as np
            mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask = 1.0 - torch.from_numpy(mask)
        else:
            mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")

        # 3. Extract Prompt using shared utility
        prompt_text = PromptExtractor.extract_from_image(img)

        return (image_tensor, mask, prompt_text)

    @classmethod
    def IS_CHANGED(s, image):
        image_path = folder_paths.get_annotated_filepath(image)
        m = hashlib.sha256()
        with open(image_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, image):
        if not folder_paths.exists_annotated_filepath(image):
            return "Invalid image file: {}".format(image)
        return True
