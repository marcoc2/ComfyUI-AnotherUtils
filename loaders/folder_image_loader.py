import os
import torch
import numpy as np
from PIL import Image, ImageOps

class FolderImageLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": ""}),
                "target_width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1, "tooltip": "0 = use first image width"}),
                "target_height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1, "tooltip": "0 = use first image height"}),
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "0 = unlimited"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "Skip first N images"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "filenames")
    OUTPUT_IS_LIST = (True, True, True)
    FUNCTION = "load_images"
    CATEGORY = "AnotherUtils"

    def load_images(self, directory, target_width, target_height, image_load_cap, start_index):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp')
        files = [f for f in os.listdir(directory) if f.lower().endswith(valid_ext)]
        files.sort()
        
        # Apply start index
        if start_index > 0:
            files = files[start_index:]

        # Apply Cap
        if image_load_cap > 0:
            files = files[:image_load_cap]

        if not files:
            raise FileNotFoundError(f"No valid images found in: {directory}")

        images = []
        masks = []
        filenames = []
        
        # Determine strict target size
        if target_width == 0 or target_height == 0:
             # Fallback to first image size for 0 dims
             try:
                 temp = Image.open(os.path.join(directory, files[0]))
                 if target_width == 0: target_width = temp.size[0]
                 if target_height == 0: target_height = temp.size[1]
                 temp.close()
             except:
                 target_width = 512
                 target_height = 512
        
        final_size = (target_width, target_height)
        print(f"[FolderImageLoader] Loading {len(files)} images. Canvas Size: {final_size}")

        for f in files:
            path = os.path.join(directory, f)
            try:
                img = Image.open(path)
                
                # Handle Animation
                if getattr(img, 'is_animated', False):
                    img.seek(0)
                
                img = img.convert("RGBA")
                
                # Logic: Canvas Center
                # Create uniform canvas
                new_img = Image.new("RGBA", final_size, (0, 0, 0, 0))
                
                # Calculate center position
                paste_x = (final_size[0] - img.size[0]) // 2
                paste_y = (final_size[1] - img.size[1]) // 2
                
                new_img.alpha_composite(img, (paste_x, paste_y))
                
                img = new_img
                
                # Process Image
                # Normalize to 0-1 float
                # (H, W, 4)
                i = np.array(img).astype(np.float32) / 255.0
                
                mask = i[:, :, 3] 
                image_rgb = i[:, :, :3]
                
                # Add [1, ...] dimension for ComfyUI Image format if returning list
                # Each item in list must be [1, H, W, C]
                images.append(torch.from_numpy(image_rgb).unsqueeze(0))
                masks.append(torch.from_numpy(mask)) # Shape [H, W] for ComfyUI compatibility
                # Each item in list must be [1, H, W, C] for Image
                # If output is list, we expect list of [H,W].
                # Wait, PreviewImage expects Mask [H,W].
                # Let's revert mask to just torch.from_numpy(mask) without unsqueeze if standard nodes expect [H,W].
                # But VAE Encode (for inpainting) might want [1,H,W]. 
                # Usually standard LoadImage returns MASK as [H, W].
                # Let's check: LoadImage -> mask -> torch.from_numpy(mask) -> [H,W].
                # So we keep it [H,W] per item.
                # Re-reading: masks.append(torch.from_numpy(mask))
                
                filenames.append(f)

            except Exception as e:
                print(f"[FolderImageLoader] Error loading {f}: {e}")

        if not images:
             raise ValueError("Failed to process any images.")
        
        # Don't stack. Return lists.
        # But ensure Mask is correct shape. 
        # Actually... if output is list, Comfy executes node N times. 
        # In each execution, it passes ONE item.
        # Use standard shapes: Image [1, H, W, C], Mask [H,W].

        print(f"[FolderImageLoader] Created list of {len(images)} images")
        return (images, masks, filenames)
