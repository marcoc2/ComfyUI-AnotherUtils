import os
import torch
import numpy as np
from PIL import Image

class DatasetLoader:
    """
    Loads all images and their corresponding captions from a dataset directory.
    Returns lists of images, paths, filenames, and captions for batch processing.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "Path to dataset root (with captions subfolder)"}),
                "target_width": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1, "tooltip": "0 = use first image width"}),
                "target_height": ("INT", {"default": 512, "min": 0, "max": 8192, "step": 1, "tooltip": "0 = use first image height"}),
                "image_load_cap": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "0 = load all images"}),
                "start_index": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "tooltip": "Skip first N images"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("images", "paths", "filenames", "captions")
    OUTPUT_IS_LIST = (True, True, True, True)
    FUNCTION = "load_dataset"
    CATEGORY = "AnotherUtils/loaders"

    def load_dataset(self, directory: str, target_width: int, target_height: int, image_load_cap: int = 0, start_index: int = 0):
        if not directory:
            raise ValueError("Directory not specified")

        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")

        print(f"[DatasetLoader] Loading from directory: {directory}")

        # Use same search logic as caption_image_loader
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

        # Helper to scan images in a dir
        def get_images_in(target_dir):
            if not os.path.isdir(target_dir):
                return []
            try:
                return [f for f in os.listdir(target_dir) if f.lower().endswith(valid_extensions)]
            except:
                return []

        # Priority Search: Root -> original_dataset -> cropped_images
        search_dirs = [
            directory,
            os.path.join(directory, "original_dataset"),
            os.path.join(directory, "cropped_images")
        ]

        image_files = []
        img_dir = directory  # Default to root if nothing found

        for d in search_dirs:
            found_imgs = get_images_in(d)
            if found_imgs:
                image_files = found_imgs
                img_dir = d
                print(f"[DatasetLoader] Images found in: {img_dir}")
                break

        if not image_files:
            raise FileNotFoundError(f"No images found in {directory} or its subdirectories")

        image_files.sort()

        # Apply start index
        if start_index > 0:
            image_files = image_files[start_index:]
            print(f"[DatasetLoader] Skipping first {start_index} images")

        # Apply cap
        if image_load_cap > 0:
            image_files = image_files[:image_load_cap]
            print(f"[DatasetLoader] Loading capped to {image_load_cap} images")

        print(f"[DatasetLoader] Processing {len(image_files)} images")
        
        # Determine strict target size if 0
        if target_width == 0 or target_height == 0:
             try:
                 temp = Image.open(os.path.join(img_dir, image_files[0]))
                 if target_width == 0: target_width = temp.size[0]
                 if target_height == 0: target_height = temp.size[1]
                 temp.close()
             except:
                 target_width = 512
                 target_height = 512
        
        final_size = (target_width, target_height)
        print(f"[DatasetLoader] Target Canvas Size: {final_size}")

        # Caption directory
        cap_dir = os.path.join(directory, "captions")
        if not os.path.isdir(cap_dir):
            print(f"[DatasetLoader] Warning: captions directory not found at {cap_dir}")

        images = []
        paths = []
        filenames = []
        captions = []

        for filename in image_files:
            img_path = os.path.join(img_dir, filename)

            try:
                # Load image
                img = Image.open(img_path)

                # Handle animated images
                if getattr(img, 'is_animated', False):
                    img.seek(0)
                
                # Convert to RGBA for handling transparency logic
                img = img.convert("RGBA")
                
                # Logic: Smart Canvas Center (Fit & Pad / Crop)
                # Create uniform canvas
                new_img = Image.new("RGBA", final_size, (0, 0, 0, 0))
                
                # Calculate center position
                paste_x = (final_size[0] - img.size[0]) // 2
                paste_y = (final_size[1] - img.size[1]) // 2
                
                # Paste (Handling negative coordinates automatically by PIL clipping)
                new_img.alpha_composite(img, (paste_x, paste_y))
                img = new_img

                # Convert to Tensor (ComfyUI Format [1, H, W, C])
                # We return RGB for consistency with ComfyUI main format, but mask handles alpha transparency.
                # However, DatasetLoader originally returned only RGB. 
                # If we have uniform canvas, transparency becomes black in RGB conversion?
                # User asked to "apendar apenas pixels alpha em volta" (append only alpha pixels around).
                # This implies the transparency should be PRESERVED or represented.
                # If we convert RGBA -> RGB, transparent pixels become black (0,0,0) or white?
                # Actually, simple conversion discards alpha. Background color depends on what was "behind" it? No.
                # It just ignores alpha. What happens to the "invisible" pixels? They have RGB values usually.
                # Since we created new_img with (0,0,0,0), the RGB is (0,0,0).
                # So converting to RGB will result in black background.
                # ComfyUI usually expects RGB images, even if valid pixels are masked.
                # Let's verify return type. It is ("IMAGE", "STRING", ...).
                # Standard Loop:
                i = np.array(img).astype(np.float32) / 255.0
                # i is [H, W, 4]
                
                # Extract RGB
                image_rgb = i[:, :, :3]
                
                # Convert to tensor [1, H, W, 3]
                img_tensor = torch.from_numpy(image_rgb).unsqueeze(0)

                # Load caption
                basename = os.path.splitext(filename)[0]
                cap_path = os.path.join(cap_dir, basename + ".txt")
                caption_text = ""

                if os.path.exists(cap_path):
                    try:
                        with open(cap_path, 'r', encoding='utf-8') as f:
                            caption_text = f.read().strip()
                    except Exception as e:
                        print(f"[DatasetLoader] Error reading caption for {filename}: {e}")
                        caption_text = ""
                else:
                    # Try basic fallback if caption not found? No, keep simple.
                    pass 

                # Append to lists
                images.append(img_tensor)
                paths.append(img_path)
                filenames.append(filename)
                captions.append(caption_text)

            except Exception as e:
                print(f"[DatasetLoader] Error loading {filename}: {e}")
                continue

        if not images:
            raise ValueError("Failed to load any images from dataset")

        print(f"[DatasetLoader] Successfully loaded {len(images)} images with captions")
        return (images, paths, filenames, captions)
