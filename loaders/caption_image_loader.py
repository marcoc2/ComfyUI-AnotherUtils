import os

class CaptionImageLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "Path to root (imgs + captions subfolder)"}),
            },
            "optional": {
                "selected_basename": ("STRING", {"default": "", "forceInput": False}), 
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("caption",)
    OUTPUT_IS_LIST = (False,)
    FUNCTION = "get_caption"
    CATEGORY = "AnotherUtils/loaders"

    def get_caption(self, directory: str, selected_basename: str = ""):
        print(f"[CaptionImageLoader] Executing with directory='{directory}', selected_basename='{selected_basename}'")
        
        if not directory:
             raise ValueError("Directory not specified")
        
        if not selected_basename:
            print("[CaptionImageLoader] No file selected via UI. Returning empty string.")
            return ("",)

        cap_dir = os.path.join(directory, "captions")
        cap_path = os.path.join(cap_dir, selected_basename + ".txt")
        print(f"[CaptionImageLoader] Loading caption from: {cap_path}")

        if not os.path.exists(cap_path):
             print(f"[CaptionImageLoader] Caption file not found: {cap_path}")
             return ("",)

        try:
            with open(cap_path, 'r', encoding='utf-8') as f:
                caption_text = f.read().strip()
            print(f"[CaptionImageLoader] Loaded caption: {caption_text[:50]}...")
            return (caption_text,)
        except Exception as e:
            print(f"[CaptionImageLoader] Error reading caption: {e}")
            return ("",)
