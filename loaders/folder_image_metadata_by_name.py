import os
import hashlib
import json
from PIL import Image
from .utils import PromptExtractor

class FolderImageMetadataByName:
    """
    Loads images from a folder and extracts text based on a specific node name/title
    from the ComfyUI metadata, ignoring the score heuristics.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "directory": ("STRING", {"default": "", "placeholder": "Path to directory of images"}),
                "node_name": ("STRING", {"default": "CLIPTextEncode", "placeholder": "Node title or class type"}),
            },
        }
    
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "extracted_texts")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "load_images"
    CATEGORY = "AnotherUtils/loaders"

    def extract_from_node_name(self, img, node_name):
        metadata = img.info
        if not metadata:
            return ""
        
        if 'prompt' in metadata:
            try:
                prompt_json = json.loads(metadata['prompt'])
                for id, node in prompt_json.items():
                    title = node.get('_meta', {}).get('title', '')
                    class_type = node.get('class_type', '')
                    
                    if node_name.lower().strip() == title.lower().strip() or node_name.lower().strip() == class_type.lower().strip():
                        inputs = node.get('inputs', {})
                        # Try known text fields first
                        for key in ['text', 'text_g', 'string', 'text_l', 'caption']:
                            if key in inputs and isinstance(inputs[key], str):
                                return inputs[key]
                        # Fallback to any string input in the node
                        for val in inputs.values():
                            if isinstance(val, str) and len(val.strip()) > 0:
                                return val
                                
            except Exception as e:
                print(f"[AnotherUtils] Error parsing prompt JSON: {e}")
                
        # A1111 / WebUI Fallback just in case
        if 'parameters' in metadata:
            return metadata['parameters'].split("\n")[0]
            
        return "Prompt not found"

    def load_images(self, directory: str, node_name: str):
        if not os.path.isdir(directory):
            raise FileNotFoundError(f"Directory '{directory}' cannot be found.")
        
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        image_files = [f for f in os.listdir(directory) 
                      if f.lower().endswith(valid_extensions)]
        
        if len(image_files) == 0:
            raise FileNotFoundError(f"No valid image files found in directory '{directory}'.")
        
        image_files.sort()
        
        images_list = []
        texts_list = []

        for filename in image_files:
            filepath = os.path.join(directory, filename)
            try:
                img = Image.open(filepath)
                
                # Image processing
                image_tensor = PromptExtractor.preprocess_image(img)
                images_list.append(image_tensor)
                
                # Text extraction by exact node name
                extracted = self.extract_from_node_name(img, node_name)
                texts_list.append(extracted)
                
            except Exception as e:
                print(f"[AnotherUtils] Error loading image {filename}: {e}")
        
        return (images_list, texts_list)
    
    @classmethod
    def IS_CHANGED(s, directory: str, node_name: str):
        if not os.path.isdir(directory):
            return False
            
        m = hashlib.sha256()
        valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
        for filename in sorted(os.listdir(directory)):
            if filename.lower().endswith(valid_extensions):
                filepath = os.path.join(directory, filename)
                m.update(str(os.path.getmtime(filepath)).encode())
        m.update(node_name.encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(s, directory: str, node_name: str):
        if not directory.strip():
            return "Directory path cannot be empty"
            
        if not os.path.isdir(directory):
            return f"Directory '{directory}' cannot be found."
        return True
