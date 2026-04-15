import json
import torch
import numpy as np
from PIL import Image, ImageOps

class PromptExtractor:
    @staticmethod
    def extract_from_image(img):
        """
        Extracts the prompt string from PIL Image metadata.
        Returns a string.
        """
        metadata = img.info
        if not metadata:
            return ""
        
        # 1. ComfyUI Standard
        if 'prompt' in metadata:
            try:
                prompt_json = json.loads(metadata['prompt'])
                return PromptExtractor.extract_from_json(prompt_json)
            except Exception as e:
                print(f"[AnotherUtils] Error parsing prompt JSON: {e}")
        
        # 2. A1111 / WebUI Standard
        if 'parameters' in metadata:
            return metadata['parameters'].split("\n")[0]
            
        return ""

    @staticmethod
    def extract_from_json(prompt_json):
        """
        Heuristic to find the 'main' positive prompt in a ComfyUI graph.
        """
        candidates = []

        # Find all sampler-like nodes
        sampler_types = ['Sampler', 'Guider', 'SamplerCustom', 'KSampler', 'Flux2Sampler']
        sampler_nodes = {id: node for id, node in prompt_json.items() 
                         if any(st in node.get('class_type', '') for st in sampler_types)}

        for sid, snode in sampler_nodes.items():
            # Follow 'positive' link
            inputs = snode.get('inputs', {})
            pos = inputs.get('positive') or inputs.get('cond') or inputs.get('conditioning')
            if isinstance(pos, list):
                text = PromptExtractor._trace_back(prompt_json, str(pos[0]))
                if text:
                    # Give higher score if it's connected to a sampler
                    candidates.append((text, 10))

        # Fallback: Look for all CLIPTextEncode nodes
        for id, node in prompt_json.items():
            ctype = node.get('class_type', '')
            if 'CLIPTextEncode' in ctype:
                inputs = node.get('inputs', {})
                # Handle SDXL (text_g, text_l)
                text = inputs.get('text') or inputs.get('text_g') or inputs.get('text_l')
                if text and isinstance(text, str):
                    score = 5
                    title = node.get('_meta', {}).get('title', '').lower()
                    if 'positive' in title:
                        score += 5
                    candidates.append((text, score))

        if not candidates:
            # Last ditch: look for any string longer than 20 chars in inputs
            for id, node in prompt_json.items():
                for val in node.get('inputs', {}).values():
                    if isinstance(val, str) and len(val) > 20:
                        candidates.append((val, 1))

        if candidates:
            # Sort by score (desc), then by length (desc)
            candidates.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
            return candidates[0][0]

        return "Prompt not found"

    @staticmethod
    def _trace_back(prompt_json, node_id, visited=None):
        if visited is None:
            visited = set()
        if node_id in visited or node_id not in prompt_json:
            return None
        visited.add(node_id)

        node = prompt_json[node_id]
        inputs = node.get('inputs', {})

        # Direct text hit
        text = inputs.get('text') or inputs.get('text_g') or inputs.get('text_l')
        if text and isinstance(text, str) and len(text.strip()) > 0:
            return text

        # Recursive trace for conditioning links
        for key in ['conditioning', 'cond', 'text', 'string', 'samples']:
            val = inputs.get(key)
            if isinstance(val, list):
                result = PromptExtractor._trace_back(prompt_json, str(val[0]), visited)
                if result:
                    return result
        
        return None

    @staticmethod
    def preprocess_image(img):
        """Shared image to tensor logic"""
        img = ImageOps.exif_transpose(img)
        image = img.convert("RGB")
        image = np.array(image).astype(np.float32) / 255.0
        return torch.from_numpy(image)[None,]
