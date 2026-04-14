import torch
import cv2
import numpy as np
import os
import requests
from tqdm import tqdm

class DepthAnythingService:
    """
    Service for monocular depth estimation using Depth-Anything models.
    Adapted for AnotherUtils.
    """
    
    MODELS = {
        "v3-large": "depth-anything/da3-large",
        "v3-medium": "depth-anything/da3-base",
        "v3-small": "depth-anything/da3-small",
        "v3-tiny": "depth-anything/da3-tiny"
    }

    def __init__(self, model_id="v3-small", device="auto"):
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None

    def load_model(self):
        if self.model is not None:
            return

        # Map to correct HF ID
        hf_id = self.MODELS.get(self.model_id, self.model_id)
        if "/" not in hf_id: # Case handling for IDs like "v3-small"
             hf_id = f"depth-anything/{hf_id.replace('v3-', 'da3-')}"

        print(f"[AnotherUtils] Loading Depth-Anything model: {hf_id} on {self.device}")
        
        try:
            if "v3" in self.model_id or "da3" in hf_id:
                from depth_anything_3.api import DepthAnything3
                # Auto-downloads from HF using the correct ID
                self.model = DepthAnything3.from_pretrained(hf_id)
            else:
                # Logic for V2 or older if needed
                raise NotImplementedError("Only Depth-Anything V3 is currently supported in this service.")
            
            self.model = self.model.to(self.device)
            self.model.eval()
        except ImportError:
            raise ImportError("The 'depth_anything_3' library is not installed. Please install it to use this node.")
        except Exception as e:
            raise Exception(f"Failed to load Depth-Anything model: {e}")

    def infer(self, image_np, process_res=1008):
        """
        Inference on a single numpy image (H, W, 3) BGR or RGB.
        Assumes RGB as standard for IA.
        """
        if self.model is None:
            self.load_model()

        # DepthAnything3 expects RGB [H, W, 3]
        prediction = self.model.inference(
            image=[image_np],
            process_res=process_res
        )
        
        depth = prediction.depth
        if len(depth.shape) == 3:
            depth = depth[0]
            
        # Resize back to original
        depth = cv2.resize(depth, (image_np.shape[1], image_np.shape[0]), interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-1
        d_min = depth.min()
        d_max = depth.max()
        if d_max > d_min:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)
            
        return depth
