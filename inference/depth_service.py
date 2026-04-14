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
        "v2-tiny": "https://huggingface.co/depth-anything/Depth-Anything-V2-Tiny/resolve/main/depth_anything_v2_vitl.pth", # Placeholder URL
        "v3-large": "depth-anything/Depth-Anything-V3-Large" # HuggingFace ID
    }

    def __init__(self, model_id="v3-large", device="auto"):
        self.model_id = model_id
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        self.model = None

    def load_model(self):
        if self.model is not None:
            return

        print(f"[AnotherUtils] Loading Depth-Anything model: {self.model_id} on {self.device}")
        
        try:
            if "v3" in self.model_id:
                from depth_anything_3.api import DepthAnything3
                # Auto-downloads from HF
                self.model = DepthAnything3.from_pretrained("depth-anything/Depth-Anything-V3-Large")
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
