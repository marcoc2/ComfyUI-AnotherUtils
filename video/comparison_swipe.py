import torch
import numpy as np
import cv2

class ComparisonSwipeNode:
    """
    A custom node that creates a video transition (swipe) between two images.
    Returns a batch of images (frames).
    """
    
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_before": ("IMAGE",),
                "image_after": ("IMAGE",),
                "length": ("INT", {"default": 60, "min": 2, "max": 1000, "label": "Length (frames)"}),
                "resolution_source": (["match_before", "match_after"],),
                "division_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 16.0, "step": 0.1}),
                "line_color": (["white", "black"],),
                "line_width": ("INT", {"default": 4, "min": 1, "max": 50}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    FUNCTION = "generate_swipe"
    CATEGORY = "image/animation"

    def generate_swipe(self, image_before, image_after, length, resolution_source, division_factor, line_color, line_width):
        # Handle inputs: take first frame
        if isinstance(image_before, torch.Tensor):
            img_before = image_before[0].cpu().numpy()
        else:
            img_before = image_before
            
        if isinstance(image_after, torch.Tensor):
            img_after = image_after[0].cpu().numpy()
        else:
            img_after = image_after

        # Pre-process numpy arrays (0-255 uint8)
        if img_before.dtype != np.uint8:
             img_before_u8 = (img_before * 255).astype(np.uint8)
        else:
             img_before_u8 = img_before
             
        if img_after.dtype != np.uint8:
             img_after_u8 = (img_after * 255).astype(np.uint8)
        else:
             img_after_u8 = img_after

        # Ensure consistent channels (RGB)
        # Check if alpha exists and convert to RGB
        if len(img_before_u8.shape) > 2 and img_before_u8.shape[2] == 4:
            img_before_u8 = cv2.cvtColor(img_before_u8, cv2.COLOR_RGBA2RGB)
            
        if len(img_after_u8.shape) > 2 and img_after_u8.shape[2] == 4:
            img_after_u8 = cv2.cvtColor(img_after_u8, cv2.COLOR_RGBA2RGB)

        # Determine target resolution
        h_before, w_before = img_before_u8.shape[:2]
        h_after, w_after = img_after_u8.shape[:2]

        if resolution_source == "match_after":
            target_w, target_h = w_after, h_after
        else:
            target_w, target_h = w_before, h_before

        # Apply division factor
        if division_factor != 1.0:
            target_w = int(target_w / division_factor)
            target_h = int(target_h / division_factor)

        # Ensure valid dimensions
        target_w = max(1, target_w)
        target_h = max(1, target_h)

        # Resize both images to target resolution
        if (w_before, h_before) != (target_w, target_h):
            img_before_u8 = cv2.resize(img_before_u8, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        if (w_after, h_after) != (target_w, target_h):
            img_after_u8 = cv2.resize(img_after_u8, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)

        # Final dimensions used for the loop
        h, w = target_h, target_w

        # Define line color
        if line_color == "white":
            color = (255, 255, 255)
        else:
            color = (0, 0, 0)
        
        frames = []

        # Generate frames
        for i in range(length):
            progress = i / (length - 1)
            cx = int(progress * w)
            
            # Start with 'before' image
            frame = img_before_u8.copy()
            
            # Draw 'after' image on the left side
            if cx > 0:
                frame[:, :cx] = img_after_u8[:, :cx]
            
            # Draw the vertical line
            half_width = line_width // 2
            x1 = max(0, cx - half_width)
            x2 = min(w, cx + half_width)
            
            if x2 > x1:
                cv2.rectangle(frame, (x1, 0), (x2, h), color, -1)
            
            frames.append(frame)

        # Convert back
        frames_np = np.array(frames).astype(np.float32) / 255.0
        frames_tensor = torch.from_numpy(frames_np)
        
        return (frames_tensor,)
