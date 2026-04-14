import numpy as np
import json

class SEGStoBBox:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"segs": ("SEGS",),}}

    RETURN_TYPES = ("BBOX",)
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/SEGS"

    def convert(self, segs):
        """
        Converts Impact Pack SEGS to SAM 2 BBOX format.
        SAM 2 expects a list of lists of bounding boxes: [ [ [x1,y1,x2,y2], ... ] ]
        """
        if not segs or len(segs) < 2:
            return ([[]],)
            
        bboxes = []
        for seg in segs[1]:
            # seg.bbox is (x1, y1, x2, y2)
            bboxes.append(list(seg.bbox))
        
        # Return as a batch of boxes (standard for single image context)
        return ([bboxes],)

class SEGStoSAM2Points:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"segs": ("SEGS",),}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("json_points",)
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/SEGS"

    def convert(self, segs):
        """
        Converts Impact Pack SEGS to SAM 2 JSON point format.
        Useful for 'coordinates_positive' input in SAM 2 Video nodes.
        """
        if not segs or len(segs) < 2 or not segs[1]:
            print("!!! [AnotherUtils] YOLO found no objects. SAM2 tracking might fail.")
            return ("[]",)
            
        points = []
        for seg in segs[1]:
            x1, y1, x2, y2 = seg.bbox
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            points.append({"x": center_x, "y": center_y})
        
        return (json.dumps(points),)

class GetFirstFrame:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"images": ("IMAGE",),}}

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "get"
    CATEGORY = "AnotherUtils/image"

    def get(self, images):
        return (images[0:1],)

class ManualPointToSAM2:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "x": ("INT", {"default": 333, "min": 0, "max": 8192}),
                "y": ("INT", {"default": 333, "min": 0, "max": 8192}),
                "num_points": ("INT", {"default": 1, "min": 1, "max": 100}),
                "radius": ("INT", {"default": 0, "min": 0, "max": 500}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/image"

    def convert(self, x, y, num_points, radius, seed):
        import json
        import random
        import math
        
        random.seed(seed)
        points = []
        
        if num_points <= 1 or radius <= 0:
            points.append({"x": x, "y": y})
        else:
            for _ in range(num_points):
                # Use sqrt(r) to get uniform distribution in a circle
                r = radius * math.sqrt(random.random())
                theta = random.random() * 2 * math.pi
                px = int(x + r * math.cos(theta))
                py = int(y + r * math.sin(theta))
                points.append({"x": px, "y": py})
        
        return (json.dumps(points),)

class RefineMask:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "expand": ("INT", {"default": 0, "min": -100, "max": 100, "step": 1}),
                "blur": ("INT", {"default": 0, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "refine"
    CATEGORY = "AnotherUtils/mask"

    def refine(self, mask, expand, blur):
        import scipy.ndimage
        import numpy as np
        import torch

        # mask shape is [B, H, W] or [H, W]
        mask_np = mask.cpu().numpy()
        
        # Dilate/Erode
        if expand != 0:
            if expand > 0:
                mask_np = scipy.ndimage.binary_dilation(mask_np, iterations=expand)
            else:
                mask_np = scipy.ndimage.binary_erosion(mask_np, iterations=abs(expand))
        
        mask_np = mask_np.astype(np.float32)

        # Blur
        if blur > 0:
            for i in range(mask_np.shape[0] if len(mask_np.shape) == 3 else 1):
                m = mask_np[i] if len(mask_np.shape) == 3 else mask_np
                m = scipy.ndimage.gaussian_filter(m, sigma=blur)
                if len(mask_np.shape) == 3: mask_np[i] = m
                else: mask_np = m

        return (torch.from_numpy(mask_np),)
