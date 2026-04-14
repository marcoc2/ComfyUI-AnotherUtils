import torch
import numpy as np
import os
import json
from .inference.depth_service import DepthAnythingService
from .inference.detection_service import DetectionService
from .inference.segmentation_service import SAM2Service

class AnotherLoadInferenceModel:
    @classmethod
    def INPUT_TYPES(cls):
        # We look for models in ComfyUI/models/another_utils/
        # and also common paths
        return {
            "required": {
                "model_type": (["YOLO", "SAM2", "DepthAnything"],),
                "model_name": ("STRING", {"default": "yolov8n.pt"}),
            },
            "optional": {
                "device": (["auto", "cuda", "cpu"],),
            }
        }

    RETURN_TYPES = ("ANOTHER_MODEL",)
    FUNCTION = "load_model"
    CATEGORY = "AnotherUtils/inference"

    def load_model(self, model_type, model_name, device="auto"):
        # This node just prepares the config and service
        # Actual loading can be lazy or immediate.
        
        # Determine paths
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        another_models_path = os.path.join(base_path, "another_utils")
        
        if not os.path.exists(another_models_path):
            os.makedirs(another_models_path, exist_ok=True)

        model_path = os.path.join(another_models_path, model_name)
        
        # If not in another_utils, check common comfyui folders
        if not os.path.exists(model_path):
            if model_type == "YOLO":
                model_path = os.path.join(base_path, "ultralytics", model_name)
            elif model_type == "SAM2":
                model_path = os.path.join(base_path, "sam2", model_name)

        model_data = {
            "type": model_type,
            "name": model_name,
            "path": model_path,
            "device": device,
            "service": None
        }

        # Immediate load for validation if possible
        if model_type == "YOLO":
            model_data["service"] = DetectionService(model_path, device=device)
        elif model_type == "DepthAnything":
            model_data["service"] = DepthAnythingService(model_name, device=device)
        elif model_type == "SAM2":
            # SAM2 needs cfg and checkpoint. We assume model_name is the checkpoint.
            # We use base_plus config as default if not specified.
            cfg = "sam2_hiera_b+.yaml" # Default
            if "tiny" in model_name: cfg = "sam2_hiera_t.yaml"
            elif "small" in model_name: cfg = "sam2_hiera_s.yaml"
            elif "large" in model_name: cfg = "sam2_hiera_l.yaml"
            model_data["service"] = SAM2Service(cfg, model_path, device=device)

        return (model_data,)

class AnotherYOLOInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": ("ANOTHER_MODEL",),
                "threshold": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("BBOX", "KEYPOINTS", "STRING", "IMAGE")
    RETURN_NAMES = ("bboxes", "keypoints", "labels", "debug_image")
    FUNCTION = "detect"
    CATEGORY = "AnotherUtils/inference"

    def detect(self, images, model, threshold):
        if model["type"] != "YOLO":
            raise ValueError("Model must be of type YOLO for AnotherYOLOInference")
        
        service: DetectionService = model["service"]
        service.threshold = threshold
        
        all_bboxes = []
        all_kpts = []
        all_labels = []
        
        # Process batch
        for i in range(images.shape[0]):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            detections = service.infer(img_np)
            
            # ComfyUI BBOX format: [[x1, y1, x2, y2], ...]
            # or sometimes dicts. Impact pack uses [x1, y1, x2, y2].
            img_bboxes = [d["bbox"] for d in detections]
            img_kpts = [d.get("pose", []) for d in detections]
            img_labels = [d["label"] for d in detections]
            
            all_bboxes.append(img_bboxes)
            all_kpts.append(img_kpts)
            all_labels.append(img_labels)

        # Return format: BBOX and KEYPOINTS are usually lists of lists in ComfyUI
        # Since we might have multiple images, we return the first one's detection for simplicity 
        # OR we return the batch if the receiver supports it.
        # Most BBOX nodes expect a single list for the current image.
        
        # Flattening or returning batch is tricky in ComfyUI.
        # For now, let's return the full list and hope the next node handles iterables.
        return (all_bboxes, all_kpts, json.dumps(all_labels), images)

class AnotherSAM2Inference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "model": ("ANOTHER_MODEL",),
            },
            "optional": {
                "points_pos": ("STRING", {"default": "[]"}),
                "points_neg": ("STRING", {"default": "[]"}),
                "bboxes": ("BBOX",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "segment"
    CATEGORY = "AnotherUtils/inference"

    def segment(self, image, model, points_pos="[]", points_neg="[]", bboxes=None):
        if model["type"] != "SAM2":
            raise ValueError("Model must be of type SAM2 for AnotherSAM2Inference")
            
        service: SAM2Service = model["service"]
        
        pos = json.loads(points_pos) if isinstance(points_pos, str) else points_pos
        neg = json.loads(points_neg) if isinstance(points_neg, str) else points_neg
        
        # Convert JSON structure to simple list of coords
        p_coords = []
        p_labels = []
        
        for p in pos:
            p_coords.append([p["x"], p["y"]])
            p_labels.append(1)
        for p in neg:
            p_coords.append([p["x"], p["y"]])
            p_labels.append(0)
            
        all_masks = []
        for i in range(image.shape[0]):
            img_np = (image[i].cpu().numpy() * 255).astype(np.uint8)
            service.set_image(img_np)
            
            # If bboxes is provided, use them. If it's a batch of bboxes, take the correct index.
            current_bboxes = None
            if bboxes is not None:
                if isinstance(bboxes[0], list) and len(bboxes) == image.shape[0]:
                    current_bboxes = bboxes[i]
                else:
                    current_bboxes = bboxes
            
            mask = service.predict(
                points=p_coords if p_coords else None,
                labels=p_labels if p_labels else None,
                bboxes=current_bboxes if current_bboxes else None
            )
            # mask is [N, H, W] bool. We merge all N detections into one mask.
            if mask.shape[0] > 0:
                merged_mask = np.any(mask, axis=0)
            else:
                merged_mask = np.zeros((image.shape[1], image.shape[2]), dtype=bool)
                
            all_masks.append(torch.from_numpy(merged_mask.astype(np.float32)))

        return (torch.stack(all_masks),)

class AnotherDepthInference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "model": ("ANOTHER_MODEL",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    FUNCTION = "infer"
    CATEGORY = "AnotherUtils/inference"

    def infer(self, images, model):
        if model["type"] != "DepthAnything":
            raise ValueError("Model must be of type DepthAnything for AnotherDepthInference")
            
        service: DepthAnythingService = model["service"]
        
        all_depths = []
        for i in range(images.shape[0]):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            depth = service.infer(img_np)
            all_depths.append(torch.from_numpy(depth))

        depth_batch = torch.stack(all_depths)
        # depth_batch is [B, H, W]. Return as image [B, H, W, 1] and mask [B, H, W]
        return (depth_batch.unsqueeze(-1).repeat(1, 1, 1, 3), depth_batch)

class AnotherBBoxToPoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "bboxes": ("BBOX",),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("points_json",)
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/utils"

    def convert(self, bboxes):
        # bboxes: [[x1, y1, x2, y2], ...]
        points = []
        # If it's a batch of bboxes, we might need to handle it. 
        # For simplicity, if it's nested [[...]], we take the first image's boxes.
        target_boxes = bboxes
        if bboxes and isinstance(bboxes[0], list) and len(bboxes[0]) > 0 and isinstance(bboxes[0][0], (int, float, list)):
             target_boxes = bboxes[0]

        for box in target_boxes:
            x1, y1, x2, y2 = box
            cx = x1 + (x2 - x1) / 2
            cy = y1 + (y2 - y1) / 2
            points.append({"x": cx, "y": cy})
        
        return (json.dumps(points),)

class AnotherPoseToPoints:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("KEYPOINTS",),
                "point_index": ("INT", {"default": 0, "min": 0, "max": 16, "step": 1}),
                "filter": ("STRING", {"default": "nose, eyes"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("points_json",)
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/utils"

    def convert(self, keypoints, point_index, filter):
        # keypoints: List of [N, 17, 2] (or [17, 2] per detection)
        # We'll support filtering by index for now as it's more precise
        
        # 0: nose, 1: l_eye, 2: r_eye, 5: l_sh, 6: r_sh...
        COCO_MAP = {
            "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
            "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
        }

        points = []
        
        # Handle batching
        target_kpts = keypoints
        if keypoints and isinstance(keypoints[0], list):
            target_kpts = keypoints[0] # Take first image detections

        selected_indices = []
        if filter.strip():
            for part in filter.split(","):
                part = part.strip().lower().replace(" ", "_")
                if part in COCO_MAP:
                    selected_indices.append(COCO_MAP[part])
        
        if not selected_indices:
            selected_indices = [point_index]

        for det_kpts in target_kpts:
            # det_kpts is [17, 2]
            for idx in selected_indices:
                if idx < len(det_kpts):
                    p = det_kpts[idx]
                    points.append({"x": float(p[0]), "y": float(p[1])})
        
        return (json.dumps(points),)

class AnotherImageToMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "channel": (["red", "green", "blue", "alpha", "luminance"], {"default": "luminance"}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/utils"

    def convert(self, image, channel):
        if channel == "luminance":
            mask = image.mean(dim=-1)
        elif channel == "red":
            mask = image[:, :, :, 0]
        elif channel == "green":
            mask = image[:, :, :, 1]
        elif channel == "blue":
            mask = image[:, :, :, 2]
        elif channel == "alpha":
            if image.shape[-1] == 4:
                mask = image[:, :, :, 3]
            else:
                mask = torch.ones((image.shape[0], image.shape[1], image.shape[2]), device=image.device)
        return (mask,)

class AnotherMaskToImage:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/utils"

    def convert(self, mask):
        # mask is [B, H, W]
        if mask.ndim == 2:
            mask = mask.unsqueeze(0)
        return (mask.unsqueeze(-1).repeat(1, 1, 1, 3),)

class AnotherMaskMath:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask_a": ("MASK",),
                "operation": (["multiply", "add", "subtract", "and", "or", "xor"], {"default": "multiply"}),
            },
            "optional": {
                "mask_b": ("MASK",),
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "apply"
    CATEGORY = "AnotherUtils/utils"

    def apply(self, mask_a, operation, mask_b=None, value=1.0):
        # Ensure mask_a is tensor
        res = mask_a.clone()
        
        # If mask_b is not provided, use the scalar value
        other = mask_b if mask_b is not None else value
        
        if operation == "multiply":
            res = res * other
        elif operation == "add":
            res = res + other
        elif operation == "subtract":
            res = res - other
        elif operation == "and":
            res = torch.min(res, torch.tensor(other))
        elif operation == "or":
            res = torch.max(res, torch.tensor(other))
        elif operation == "xor":
            res = torch.abs(res - other)
            
        return (torch.clamp(res, 0.0, 1.0),)

class AnotherMaskBlur:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
                "blur_radius": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                "sigma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "blur"
    CATEGORY = "AnotherUtils/utils"

    def blur(self, mask, blur_radius, sigma):
        if blur_radius == 0:
            return (mask,)
            
        import scipy.ndimage
        # Convert to numpy for scipy
        mask_np = mask.cpu().numpy()
        blurred_np = np.zeros_like(mask_np)
        
        for i in range(mask_np.shape[0]):
            blurred_np[i] = scipy.ndimage.gaussian_filter(mask_np[i], sigma=sigma, radius=blur_radius)
            
        return (torch.from_numpy(blurred_np).to(mask.device),)
