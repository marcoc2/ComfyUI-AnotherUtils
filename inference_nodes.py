import torch
import numpy as np
import os
import json
from .inference.depth_service import DepthAnythingService
from .inference.detection_service import DetectionService
from .inference.segmentation_service import SAM2Service, SAM2VideoService

def filter_models(extension_list, subfolder):
    base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    search_path = os.path.join(base_path, subfolder)
    another_path = os.path.join(base_path, "another_utils")
    
    found = []
    for path in [search_path, another_path]:
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for file in files:
                    if any(file.endswith(ext) for ext in extension_list):
                        found.append(file)
    return sorted(list(set(found)))

class AnotherLoadYOLO:
    @classmethod
    def INPUT_TYPES(cls):
        yolo_sizes = ["n", "s", "m", "l", "x"]
        yolo_tasks = ["", "-pose", "-seg"]
        yolo_versions = ["yolov8", "yolo11"]
        presets = [f"{v}{s}{t}.pt" for v in yolo_versions for t in yolo_tasks for s in yolo_sizes]
        local = filter_models([".pt", ".pth", ".safetensors"], "ultralytics")
        models = sorted(list(set(presets + local)))
        return {
            "required": {
                "model_name": (models, {"default": "yolov8m.pt"}),
                "device": (["auto", "cuda", "cpu"],),
            }
        }
    RETURN_TYPES = ("ANOTHER_MODEL",)
    FUNCTION = "load"
    CATEGORY = "AnotherUtils/inference"
    def load(self, model_name, device="auto"):
        return (AnotherLoadInferenceModel().load_model("YOLO", model_name, device),)

class AnotherLoadSAM2:
    @classmethod
    def INPUT_TYPES(cls):
        presets = ["sam2_hiera_tiny.pt", "sam2_hiera_small.pt", "sam2_hiera_base_plus.pt", "sam2_hiera_large.pt"]
        local = filter_models([".pt", ".pth", ".safetensors"], "sam2")
        models = sorted(list(set(presets + local)))
        return {
            "required": {
                "model_name": (models, {"default": "sam2_hiera_small.pt"}),
                "mode": (["single_image", "video"], {"default": "single_image"}),
                "device": (["auto", "cuda", "cpu"],),
            }
        }
    RETURN_TYPES = ("ANOTHER_MODEL",)
    FUNCTION = "load"
    CATEGORY = "AnotherUtils/inference"
    def load(self, model_name, mode="single_image", device="auto"):
        model_type = "SAM2" if mode == "single_image" else "SAM2_VIDEO"
        return (AnotherLoadInferenceModel().load_model(model_type, model_name, device),)

class AnotherLoadDepth:
    @classmethod
    def INPUT_TYPES(cls):
        models = ["v3-tiny", "v3-small", "v3-medium", "v3-large"]
        return {
            "required": {
                "model_name": (models, {"default": "v3-small"}),
                "device": (["auto", "cuda", "cpu"],),
            }
        }
    RETURN_TYPES = ("ANOTHER_MODEL",)
    FUNCTION = "load"
    CATEGORY = "AnotherUtils/inference"
    def load(self, model_name, device="auto"):
        return (AnotherLoadInferenceModel().load_model("DepthAnything", model_name, device),)

class AnotherLoadInferenceModel:
    # Keeps this internal for backward compatibility of logic 
    # but we remove it from NODE_CLASS_MAPPINGS later
    def load_model(self, model_type, model_name, device="auto"):
        # Determine paths
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
        
        # Search for the model file
        model_path = model_name 
        
        if "." in model_name:
            found = False
            # Check standard subfolders
            for sub in ["sam2", "ultralytics", "another_utils", "ultralytics/bbox", "ultralytics/segm"]:
                check_path = os.path.join(base_path, sub, model_name)
                if os.path.exists(check_path):
                    model_path = check_path
                    found = True
                    break
            
            if not found:
                # If not found, we set the target path but the service will handle the download
                subfolder = "sam2" if model_type == "SAM2" else "ultralytics"
                model_path = os.path.join(base_path, subfolder, model_name)

        model_data = {
            "type": model_type,
            "name": model_name,
            "path": model_path,
            "device": device,
            "service": None
        }

        # Initialize the service (will trigger download if missing)
        if model_type == "YOLO":
            model_data["service"] = DetectionService(model_path, device=device)
        elif model_type == "DepthAnything":
            model_data["service"] = DepthAnythingService(model_name, device=device)
        elif model_type == "SAM2":
            model_data["service"] = SAM2Service(None, model_path, device=device)
        elif model_type == "SAM2_VIDEO":
            model_data["service"] = SAM2VideoService(None, model_path, device=device)

        return model_data

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

    RETURN_TYPES = ("BBOX", "KEYPOINTS", "MASK", "STRING", "IMAGE")
    RETURN_NAMES = ("bboxes", "keypoints", "mask", "labels", "debug_image")
    FUNCTION = "detect"
    CATEGORY = "AnotherUtils/inference"

    def detect(self, images, model, threshold):
        if model["type"] != "YOLO":
            raise ValueError("Model must be of type YOLO for AnotherYOLOInference")
        
        service: DetectionService = model["service"]
        service.threshold = threshold
        
        all_bboxes = []
        all_kpts = []
        all_masks = []
        all_labels = []
        debug_images = []
        
        # Process batch
        for i in range(images.shape[0]):
            img_np = (images[i].cpu().numpy() * 255).astype(np.uint8)
            detections = service.infer(img_np, conf=threshold)
            
            img_bboxes = [d["bbox"] for d in detections]
            img_kpts = [d.get("pose", []) for d in detections]
            img_labels = [d["label"] for d in detections]
            
            # Merge masks for this image
            current_img_masks = [d["mask"] for d in detections if "mask" in d]
            if current_img_masks:
                merged = np.any(current_img_masks, axis=0)
            else:
                merged = np.zeros((images.shape[1], images.shape[2]), dtype=bool)
            
            # Draw debug image with bboxes, labels, and pose
            debug_img = img_np.copy()
            h, w = debug_img.shape[:2]
            
            # COCO skeleton connections for drawing lines between keypoints
            SKELETON = [
                (0,1),(0,2),(1,3),(2,4),           # head
                (5,6),(5,7),(7,9),(6,8),(8,10),     # arms
                (5,11),(6,12),(11,12),              # torso
                (11,13),(13,15),(12,14),(14,16)     # legs
            ]
            # Colors per keypoint region (RGB)
            KPT_COLOR = [
                [255,0,0],    # 0 nose - red
                [255,85,0],   # 1 l_eye
                [255,170,0],  # 2 r_eye
                [255,255,0],  # 3 l_ear
                [170,255,0],  # 4 r_ear
                [0,255,0],    # 5 l_shoulder - green
                [0,255,85],   # 6 r_shoulder
                [0,255,170],  # 7 l_elbow
                [0,255,255],  # 8 r_elbow
                [0,170,255],  # 9 l_wrist
                [0,85,255],   # 10 r_wrist
                [0,0,255],    # 11 l_hip - blue
                [85,0,255],   # 12 r_hip
                [170,0,255],  # 13 l_knee
                [255,0,255],  # 14 r_knee
                [255,0,170],  # 15 l_ankle
                [255,0,85],   # 16 r_ankle
            ]
            
            for det in detections:
                x1, y1, x2, y2 = [int(c) for c in det["bbox"]]
                conf = det.get("confidence", 0)
                label = f'{det["label"]} {conf:.2f}'
                
                # Draw bbox rectangle (green, 2px)
                t = 2  # thickness
                debug_img[max(0,y1):min(h,y1+t), max(0,x1):min(w,x2)] = [0, 255, 0]
                debug_img[max(0,y2-t):min(h,y2), max(0,x1):min(w,x2)] = [0, 255, 0]
                debug_img[max(0,y1):min(h,y2), max(0,x1):min(w,x1+t)] = [0, 255, 0]
                debug_img[max(0,y1):min(h,y2), max(0,x2-t):min(w,x2)] = [0, 255, 0]
                
                # Draw label background
                label_h = 16
                label_w = min(len(label) * 8 + 4, w - x1)
                ly1 = max(0, y1 - label_h)
                debug_img[ly1:y1, x1:min(w, x1 + label_w)] = [0, 255, 0]
                
                # Draw pose keypoints and skeleton if available
                kpts = det.get("pose", [])
                if kpts and len(kpts) >= 17:
                    # Draw skeleton lines first (so dots are on top)
                    for (a, b) in SKELETON:
                        ax, ay = int(kpts[a][0]), int(kpts[a][1])
                        bx, by = int(kpts[b][0]), int(kpts[b][1])
                        if ax > 0 and ay > 0 and bx > 0 and by > 0:
                            # Simple line drawing using numpy (Bresenham-lite)
                            n_steps = max(abs(bx - ax), abs(by - ay), 1)
                            for s in range(n_steps + 1):
                                px = int(ax + (bx - ax) * s / n_steps)
                                py = int(ay + (by - ay) * s / n_steps)
                                if 0 <= py < h and 0 <= px < w:
                                    debug_img[max(0,py-1):min(h,py+1), max(0,px-1):min(w,px+1)] = [200, 200, 200]
                    
                    # Draw keypoint dots
                    r = 3  # radius
                    for ki, (kx, ky) in enumerate(kpts[:17]):
                        kx, ky = int(kx), int(ky)
                        if kx > 0 and ky > 0 and 0 <= ky < h and 0 <= kx < w:
                            color = KPT_COLOR[ki] if ki < len(KPT_COLOR) else [255, 255, 255]
                            debug_img[max(0,ky-r):min(h,ky+r), max(0,kx-r):min(w,kx+r)] = color
            
            all_bboxes.append(img_bboxes)
            all_kpts.append(img_kpts)
            all_masks.append(torch.from_numpy(merged.astype(np.float32)))
            all_labels.append(img_labels)
            debug_images.append(torch.from_numpy(debug_img.astype(np.float32) / 255.0))

        return (all_bboxes, all_kpts, torch.stack(all_masks), json.dumps(all_labels), torch.stack(debug_images))

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


class AnotherSAM2VideoAddPoints:
    """Add point prompts to a specific frame for video segmentation.
    Copied from comfyui-segment-anything-2 Sam2VideoSegmentationAddPoints."""
    
    @classmethod
    def IS_CHANGED(s):
        return ""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ANOTHER_MODEL",),
                "coordinates_positive": ("STRING", {"forceInput": True}),
                "frame_index": ("INT", {"default": 0}),
                "object_index": ("INT", {"default": 0}),
            },
            "optional": {
                "image": ("IMAGE",),
                "coordinates_negative": ("STRING", {"forceInput": True}),
                "prev_state": ("ANOTHER_SAM2_STATE",),
            }
        }

    RETURN_TYPES = ("ANOTHER_MODEL", "ANOTHER_SAM2_STATE")
    RETURN_NAMES = ("model", "state")
    FUNCTION = "add_points"
    CATEGORY = "AnotherUtils/inference"

    def add_points(self, model, coordinates_positive, frame_index, object_index,
                   image=None, coordinates_negative=None, prev_state=None):
        if model["type"] != "SAM2_VIDEO":
            raise ValueError("Model must be loaded with mode='video' for video segmentation")
        
        service: SAM2VideoService = model["service"]
        
        # Parse coordinates
        try:
            pos = json.loads(coordinates_positive.replace("'", '"'))
            pos = [(c['x'], c['y']) for c in pos]
        except:
            pos = []
        
        neg = None
        if coordinates_negative:
            try:
                neg_parsed = json.loads(coordinates_negative.replace("'", '"'))
                neg = [(c['x'], c['y']) for c in neg_parsed]
            except:
                neg = None
        
        # Initialize state from images if no previous state
        if prev_state is None:
            if image is None:
                raise ValueError("Either 'image' or 'prev_state' is required")
            B, H, W, C = image.shape
            model_input_size = service.model.image_size
            # Resize to model input size: [B,H,W,C] -> [B,C,H,W]
            from comfy.utils import common_upscale
            resized = common_upscale(
                image.movedim(-1, 1),
                model_input_size, model_input_size,
                "bilinear", "disabled"
            ).movedim(1, -1)
            # init_state expects [B, C, H, W]
            service.init_state(resized.permute(0, 3, 1, 2).contiguous(), H, W)
            num_frames = B
        else:
            service.inference_state = prev_state["inference_state"]
            num_frames = prev_state["num_frames"]
        
        # Add points
        if pos:
            service.add_points(frame_index, object_index, pos, neg)
        
        state = {
            "inference_state": service.inference_state,
            "num_frames": num_frames,
        }
        return (model, state)


class AnotherSAM2VideoPropagate:
    """Propagate segmentation across all video frames.
    Copied from comfyui-segment-anything-2 Sam2VideoSegmentation."""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("ANOTHER_MODEL",),
                "state": ("ANOTHER_SAM2_STATE",),
            }
        }

    RETURN_TYPES = ("MASK",)
    FUNCTION = "propagate"
    CATEGORY = "AnotherUtils/inference"

    def propagate(self, model, state):
        if model["type"] != "SAM2_VIDEO":
            raise ValueError("Model must be loaded with mode='video'")
        
        service: SAM2VideoService = model["service"]
        service.inference_state = state["inference_state"]
        
        mask_tensor = service.propagate()
        return (mask_tensor,)


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
    BODY_PARTS = [
        "all", "face", "upper_body",
        "nose", "eyes", "ears", "shoulders", "elbows", "wrists", "hips", "knees", "ankles",
        "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "keypoints": ("KEYPOINTS",),
                "body_part": (cls.BODY_PARTS, {"default": "all"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("points_json",)
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/utils"

    def convert(self, keypoints, body_part):
        # 0: nose, 1: l_eye, 2: r_eye, 5: l_sh, 6: r_sh...
        COCO_MAP = {
            "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
            "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
        }
        # Group aliases for convenience
        ALIASES = {
            "eyes": ["left_eye", "right_eye"],
            "ears": ["left_ear", "right_ear"],
            "shoulders": ["left_shoulder", "right_shoulder"],
            "elbows": ["left_elbow", "right_elbow"],
            "wrists": ["left_wrist", "right_wrist"],
            "hips": ["left_hip", "right_hip"],
            "knees": ["left_knee", "right_knee"],
            "ankles": ["left_ankle", "right_ankle"],
            "face": ["nose", "left_eye", "right_eye"],
            "upper_body": ["nose", "left_eye", "right_eye", "left_shoulder", "right_shoulder"],
            "all": list(COCO_MAP.keys()),
        }

        points = []
        
        # Handle batching: keypoints is List[List[kpts_per_detection]]
        target_kpts = keypoints
        if keypoints and isinstance(keypoints[0], list):
            target_kpts = keypoints[0] # Take first image's detections

        # Resolve body_part to keypoint indices
        if body_part in ALIASES:
            selected_indices = [COCO_MAP[name] for name in ALIASES[body_part]]
        elif body_part in COCO_MAP:
            selected_indices = [COCO_MAP[body_part]]
        else:
            selected_indices = list(COCO_MAP.values())  # fallback to all

        for det_kpts in target_kpts:
            # det_kpts is [[x,y], [x,y], ...] with 17 entries
            for idx in selected_indices:
                if idx < len(det_kpts):
                    p = det_kpts[idx]
                    # Skip keypoints with 0,0 (not detected)
                    if float(p[0]) > 0 and float(p[1]) > 0:
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
