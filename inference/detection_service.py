import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import batched_nms
import os

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

def letterbox(img: torch.Tensor, new_shape=(640, 640), color=(114, 114, 114)):
    """Convert image to standard dimension, letterbox format (CHW)"""
    if img.ndim == 3:
        _, h, w = img.shape
    elif img.ndim == 4 and img.shape[0] == 1:
        _, _, h, w = img.shape
        img = img.squeeze(0)
    else:
        raise ValueError(f"Input tensor must be 3D (C, H, W), got {img.shape}")
        
    new_h, new_w = new_shape
    scale = min(new_w / w, new_h / h)
    nh, nw = int(round(h * scale)), int(round(w * scale))

    img = F.interpolate(img.unsqueeze(0), size=(nh, nw), mode='bilinear', align_corners=False)[0]

    pad_w, pad_h = new_w - nw, new_h - nh
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    
    fill_value = float(color[0]) 
    img = F.pad(img, (pad_left, pad_right, pad_top, pad_bottom), value=fill_value)

    return img

def scale_boxes(img_shape, boxes, orig_shape):
    """Rescale boxes from model size to original image size"""
    gain = min(img_shape[0] / orig_shape[0], img_shape[1] / orig_shape[1])
    pad_x = (img_shape[1] - orig_shape[1] * gain) / 2
    pad_y = (img_shape[0] - orig_shape[0] * gain) / 2

    boxes[:, [0, 2]] -= pad_x
    boxes[:, [1, 3]] -= pad_y
    boxes[:, :4] /= gain

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_shape[0])
    return boxes

class DetectionService:
    def __init__(self, model_path, threshold=0.25, device="auto", half=True):
        if YOLO is None:
            raise ImportError("The 'ultralytics' library is not installed. Please install it to use this node.")
            
        self.model_path = model_path
        self.threshold = threshold
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.half = half and self.device == "cuda"
        
        self.model = YOLO(model_path).to(self.device)
        self.model.eval()
        if self.half:
            self.model.half()
        else:
            self.model.float()
            
        self.labels = self.model.names

    def infer(self, image_np):
        """
        image_np: [H, W, 3] uint8
        Returns list of detections: [{"bbox": [x1,y1,x2,y2], "conf": 0.9, "class_id": 0, "label": "person", "pose": kpts}]
        """
        h, w = image_np.shape[:2]
        
        # Pre-process
        img_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float().to(self.device)
        input_tensor = letterbox(img_tensor, (640, 640)) / 255.0
        if self.half:
            input_tensor = input_tensor.half()
        input_tensor = input_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            results = self.model(input_tensor, conf=self.threshold, verbose=False)[0]

        detections = []
        
        # Results from Ultralytics
        # boxes = results.boxes (xyxy, conf, cls)
        # keypoints = results.keypoints (if available)
        
        if results.boxes is not None:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)
            
            # Scale boxes
            boxes = scale_boxes((640, 640), boxes, (h, w))
            
            has_pose = results.keypoints is not None
            kpts = None
            if has_pose:
                kpts = results.keypoints.data.cpu().numpy() # [N, 17, 3] or [N, 17, 2]
                
                # Scale keypoints
                gain = min(640 / w, 640 / h)
                pad_x = (640 - w * gain) / 2
                pad_y = (640 - h * gain) / 2
                
                kpts[..., 0] = (kpts[..., 0] - pad_x) / gain
                kpts[..., 1] = (kpts[..., 1] - pad_y) / gain
                
                # Clip
                kpts[..., 0] = np.clip(kpts[..., 0], 0, w - 1)
                kpts[..., 1] = np.clip(kpts[..., 1], 0, h - 1)

            for i in range(len(boxes)):
                det = {
                    "bbox": boxes[i].tolist(),
                    "conf": float(confs[i]),
                    "class_id": int(clss[i]),
                    "label": self.labels[int(clss[i])]
                }
                if has_pose:
                    det["pose"] = kpts[i]
                detections.append(det)

        return detections
