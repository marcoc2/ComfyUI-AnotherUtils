import torch
import numpy as np
import os
from ultralytics import YOLO
from .utils import ModelDownloader, YOLO_URLS

class DetectionService:
    def __init__(self, model_path, device="auto"):
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        
        # Check and download if missing
        if not os.path.exists(model_path):
            filename = os.path.basename(model_path)
            if filename in YOLO_URLS:
                success = ModelDownloader.download(YOLO_URLS[filename], model_path)
                if not success:
                    print(f"[AnotherUtils] Shared downloader failed for {filename}, falling back to Ultralytics native loader.")
            # If not in our URL list, let ultralytics handle it (it has its own download logic)

        try:
            self.model = YOLO(model_path).to(self.device).float()
            print(f"[AnotherUtils] YOLO Model loaded: {model_path} on {self.device}")
        except Exception as e:
            print(f"[AnotherUtils] Failed to load YOLO model: {e}")
            raise e

    def infer(self, image_np, conf=0.25, iou=0.45):
        """
        image_np: [H, W, 3] RGB uint8
        returns: List[dict] where each dict has:
            - "bbox": [x1, y1, x2, y2]
            - "label": str (class name)
            - "confidence": float
            - "mask": np.ndarray (H, W) bool (if segmentation model)
            - "pose": list of [x, y] keypoints (if pose model)
        """
        raw_results = self.model.predict(
            source=image_np,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )

        detections = []
        for result in raw_results:
            names = result.names  # {0: 'person', 1: 'bicycle', ...}
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for i in range(len(boxes)):
                det = {
                    "bbox": boxes.xyxy[i].cpu().numpy().tolist(),
                    "label": names[int(boxes.cls[i].item())],
                    "confidence": float(boxes.conf[i].item()),
                }

                # Segmentation masks (if using a -seg model)
                if result.masks is not None and i < len(result.masks):
                    det["mask"] = result.masks.data[i].cpu().numpy().astype(bool)

                # Pose keypoints (if using a -pose model)
                if result.keypoints is not None and i < len(result.keypoints):
                    kpts = result.keypoints.xy[i].cpu().numpy().tolist()
                    det["pose"] = kpts

                detections.append(det)

        return detections
