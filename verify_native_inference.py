import os
import sys
import torch
import numpy as np
import cv2
from PIL import Image

# Add ComfyUI path to sys.path to allow imports from custom_nodes
comfy_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if comfy_path not in sys.path:
    sys.path.append(comfy_path)

# Correct target directory for AnotherUtils
another_utils_path = os.path.dirname(os.path.abspath(__file__))
if another_utils_path not in sys.path:
    sys.path.append(another_utils_path)

from inference.detection_service import DetectionService
from inference.segmentation_service import SAM2Service
from inference.depth_service import DepthAnythingService

def test_inference():
    print("=== AnotherUtils Native Inference Verification (Manual Loader) ===")
    
    # 1. Prepare Image
    input_dir = os.path.join(comfy_path, "input")
    try:
        test_img_path = os.path.join(input_dir, "example.png")
        if not os.path.exists(test_img_path):
             files = [f for f in os.listdir(input_dir) if f.endswith((".png", ".jpg", ".webp"))]
             if not files:
                 print("Error: No test image found in input folder.")
                 return
             test_img_path = os.path.join(input_dir, files[0])
        
        print(f"Testing with image: {test_img_path}")
        img = Image.open(test_img_path).convert("RGB")
        img_np = np.array(img)
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 2. Test YOLO (Detection)
    try:
        print("\n--- Testing YOLOv8 (Detection) ---")
        # Try finding a face model or person model
        models_dir = os.path.join(comfy_path, "models", "ultralytics", "bbox")
        model_path = os.path.join(models_dir, "face_yolov8m.pt")
        if not os.path.exists(model_path):
             model_path = os.path.join(comfy_path, "models", "ultralytics", "segm", "person_yolov8m-seg.pt")
        
        if os.path.exists(model_path):
            yolo = DetectionService(model_path)
            results = yolo.infer(img_np)
            print(f"Success! Found {len(results)} detections.")
        else:
            print(f"Warning: YOLO model not found. Skipping.")
    except Exception as e:
        print(f"FAILED: YOLO Error: {e}")

    # 3. Test SAM 2
    try:
        print("\n--- Testing SAM 2 ---")
        sam_dir = os.path.join(comfy_path, "models", "sam2")
        sam_files = []
        if os.path.exists(sam_dir):
            sam_files = [f for f in os.listdir(sam_dir) if f.endswith((".safetensors", ".pt"))]
        if sam_files:
            sam_path = os.path.join(sam_dir, sam_files[0])
            print(f"Loading SAM2 from: {sam_path}")
            sam2_svc = SAM2Service(None, sam_path)
            sam2_svc.set_image(img_np)
            mask = sam2_svc.predict(bboxes=[[10, 10, 100, 100]])
            print(f"Success! Produced mask of shape {mask.shape}")
        else:
            print(f"Warning: SAM2 model not found in {sam_dir}. Skipping.")
    except Exception as e:
        print(f"FAILED: SAM 2 Error: {e}")

    # 4. Test DepthAnything V3
    try:
        print("\n--- Testing DepthAnything V3 ---")
        depth = DepthAnythingService("v3-small")
        d_map = depth.infer(img_np)
        print(f"Success! Produced depth map of shape {d_map.shape}")
    except Exception as e:
        print(f"FAILED: Depth Error: {e}")

    print("\n=== Verification Finished ===")

if __name__ == "__main__":
    test_inference()
