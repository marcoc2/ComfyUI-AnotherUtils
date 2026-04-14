import os
import urllib.request
from tqdm import tqdm

class ModelDownloader:
    @staticmethod
    def download(url, dest_path):
        """
        Downloads a file from url to dest_path with a progress bar.
        """
        if os.path.exists(dest_path):
            return True

        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        print(f"[AnotherUtils] Downloading model from: {url}")
        print(f"[AnotherUtils] Dest: {dest_path}")
        
        try:
            # We use tqmd for the progress bar if available
            response = urllib.request.urlopen(url)
            total_size = int(response.headers.get('content-length', 0))
            
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(dest_path)) as pbar:
                with open(dest_path, 'wb') as f:
                    while True:
                        chunk = response.read(8192)
                        if not chunk:
                            break
                        f.write(chunk)
                        pbar.update(len(chunk))
            return True
        except Exception as e:
            print(f"[AnotherUtils] Download failed: {e}")
            if os.path.exists(dest_path):
                os.remove(dest_path)
            return False

# Official model URLs
SAM2_URLS = {
    "sam2_hiera_tiny.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt",
    "sam2_hiera_small.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt",
    "sam2_hiera_base_plus.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt",
    "sam2_hiera_large.pt": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt",
}

YOLO_URLS = {
    "yolov8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolov8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolov8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    "yolov8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
    "yolov8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
    "yolo11n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11n.pt",
    "yolo11s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11s.pt",
    "yolo11m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11m.pt",
    "yolo11l.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11l.pt",
    "yolo11x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolo11x.pt",
    "face_yolov8m.pt": "https://github.com/anotherutils/models/releases/download/v1.0/face_yolov8m.pt", # Example 
}
