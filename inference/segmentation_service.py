import numpy as np
import torch
import os

try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    build_sam2 = None

class SAM2Service:
    def __init__(self, model_cfg, checkpoint_path, device="auto"):
        if build_sam2 is None:
            raise ImportError("The 'sam2' library is not installed. Please install it to use this node.")
            
        self.device = "cuda" if (device == "auto" and torch.cuda.is_available()) else device
        self.model = build_sam2(model_cfg, checkpoint_path, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)

    def set_image(self, image_np):
        """image_np: [H, W, 3] RGB"""
        self.predictor.set_image(image_np)

    def predict(self, points=None, labels=None, bboxes=None):
        """
        Predict masks for the current image.
        points: List of [x, y]
        labels: List of 1 (pos) or 0 (neg)
        bboxes: List of [x1, y1, x2, y2]
        Returns: [N, H, W] boolean mask
        """
        # Ensure numpy
        if points is not None:
            points = np.array(points)
        if labels is not None:
            labels = np.array(labels)
        if bboxes is not None:
            bboxes = np.array(bboxes)

        # SAM 2 predictor.predict handles points, boxes or both
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            box=bboxes,
            multimask_output=False,
        )
        
        # masks is [N, 1, H, W] or [H, W] depending on input
        if masks.ndim == 4:
            masks = masks.squeeze(1)
            
        return masks
