"""
Comic Panel Detector Node for ComfyUI
Segments comic book pages into individual panels.

Based on Manga-Panel-Extractor (reference/comic_panel_detector/).
CV logic is inlined to avoid external path dependencies.
AI method (YOLOv5) lazy-loads from reference weights.
"""

import os
import logging
import numpy as np
import cv2
import torch
import warnings

logger = logging.getLogger(__name__)


# ── CV Helper Functions (from Manga-Panel-Extractor) ────────────────────────

def _is_contour_rectangular(contour: np.ndarray) -> bool:
    perimeter = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.01 * perimeter, True)
    return len(approx) == 4


def _get_background_intensity_range(grayscale_image: np.ndarray, min_range: int = 1):
    edges = [
        grayscale_image[-1, :], grayscale_image[0, :],
        grayscale_image[:, 0], grayscale_image[:, -1],
    ]
    sorted_edges = sorted(edges, key=lambda x: np.var(x))
    least_varied_edge = sorted_edges[0]
    max_intensity = max(least_varied_edge)
    min_intensity = max(min(min(least_varied_edge), max_intensity - min_range), 0)
    return min_intensity, max_intensity


def _generate_background_mask(grayscale_image: np.ndarray) -> np.ndarray:
    WHITE = 255
    LESS_WHITE, _ = _get_background_intensity_range(grayscale_image, 25)
    LESS_WHITE = max(LESS_WHITE, 240)

    _, thresh = cv2.threshold(grayscale_image, LESS_WHITE, WHITE, cv2.THRESH_BINARY)
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh)

    mask = np.zeros_like(thresh)
    PAGE_TO_SEGMENT_RATIO = 1024
    halting_area_size = mask.size // PAGE_TO_SEGMENT_RATIO

    mask_height, mask_width = mask.shape
    base_background_size_error_threshold = 0.05
    whole_background_min_width = mask_width * (1 - base_background_size_error_threshold)
    whole_background_min_height = mask_height * (1 - base_background_size_error_threshold)

    for i in np.argsort(stats[1:, 4])[::-1]:
        contour_index = i + 1
        x, y, w, h, area = stats[contour_index]
        if area < halting_area_size:
            break
        if (
            (w > whole_background_min_width) or
            (h > whole_background_min_height) or
            (_is_contour_rectangular(
                cv2.findContours(
                    (labels == contour_index).astype(np.uint8),
                    cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                )[0][0]
            ))
        ):
            mask[labels == contour_index] = WHITE

    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)
    return mask


def _preprocess_image_with_dilation(grayscale_image: np.ndarray) -> np.ndarray:
    processed = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
    processed = cv2.Laplacian(processed, -1)
    processed = cv2.dilate(processed, np.ones((5, 5), np.uint8), iterations=1)
    processed = 255 - processed
    return processed


def _preprocess_image(grayscale_image: np.ndarray) -> np.ndarray:
    processed = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
    processed = cv2.Laplacian(processed, -1)
    return processed


def _apply_adaptive_threshold(image: np.ndarray) -> np.ndarray:
    return cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 0
    )


def _is_contour_sufficiently_big(contour, image_height, image_width, min_ratio=1/32):
    image_area = image_width * image_height
    area_threshold = image_area * min_ratio
    area = cv2.contourArea(contour)
    return area > area_threshold


def _threshold_extraction(image, grayscale_image, min_ratio=1/32):
    """Fallback extraction using adaptive thresholding."""
    processed = cv2.GaussianBlur(grayscale_image, (3, 3), 0)
    processed = cv2.Laplacian(processed, -1)
    _, thresh = cv2.threshold(processed, 8, 255, cv2.THRESH_BINARY)
    adaptive = _apply_adaptive_threshold(processed)
    processed = cv2.subtract(adaptive, thresh)
    processed = cv2.dilate(processed, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    contours = [c for c in contours if _is_contour_sufficiently_big(c, h, w, min_ratio)]
    return _extract_panels_from_contours(image, contours, accept_page=False)


def _extract_panels_from_contours(image, contours, accept_page=True):
    """Crop bounding boxes from contours."""
    height, width = image.shape[:2]
    panels = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if not accept_page and ((w >= width * 0.99) or (h >= height * 0.99)):
            continue
        panels.append(image[y:y+h, x:x+w])
    return panels


def _detect_panels_cv(image_bgr: np.ndarray, min_ratio: float = 1/32):
    """
    Full CV pipeline: grayscale → preprocess → background mask → contours → panels.
    split_joint_panels is disabled to avoid opencv-contrib dependency.
    """
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    processed = _preprocess_image_with_dilation(grayscale)
    background_mask = _generate_background_mask(processed)

    # page without background (no split_joint_panels to avoid ximgproc dep)
    page_no_bg = cv2.subtract(grayscale, background_mask)

    contours, _ = cv2.findContours(page_no_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_bgr.shape[:2]
    contours = [c for c in contours if _is_contour_sufficiently_big(c, h, w, min_ratio)]

    panels = _extract_panels_from_contours(image_bgr, contours)

    # Fallback: if we got < 2 panels, try threshold extraction
    if len(panels) < 2:
        fallback = _threshold_extraction(image_bgr, grayscale, min_ratio)
        if len(fallback) > len(panels):
            panels = fallback

    return panels


# ── AI (YOLOv5) Method ──────────────────────────────────────────────────────

_yolo_model = None


def _load_yolo_model():
    """Lazy-load YOLOv5 custom model from reference weights."""
    global _yolo_model
    if _yolo_model is not None:
        return _yolo_model

    import pathlib
    import sys

    weights_path = os.path.join(
        os.path.dirname(__file__), '..', 'reference', 'comic_panel_detector',
        'Manga-Panel-Extractor', 'src', 'ai-models', '2024-11-00', 'best.pt'
    )
    weights_path = os.path.abspath(weights_path)

    if not os.path.isfile(weights_path):
        raise FileNotFoundError(
            f"YOLOv5 weights not found at: {weights_path}\n"
            "Make sure the reference/comic_panel_detector/ directory is intact."
        )

    # Redirect stderr if None (PyQt frozen apps)
    if sys.stderr is None:
        sys.stderr = open(os.devnull, 'w')

    # Handle PosixPath/WindowsPath compat for the saved model
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath
    try:
        _yolo_model = torch.hub.load(
            'ultralytics/yolov5', 'custom', path=weights_path, trust_repo=True
        )
    finally:
        pathlib.PosixPath = temp

    logger.info(f"YOLOv5 comic panel model loaded from {weights_path}")
    return _yolo_model


def _detect_panels_ai(image_bgr: np.ndarray):
    """Detect panels using YOLOv5 custom model."""
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    processed = _preprocess_image(grayscale)

    model = _load_yolo_model()

    warnings.filterwarnings("ignore", category=FutureWarning)
    results = model(processed)
    warnings.filterwarnings("default", category=FutureWarning)

    panels = []
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        panel = image_bgr[y1:y2, x1:x2]
        panels.append(panel)

    return panels


# ── Panel Sorting ───────────────────────────────────────────────────────────

def _sort_panels(panels, bboxes, reading_order="left_to_right"):
    """
    Sort panels in reading order: top-to-bottom rows, then left-to-right or
    right-to-left within each row.

    bboxes: list of (x, y, w, h) for each panel.
    """
    if not panels:
        return panels

    # Pair panels with their bboxes
    paired = list(zip(panels, bboxes))

    # Group into rows by vertical overlap
    # Sort by y-center first
    paired.sort(key=lambda p: p[1][1] + p[1][3] / 2)

    rows = []
    current_row = [paired[0]]

    for i in range(1, len(paired)):
        _, (_, cy, _, ch) = paired[i][0], paired[i][1]
        _, (_, py, _, ph) = current_row[-1][0], current_row[-1][1]

        center_curr = cy + ch / 2
        center_prev = py + ph / 2

        # If vertical centers are within half of the smaller height, same row
        row_threshold = min(ch, ph) * 0.5
        if abs(center_curr - center_prev) < row_threshold:
            current_row.append(paired[i])
        else:
            rows.append(current_row)
            current_row = [paired[i]]

    rows.append(current_row)

    # Sort within each row
    reverse = (reading_order == "right_to_left")
    sorted_panels = []
    for row in rows:
        row.sort(key=lambda p: p[1][0], reverse=reverse)
        sorted_panels.extend([p[0] for p in row])

    return sorted_panels


def _get_bboxes_from_panels_cv(image_bgr, min_ratio=1/32):
    """Run CV pipeline but return both panels and bboxes for sorting."""
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    processed = _preprocess_image_with_dilation(grayscale)
    background_mask = _generate_background_mask(processed)
    page_no_bg = cv2.subtract(grayscale, background_mask)

    contours, _ = cv2.findContours(page_no_bg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image_bgr.shape[:2]
    contours = [c for c in contours if _is_contour_sufficiently_big(c, h, w, min_ratio)]

    panels = []
    bboxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        panels.append(image_bgr[y:y+ch, x:x+cw])
        bboxes.append((x, y, cw, ch))

    # Fallback
    if len(panels) < 2:
        fallback_panels, fallback_bboxes = _threshold_extraction_with_bboxes(
            image_bgr, grayscale, min_ratio
        )
        if len(fallback_panels) > len(panels):
            panels = fallback_panels
            bboxes = fallback_bboxes

    return panels, bboxes


def _threshold_extraction_with_bboxes(image, grayscale, min_ratio=1/32):
    """Fallback extraction returning panels AND bboxes."""
    processed = cv2.GaussianBlur(grayscale, (3, 3), 0)
    processed = cv2.Laplacian(processed, -1)
    _, thresh = cv2.threshold(processed, 8, 255, cv2.THRESH_BINARY)
    adaptive = _apply_adaptive_threshold(processed)
    processed = cv2.subtract(adaptive, thresh)
    processed = cv2.dilate(processed, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(processed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = image.shape[:2]
    contours = [c for c in contours if _is_contour_sufficiently_big(c, h, w, min_ratio)]

    panels = []
    bboxes = []
    for contour in contours:
        x, y, cw, ch = cv2.boundingRect(contour)
        if (cw >= w * 0.99) or (ch >= h * 0.99):
            continue
        panels.append(image[y:y+ch, x:x+cw])
        bboxes.append((x, y, cw, ch))

    return panels, bboxes


def _get_bboxes_from_panels_ai(image_bgr):
    """Run AI pipeline returning panels and bboxes for sorting."""
    grayscale = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    processed = _preprocess_image(grayscale)

    model = _load_yolo_model()

    warnings.filterwarnings("ignore", category=FutureWarning)
    results = model(processed)
    warnings.filterwarnings("default", category=FutureWarning)

    panels = []
    bboxes = []
    for detection in results.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection.tolist()
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        panels.append(image_bgr[y1:y2, x1:x2])
        bboxes.append((x1, y1, x2 - x1, y2 - y1))

    return panels, bboxes


# ── Tensor Conversion Helpers ───────────────────────────────────────────────

def _comfy_to_bgr(image_tensor):
    """Convert ComfyUI tensor [B,H,W,C] float32 RGB → numpy BGR uint8."""
    # Take single image from batch
    img_np = image_tensor.cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    # RGB → BGR
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    return img_bgr


def _bgr_to_comfy(image_bgr):
    """Convert numpy BGR uint8 → ComfyUI tensor [1,H,W,C] float32 RGB."""
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img_float = img_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_float).unsqueeze(0)  # [1,H,W,C]
    return tensor


# ── ComfyUI Node ────────────────────────────────────────────────────────────

class ComicPanelDetector:
    """
    Segments comic book pages into individual panels.

    Supports two methods:
    - CV: Traditional computer vision (fast, no GPU needed)
    - AI: YOLOv5 custom model (more robust for complex layouts)

    Returns a list of panel images sorted in reading order.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "method": (["cv", "ai"], {"default": "cv"}),
                "reading_order": (
                    ["left_to_right", "right_to_left"],
                    {"default": "left_to_right"},
                ),
                "min_panel_ratio": (
                    "FLOAT",
                    {"default": 0.03, "min": 0.005, "max": 0.25, "step": 0.005},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("panels", "panel_count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "detect_panels"
    CATEGORY = "AnotherUtils/image_processing"

    def detect_panels(self, image, method, reading_order, min_panel_ratio):
        batch_size = image.shape[0]
        all_panels = []
        total_count = 0

        for b in range(batch_size):
            single_img = image[b]  # [H, W, C]
            img_bgr = _comfy_to_bgr(single_img)

            if method == "ai":
                panels, bboxes = _get_bboxes_from_panels_ai(img_bgr)
            else:
                panels, bboxes = _get_bboxes_from_panels_cv(img_bgr, min_ratio=min_panel_ratio)

            # Sort panels in reading order
            if panels and bboxes:
                panels = _sort_panels(panels, bboxes, reading_order)

            # Convert back to ComfyUI tensors
            for panel in panels:
                all_panels.append(_bgr_to_comfy(panel))

            total_count += len(panels)
            logger.info(
                f"[ComicPanelDetector] Page {b+1}/{batch_size}: "
                f"{len(panels)} panels detected ({method})"
            )

        # If no panels detected, return the original image as-is
        if not all_panels:
            logger.warning("[ComicPanelDetector] No panels detected, returning original image")
            all_panels = [image[b:b+1] for b in range(batch_size)]
            total_count = batch_size

        return (all_panels, total_count)
