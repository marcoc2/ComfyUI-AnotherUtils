# 🚀 Native Inference Suite

A robust, self-contained suite of computer vision nodes utilizing embedded Python integrations. It automatically downloads its own models and handles zero-shot segmentation and detection internally, keeping workflows clean.

> **Note**: These components do not rely on large external node packs and are fully embedded within the node logic.

### 🔍 Model Loaders
- **AnotherLoadYOLO**: Downloads and loads Ultralytics models (`v8`, `11`, `pose`, `seg`). Includes built-in support for YOLO size presets.
- **AnotherLoadSAM2**: Downloads and loads META's Segment Anything 2 models (`hiera_tiny` through `large`). Supports modes for `single_image` estimation or `video` propagation.
- **AnotherLoadDepth**: Downloads and loads Depth-Anything-V3 pipelines seamlessly.

### 📐 Inference Engines
- **AnotherYOLOInference**: Performs standard object or pose detection. Returns bounding boxes, keypoints, tracking IDs, labels, and debugging frames with full visual skeletons.
- **AnotherSAM2Inference**: Single-image zero-shot segmentation supporting coordinates and bounding boxes.
- **AnotherSAM2VideoAddPoints**: Video tracking initialization. Maps point prompts natively onto a specific frame index to kickstart stateful segmentation tracking.
- **AnotherSAM2VideoPropagate**: Tracks marked subjects linearly across all frames using the `SAM2VideoPredictor` backend.

### 🛠️ Geometric Utility
- **AnotherBBoxToPoints**: Parses YOLO bounding boxes into format-friendly points for SAM 2 mapping.
- **AnotherPoseToPoints**: Transforms YOLO COCO-topology keypoints into targeted point lists using body-map dropdown menus (e.g., `face`, `left_arm`, `both_legs_only`).
- **AnotherImageToMask / AnotherMaskToImage**: Clean converters bridging binary masks to grayscale RGB imagery.
- **AnotherMaskMath**: Executes math operations dynamically on masks (`add`, `subtract`, `multiply`, `logical_and`/`or`).
- **AnotherMaskBlur**: Quick standalone structural blur for mask feathering.
