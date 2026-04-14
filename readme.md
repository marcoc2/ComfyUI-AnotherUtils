# ComfyUI AnotherUtils

A comprehensive and standalone collection of robust custom nodes for ComfyUI. 
AnotherUtils focuses on providing heavy-lifting utilities without requiring massive external dependencies. It encompasses advanced AI pipelines natively, classic pixel transformations, character animations, professional GEGL/GIMP filters, operations, and robust dataset preparation functions.

## 📚 Documentation

The documentation has been categorized into multiple modules to keep things clean. Click on any category below for a detailed list and explanation of the nodes available:

- [🚀 Native Inference Suite](docs/INFERENCE.md) - YOLO Detection, SAM 2 (Image & Video), DepthAnything, and bounding box geometric utilities.
- [📋 Trello Integrations](docs/TRELLO_INTEGRATION.md) - Interactive web-browser nodes and automated benchmark loaders targeting Trello boards.
- [🖼️ Image Processing](docs/IMAGE_PROCESSING.md) - Batch-capable image loaders, boundary-aware resizers, precision croppers, and type-detection logic.
- [🎨 Pixel Art Tools](docs/PIXEL_ART.md) - True mathematical pixel art normalization, scaling, and color quantization.
- [👾 Character & Animation](docs/CHARACTER_AND_ANIMATION.md) - Randomizer grids, prompt builders, sprite creators, and walking pose engines.
- [🖌️ GIMP-Style Filters](docs/FILTERS.md) - High-quality mathematical filters like Mean Curvature Blur, Adaptive Frequency Noise, and LCH processing logic.

## 📦 Installation

1. Navigate to your ComfyUI `custom_nodes` directory
2. Clone this repository:
```bash
git clone https://github.com/marcoc2/ComfyUI-AnotherUtils.git
```
3. Restart ComfyUI

*(Note: Essential weights like YOLO, SAM2, and DepthAnything models are automatically fetched and downloaded into the `models/another_utils` or generic `models/sub-folder` namespaces seamlessly on their first run. No manual weight hunting required!)*

## 🧩 Features
- **Zero-Dependency Core Models**: The Inference Suite runs logic locally using internal packages heavily mimicking their official counterparts, preventing dependency hell against major node packs.
- **Batch Processing**: Most nodes scale into dimensional batches instantly for video outputs or dataset rendering.
- **Memory Efficient**: Internal GPU structures and tensors are forcefully released or managed via JIT routines for optimal layout.
- **Deterministic**: Complete seed-based structural operations for perfect tracking tests.

## 🤝 Contributing
Issues and pull requests are welcome! Please follow the existing code paradigms natively embedded inside.

## 📜 License
MIT License - See LICENSE file for details.

---
*Created by marcoags*