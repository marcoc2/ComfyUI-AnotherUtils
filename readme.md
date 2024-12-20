# Image Processing Suite for ComfyUI

A collection of specialized image processing nodes for ComfyUI, focused on dataset preparation and pixel art manipulation.

## Installation

1. Create a `custom_nodes` directory in your ComfyUI installation if it doesn't exist
2. Clone this repository inside the `custom_nodes` directory:
```bash
cd custom_nodes
git clone https://github.com/marcoc2/ComfyUI-AnotherUtils
```
3. Restart ComfyUI

## Nodes

### Load Images (Original Size)
Loads all images from a directory while preserving their original dimensions.

**Inputs:**
- `directory`: Path to the directory containing images

**Outputs:**
- List of images in their original sizes

**Features:**
- Preserves original image dimensions
- Compatible with ComfyUI's RebatchImages node
- Supports common image formats (png, jpg, jpeg, bmp, webp)

### Custom Crop
Crops images with specific positioning options.

**Inputs:**
- `image`: Input image
- `crop_width`: Width of crop area
- `crop_height`: Height of crop area
- `crop_mode`: Cropping position ("center", "left", "right", "top", "bottom")

**Outputs:**
- Cropped image

### Smart Resize
Resizes images to a target size while intelligently filling missing areas.

**Inputs:**
- `image`: Input image
- `target_size`: Desired size
- `border_sample_size`: Pixels to sample for border color
- `color_method`: Method to determine fill color ("mean" or "mode")

**Outputs:**
- Resized image with intelligent border filling

### Nearest Neighbor Upscale
Performs upscaling using nearest neighbor interpolation, perfect for pixel art.

**Inputs:**
- `image`: Input image
- `scale_factor`: Multiplication factor for upscaling (1-8)

**Outputs:**
- Upscaled image without interpolation artifacts

### Pixel Art Normalizer
Normalizes images into pixel art style with consistent grid sizes.

**Inputs:**
- `image`: Input image
- `block_size`: Size of pixel blocks (0 for auto-detection)
- `n_colors`: Number of colors in output (0 for auto-detection)

**Outputs:**
- `normalized`: Normalized pixel art image at original size
- `block_size`: Detected/used block size
- `downscaled`: 1:1 pixel art version (downscaled by block size)

**Features:**
- Automatic grid size detection
- Color quantization
- Outputs both full-size and true 1:1 pixel art versions

## Usage Examples

### Basic Image Loading and Batching
```
LoadImagesOriginal -> RebatchImages -> [Further Processing]
```

### Pixel Art Creation Pipeline
```
LoadImagesOriginal -> PixelArtNormalizer -> NearestUpscale
```

### Dataset Preparation
```
LoadImagesOriginal -> CustomCrop -> SmartResize -> [Training]
```

## Dependencies
- NumPy
- OpenCV (cv2)
- scikit-learn
- PIL
- PyTorch (provided by ComfyUI)

## Notes
- All nodes maintain compatibility with ComfyUI's native nodes
- Images are handled in RGB format
- All operations preserve proper normalization (0-1 range)

## Contributing
Feel free to open issues or submit pull requests for improvements.

