# 🖼️ Image Processing & Loaders

This category focuses on efficient and robust image loading, batch processing, and dataset preparation.

#### Load Images (Original Size)
Loads all images from a directory while preserving their original dimensions. Compatible with RebatchImages and supports common formats.

#### Load Image (Remove Alpha)
Loads an image from a file path and removes the alpha channel, compositing over a white background.

#### Remove Alpha Channel
Removes the alpha channel from already loaded images in a workflow, compositing over a white background.

#### Custom Crop
Crops images with specific positioning options (Modes: center, left, right, top, bottom).

#### Smart Resize with Border Fill
Intelligently resizes images while filling missing areas with border colors using mean or mode for detection.

#### Nearest Neighbor Upscale
Perfect 1:1 scaling for pixel art without interpolation artifacts (scale factor 1-8).

#### Get Last Image
Utility node that separates and returns only the last image from a batch.

#### Image Type Detector
Utility that detects if an input is a single image or batch, redirecting the workflow and providing type flags for debugging or routing.
