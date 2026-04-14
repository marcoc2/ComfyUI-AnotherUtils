# 🖌️ GIMP-Style Filters

Professional-grade image post-processing filters modeled directly after GIMP and GEGL implementations for high-quality mathematical adjustments.

#### Adaptive Noise Filter
Applies frequency-aware noise (Gaussian or Uniform) that naturally adapts to image content—providing more noise structurally on edges and less on flat gradients.

#### RGB Noise (GEGL-like)
RGB channel noise filter perfectly matching GIMP's GEGL implementation. Provides separated channel noise with correlation tracking.

#### CIE LCH Noise (GEGL-like)
A perceptually uniform noise filter injected directly into the CIE LCH color space. Very useful for simulating natural film grain and distributing noise across Lightness, Chroma, and Hue axes.

#### Mean Curvature Blur (GEGL-like)
An edge-preserving blur filter based on surface curvature analysis. It smooths flat areas iteratively without blurring critical structural edges.
