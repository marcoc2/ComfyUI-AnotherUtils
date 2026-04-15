# 👾 Character & Animation

Tools for generating character descriptions, prompts, and corresponding sprite-sheet animations.

#### Character Randomizer
Generates random character attributes for fighting game characters (gender, fighting style, nationality, etc.) utilizing a deterministic seed-based system.

#### Character Constructor
Constructs detailed, rich character prompts from the attributes provided by the Randomizer to be used directly in AI image generation workflows.

#### Fighting Game Character Generator
Creates character sprites and animations from character parameters and pose settings.

#### Walking Pose Generator
Generates walking animation frame sequences given a base character and walk cycle settings.

---

### 🎥 Advanced Batch Animation
Nodes designed specifically for animating batch image sequences over backgrounds.

#### Transform Keyframes (Math)
Calculates smooth interpolation for X, Y, and Scale (Zoom) parameters across N frames. Supports multiple easing functions (linear, ease-in, etc.).

#### Animated Composite Masked (Batch)
High-performance compositor that uses Transform Data to move and resize a foreground batch over a background canvas. Built-in memory safety ensures it can handle 100+ frames without crashing ComfyUI's CPU memory. Supports **Anchor Modes** (Top-Left or Center).
