import os
import sys
import time
from pathlib import Path

project_dir = Path(__file__).resolve().parent.parent
custom_nodes_dir = project_dir.parent
comfyui_dir = custom_nodes_dir.parent

sys.path.insert(0, str(comfyui_dir))
sys.path.insert(0, str(custom_nodes_dir))
sys.path.insert(0, str(project_dir))

import torch

def benchmark_memory():
    from video.video_compare_slider import VideoCompareSlider

    node = VideoCompareSlider()

    print("Generating 500-frame test tensors (1080p)...")
    # 500 frames of 1080x1920 float32
    # In CPU RAM: 500 * 1080 * 1920 * 3 * 4 bytes = 12.44 GB
    v_a = torch.zeros((500, 1080, 1920, 3), dtype=torch.float32)
    v_b = torch.ones((500, 1080, 1920, 3), dtype=torch.float32)

    print("Running VideoCompareSlider with 500 frames, method='RTX', 2 pauses...")
    start_time = time.time()
    
    out, fps = node.generate_slider_comparison(
        video_a=v_a,
        video_b=v_b,
        pause_count=2,
        custom_pause_frames="",
        pause_duration_frames=30,
        sweep_mode="ping_pong (0% -> 100% -> 0%)",
        line_width=4,
        line_color="white",
        fps=24.0,
        resting_position=0.5,
        method="RTX",
        method_font_scale=1.0,
        method_position="bottom_corners",
    )
    
    elapsed = time.time() - start_time
    print(f"DONE in {elapsed:.2f}s! Output shape: {out.shape}")

if __name__ == "__main__":
    benchmark_memory()
