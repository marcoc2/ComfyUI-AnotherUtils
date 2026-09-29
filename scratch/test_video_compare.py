import os
import sys
from pathlib import Path

# Add ComfyUI root and custom_nodes directory to sys.path
project_dir = Path(__file__).resolve().parent.parent
custom_nodes_dir = project_dir.parent
comfyui_dir = custom_nodes_dir.parent

sys.path.insert(0, str(comfyui_dir))
sys.path.insert(0, str(custom_nodes_dir))
sys.path.insert(0, str(project_dir))

# Mock PromptServer with RouteTableDef for stand-alone testing outside server
try:
    from aiohttp import web
    from server import PromptServer
    if not hasattr(PromptServer, "instance") or PromptServer.instance is None:
        class MockServer:
            routes = web.RouteTableDef()
        PromptServer.instance = MockServer()
except Exception:
    pass

import torch

def test_video_compare_slider():
    from video.video_compare_slider import VideoCompareSlider
    
    node = VideoCompareSlider()
    
    # 1. Test pause_count = 0 (continuous sweep)
    v_a = torch.zeros((30, 64, 64, 3), dtype=torch.float32)
    v_b = torch.ones((30, 64, 64, 3), dtype=torch.float32)
    
    out_tensor, fps = node.generate_slider_comparison(
        video_a=v_a,
        video_b=v_b,
        pause_count=0,
        custom_pause_frames="",
        pause_duration_frames=10,
        sweep_mode="ping_pong (0% -> 100% -> 0%)",
        line_width=2,
        line_color="white",
        fps=24.0,
        resting_position=0.5
    )
    assert out_tensor.shape == (30, 64, 64, 3), f"Expected shape (30, 64, 64, 3), got {out_tensor.shape}"
    assert fps == 24.0, f"Expected fps 24.0, got {fps}"
    print("Test 1 (pause_count=0) PASSED. Output shape:", out_tensor.shape)
    
    # 2. Test pause_count = 1 (1 pause at halfway)
    # 30 frames -> 1 pause of 10 frames -> (30 - 1) + 10 = 39 frames
    out_tensor, fps = node.generate_slider_comparison(
        video_a=v_a,
        video_b=v_b,
        pause_count=1,
        custom_pause_frames="",
        pause_duration_frames=10,
        sweep_mode="ping_pong (0% -> 100% -> 0%)",
        line_width=2,
        line_color="red",
        fps=30.0,
        resting_position=0.5
    )
    assert out_tensor.shape == (39, 64, 64, 3), f"Expected shape (39, 64, 64, 3), got {out_tensor.shape}"
    print("Test 2 (pause_count=1) PASSED. Output shape:", out_tensor.shape)
    
    # 3. Test pause_count = 2 (2 pauses at 1/3 and 2/3)
    # 30 frames -> 2 pauses of 10 frames -> (30 - 2) + 20 = 48 frames
    out_tensor, fps = node.generate_slider_comparison(
        video_a=v_a,
        video_b=v_b,
        pause_count=2,
        custom_pause_frames="",
        pause_duration_frames=10,
        sweep_mode="left_to_right (0% -> 100%)",
        line_width=4,
        line_color="yellow",
        fps=24.0,
        resting_position=0.5
    )
    assert out_tensor.shape == (48, 64, 64, 3), f"Expected shape (48, 64, 64, 3), got {out_tensor.shape}"
    print("Test 3 (pause_count=2) PASSED. Output shape:", out_tensor.shape)
    
    # 4. Test custom_pause_frames string ("5, 12, 20")
    # 3 pauses -> (30 - 3) + 30 = 57 frames
    out_tensor, fps = node.generate_slider_comparison(
        video_a=v_a,
        video_b=v_b,
        pause_count=0, # Should be overridden by custom_pause_frames
        custom_pause_frames="5, 12, 20",
        pause_duration_frames=10,
        sweep_mode="ping_pong_rest (rest -> 100% -> 0% -> rest)",
        line_width=2,
        line_color="green",
        fps=24.0,
        resting_position=0.5
    )
    assert out_tensor.shape == (57, 64, 64, 3), f"Expected shape (57, 64, 64, 3), got {out_tensor.shape}"
    print("Test 4 (custom_pause_frames='5, 12, 20') PASSED. Output shape:", out_tensor.shape)

    # 5. Test resolution mismatch and length mismatch
    v_a_diff = torch.zeros((20, 100, 100, 3), dtype=torch.float32)
    v_b_diff = torch.ones((25, 50, 80, 3), dtype=torch.float32)
    out_tensor, fps = node.generate_slider_comparison(
        video_a=v_a_diff,
        video_b=v_b_diff,
        pause_count=1,
        custom_pause_frames="",
        pause_duration_frames=5,
        sweep_mode="ping_pong (0% -> 100% -> 0%)",
        line_width=2,
        line_color="white",
        fps=24.0,
        resting_position=0.5
    )
    # min frames = 20, 1 pause of 5 frames -> (20 - 1) + 5 = 24 frames total, shape [24, 100, 100, 3]
    assert out_tensor.shape == (24, 100, 100, 3), f"Expected shape (24, 100, 100, 3), got {out_tensor.shape}"
    print("Test 5 (resolution and length mismatch) PASSED. Output shape:", out_tensor.shape)

    # 6. Test optional input pause_frames_input as list
    # 2 pauses -> (30 - 2) + 20 = 48 frames
    out_tensor, fps = node.generate_slider_comparison(
        video_a=v_a,
        video_b=v_b,
        pause_count=0,
        custom_pause_frames="",
        pause_duration_frames=10,
        sweep_mode="ping_pong (0% -> 100% -> 0%)",
        line_width=2,
        line_color="white",
        fps=24.0,
        resting_position=0.5,
        pause_frames_input=[2, 8]
    )
    assert out_tensor.shape == (48, 64, 64, 3), f"Expected shape (48, 64, 64, 3), got {out_tensor.shape}"
    print("Test 6 (pause_frames_input as list) PASSED. Output shape:", out_tensor.shape)

    # 7. Test method overlay (e.g. 'RTX')
    v_a_hd = torch.zeros((10, 256, 256, 3), dtype=torch.float32)
    v_b_hd = torch.zeros((10, 256, 256, 3), dtype=torch.float32)
    out_tensor, fps = node.generate_slider_comparison(
        video_a=v_a_hd,
        video_b=v_b_hd,
        pause_count=0,
        custom_pause_frames="",
        pause_duration_frames=10,
        sweep_mode="ping_pong (0% -> 100% -> 0%)",
        line_width=2,
        line_color="white",
        fps=24.0,
        resting_position=0.5,
        method="RTX",
        method_font_scale=1.0,
        method_position="bottom_corners"
    )
    assert out_tensor.shape == (10, 256, 256, 3), f"Expected shape (10, 256, 256, 3), got {out_tensor.shape}"
    # Verify text was drawn (non-zero values in black background)
    assert out_tensor.max() > 0.0, "Expected non-zero pixels from method overlay"
    print("Test 7 (method overlay 'RTX') PASSED. Output shape:", out_tensor.shape)

    # 8. Test package import like ComfyUI does
    import importlib
    pkg = importlib.import_module("ComfyUI-AnotherUtils")
    assert "VideoCompareSlider" in pkg.NODE_CLASS_MAPPINGS
    assert "VideoCompareSlider" in pkg.NODE_DISPLAY_NAME_MAPPINGS
    print("Test 8 (ComfyUI-AnotherUtils package import & registration) PASSED.")

    print("\nALL 8 TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_video_compare_slider()
