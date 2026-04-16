import sys
import os
import torch

BASE_DIR = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI"
sys.path.insert(0, BASE_DIR)

output_file = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\custom_nodes\ComfyUI-AnotherUtils\scratch\audio_info.txt"

with open(output_file, "w", encoding="utf-8") as f:
    try:
        import comfy_extras.nodes_lt_audio as lta
        from comfy.ldm.lightricks.vae.audio_vae import AudioVAE
        f.write("Audio VAE Info:\n")
        # Just checking if we can see constants
    except Exception as e:
        f.write(f"Error: {e}")
