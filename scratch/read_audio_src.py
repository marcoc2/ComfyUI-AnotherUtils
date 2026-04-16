import sys
import os
import inspect

BASE_DIR = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI"
sys.path.insert(0, BASE_DIR)

output_file = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\custom_nodes\ComfyUI-AnotherUtils\scratch\audio_decode_source.txt"

with open(output_file, "w", encoding="utf-8") as f:
    try:
        import comfy_extras.nodes_lt_audio as lta
        f.write("Source code for LTXVAudioDecode:\n\n")
        f.write(inspect.getsource(lta.LTXVAudioDecode))
    except Exception as e:
        f.write(f"Error: {e}")
