import sys
import os
import traceback

# Setup paths to match ComfyUI environment
BASE_DIR = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "custom_nodes"))

output_file = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\custom_nodes\ComfyUI-AnotherUtils\scratch\traceback_output.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("Starting granular diagnostic test...\n")
    
    modules_to_test = [
        "ComfyUI-AnotherUtils.logic_management.image_list_to_batch",
        "ComfyUI-AnotherUtils.logic_management.indices_list_to_50",
        "ComfyUI-AnotherUtils.video.another_ltx_sequencer"
    ]
    
    import importlib
    for mod_name in modules_to_test:
        f.write(f"\nTesting {mod_name}...\n")
        try:
            importlib.import_module(mod_name)
            f.write(f"SUCCESS: {mod_name} imported.\n")
        except Exception:
            f.write(f"FAILURE: {mod_name} failed with:\n")
            f.write(traceback.format_exc())
            f.write("\n")
