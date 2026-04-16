import sys
import os
import traceback

# Setup paths to match ComfyUI environment
BASE_DIR = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI"
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "custom_nodes"))

print("Attempting to import ComfyUI-AnotherUtils...")
try:
    import importlib
    # ComfyUI often maps hyphenated folder names differently, 
    # but let's try direct importlib which is what ComfyUI does.
    mod = importlib.import_module("ComfyUI-AnotherUtils")
    print("SUCCESS: Package imported correctly.")
except Exception as e:
    print("FAILURE: Import failed with traceback:")
    traceback.print_exc()
