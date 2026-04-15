import json

def deep_search(data, target_keys):
    results = {}
    def _search(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k in target_keys:
                    if k not in results: results[k] = []
                    results[k].append(v)
                _search(v)
        elif isinstance(obj, list):
            for item in obj:
                _search(item)
    _search(data)
    return results

f_fast = 'F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME.json'
f_guide = 'F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME_GUIDE.json'

keys = ['steps', 'width', 'height', 'frame_count', 'frames', 'denoise', 'upscale_by', 'seed', 'sampler_name', 'scheduler']

print("--- FAST ---")
data_fast = json.load(open(f_fast, 'r', encoding='utf-8'))
res_fast = deep_search(data_fast, keys)
for k, v in res_fast.items():
    print(f"{k}: {v}")

print("\n--- GUIDE ---")
data_guide = json.load(open(f_guide, 'r', encoding='utf-8'))
res_guide = deep_search(data_guide, keys)
for k, v in res_guide.items():
    print(f"{k}: {v}")
