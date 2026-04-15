import json
def find_samplers_deep(obj):
    if isinstance(obj, dict):
        # Look for sampler types
        t = obj.get('type')
        if t in ['SamplerCustomAdvanced', 'KSampler', 'SamplerCustom', 'LTXVScheduler', 'KSamplerAdvanced']:
            print(f"Found {t}: {obj.get('widgets_values')}")
        for v in obj.values():
            find_samplers_deep(v)
    elif isinstance(obj, list):
        for item in obj:
            find_samplers_deep(item)

def extract_all_widgets(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n--- Deep Sampler Search for {path} ---")
    find_samplers_deep(data)

extract_all_widgets('F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME.json')
extract_all_widgets('F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME_GUIDE.json')
