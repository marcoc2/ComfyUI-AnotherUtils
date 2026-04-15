import json

def find_all(obj, search_keys, results):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in search_keys:
                if k not in results: results[k] = []
                results[k].append(v)
            if isinstance(v, str) and (v.endswith('.safetensors') or v.endswith('.gguf')):
                if 'models' not in results: results['models'] = []
                results['models'].append(v)
            find_all(v, search_keys, results)
    elif isinstance(obj, list):
        for item in obj:
            find_all(item, search_keys, results)

def analyze(path):
    print(f"\n--- Analysis for {path} ---")
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    results = {}
    # Common keys for steps and resolution
    keys = ['steps', 'width', 'height', 'frame_count', 'frames', 'denoise', 'batch_size']
    find_all(data, keys, results)
    
    for k, v in results.items():
        print(f"{k}: {list(set(str(i) for i in v))}")

    # Specific check for nodes widgets
    if 'nodes' in data:
        for node in data['nodes']:
            if node.get('type') == 'EmptyLTXVLatentVideo':
                print(f"EmptyLTXVLatentVideo: {node.get('widgets_values')}")
            if node.get('type') == 'LTXVScheduler':
                print(f"LTXVScheduler: {node.get('widgets_values')}")
            if node.get('type') == 'SamplerCustomAdvanced':
                print(f"SamplerCustomAdvanced: {node.get('widgets_values')}")
            if node.get('type') == 'UnetLoaderGGUF':
                print(f"UnetLoaderGGUF: {node.get('widgets_values')}")

analyze('F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME.json')
analyze('F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME_GUIDE.json')
