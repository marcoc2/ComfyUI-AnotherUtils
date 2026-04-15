import json
import sys

# Ensure stdout handles UTF-8
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_samplers(workflow):
    samplers = []
    # ComfyUI workflows can be in 'prompt' format (API) or 'workflow' format (UI)
    # The JSONs from ComfyUI are usually the UI format OR the prompt format.
    # If it's the UI format, it has 'nodes'. If it's the API format, it's just a dict of nodes.
    
    nodes = workflow.get('nodes', [])
    if not nodes:
        # Check if it's the API format
        for node_id, node_data in workflow.items():
            if isinstance(node_data, dict) and 'class_type' in node_data:
                samplers.append({
                    'id': node_id,
                    'type': node_data['class_type'],
                    'inputs': node_data.get('inputs', {})
                })
    else:
        for node in nodes:
            class_type = node.get('type')
            if class_type and ('Sampler' in class_type or 'Sampler' in node.get('comfyClass', '')):
                samplers.append({
                    'id': node.get('id'),
                    'type': class_type,
                    'properties': node.get('properties'),
                    'widgets_values': node.get('widgets_values'),
                    'inputs': node.get('inputs')
                })
    return samplers

def analyze_workflow(path):
    wf = load_json(path)
    print(f"\n--- Detailed Analysis for {path} ---")
    
    nodes = wf.get('nodes', [])
    if not nodes:
        # API format
        for node_id, node_data in wf.items():
            if isinstance(node_data, dict) and 'class_type' in node_data:
                print(f"Node {node_id}: {node_data['class_type']}")
                print(f"  Inputs: {node_data.get('inputs')}")
    else:
        # UI format
        for node in nodes:
            print(f"Node {node.get('id')}: {node.get('type')}")
            print(f"  Widgets: {node.get('widgets_values')}")
            # Also try to print inputs link count
            inputs = node.get('inputs', [])
            if inputs:
                print(f"  Inputs count: {len(inputs)}")

if __name__ == "__main__":
    analyze_workflow("F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME.json")
    analyze_workflow("F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME_GUIDE.json")
