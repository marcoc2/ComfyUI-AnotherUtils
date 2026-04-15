import json

f_fast = 'F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME.json'
with open(f_fast, 'r', encoding='utf-8') as f:
    data = json.load(f)

print("--- LTXVConcatAVLatent Node Details ---")
for node in data.get('nodes', []):
    if node.get('type') == 'LTXVConcatAVLatent':
        print(f"Node ID: {node.get('id')}")
        print(f"Inputs: {node.get('inputs')}")
        print(f"Widgets: {node.get('widgets_values')}")
        
print("\n--- Links for LTXVConcatAVLatent ---")
links = {l[0]: l for l in data.get('links', [])}
for l_id, l in links.items():
    # link format: [id, origin_node, origin_slot, target_node, target_slot, type]
    if l[3] == 4 or l[1] == 4: # Assuming Node ID 4 is the one I found earlier
        print(l)
