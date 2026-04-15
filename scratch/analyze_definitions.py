import json

def analyze_definitions(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    definitions = data.get('definitions', {})
    print(f"Found {len(definitions)} definitions.")
    
    for def_id, def_content in definitions.items():
        print(f"\n--- Definition ID: {def_id} ---")
        # Definitions usually contain their own 'nodes' and 'links'
        inner_nodes = def_content.get('nodes', [])
        print(f"Number of inner nodes: {len(inner_nodes)}")
        
        node_types = {}
        for node in inner_nodes:
            ntype = node.get('type')
            node_types[ntype] = node_types.get(ntype, 0) + 1
            
            # Check for samplers or guide nodes
            if ntype and ('Sampler' in ntype or 'Guide' in ntype or 'Inpaint' in ntype or 'Conditioning' in ntype):
                print(f"  Interesting Node found: {ntype}")
                if node.get('widgets_values'):
                    print(f"    Widgets: {node.get('widgets_values')}")
        
        print(f"Types summary: {node_types}")

analyze_definitions('F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME.json')
