import json

def trace_sampler(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    nodes = {n['id']: n for n in data.get('nodes', [])}
    links = {l[0]: l for l in data.get('links', [])} # id, source_node, source_output, target_node, target_input, type
    
    # In 'FAST' workflow, the samplers are likely buried in 'definitions'
    # But let's look for any sampler-like node in the main flow
    print(f"\n--- Tracing in {path} ---")
    
    # Find nodes that look like samplers
    samplers = []
    for node_id, node in nodes.items():
        ntype = node.get('type')
        if ntype and ('Sampler' in ntype or '-Sampler' in ntype):
            samplers.append(node)
    
    # If no samplers in main flow, it's definitely in definitions
    if not samplers:
        print("No samplers in main flow. Checking definitions/subgraphs.")
        # Some workflows store subgraphs in 'extra' or 'definitions'
        # Let's just search the whole text for 'Sampler' inside the JSON
        pass

    # Let's search for the MultiImageLoader output
    loader = None
    for node_id, node in nodes.items():
        if node.get('type') == 'MultiImageLoader':
            loader = node
            break
    
    if loader:
        print(f"Found MultiImageLoader (ID {loader['id']})")
        # Find where its outputs go
        for link_id, link in links.items():
            if link[1] == loader['id']:
                target_node_id = link[3]
                target_node = nodes.get(target_node_id)
                print(f"  Output goes to: Node {target_node_id} ({target_node.get('type') if target_node else 'Unknown'})")

    # Let's search for LTXVMultiGuide or similar in the WHOLE file (raw text)
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        if 'LTXVMultiGuide' in content:
            print("Found 'LTXVMultiGuide' in the file!")
        else:
            print("Did NOT find 'LTXVMultiGuide' in the file.")
        
        if 'LTXVConcatAVLatent' in content:
            print("Found 'LTXVConcatAVLatent' in the file!")

trace_sampler('F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME.json')
trace_sampler('F:/AppsCrucial/ComfyUI_phoenix3/ComfyUI/custom_nodes/reference/multiframe/LTX23_MULTIFRAME_GUIDE.json')
