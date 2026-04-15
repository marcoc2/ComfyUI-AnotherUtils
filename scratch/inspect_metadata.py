import json
from PIL import Image

image_path = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\custom_nodes\ComfyUI-AnotherUtils\reference\comfyui_output.png"
img = Image.open(image_path)
metadata = img.info

print("Keys in metadata:", metadata.keys())
if 'prompt' in metadata:
    prompt_json = json.loads(metadata['prompt'])
    
    # Try to find a sampler node
    sampler_nodes = [id for id, node in prompt_json.items() if 'KSampler' in node.get('class_type', '')]
    if not sampler_nodes:
         sampler_nodes = [id for id, node in prompt_json.items() if 'Sampler' in node.get('class_type', '')]
         
    positive_text = ""
    
    if sampler_nodes:
        sampler_id = sampler_nodes[0]
        sampler_node = prompt_json[sampler_id]
        positive_link = sampler_node.get('inputs', {}).get('positive')
        if positive_link and isinstance(positive_link, list):
            parent_node_id = str(positive_link[0])
            if parent_node_id in prompt_json:
                parent_node = prompt_json[parent_node_id]
                # If it's CLIPTextEncode, get text
                if 'text' in parent_node.get('inputs', {}):
                    positive_text = parent_node['inputs']['text']
                # Sometimes it's a reroute or something, but usually it's the node
    
    if not positive_text:
        # Fallback: look for any CLIPTextEncode and take the one that isn't empty?
        texts = []
        for id, node in prompt_json.items():
            if node.get('class_type') == 'CLIPTextEncode':
                text = node.get('inputs', {}).get('text', '')
                if text:
                    texts.append(text)
        if texts:
            positive_text = texts[0] # Just take the first one for now
            
    print(f"Extracted prompt: {positive_text}")
else:
    print("Metadata 'prompt' not found.")
