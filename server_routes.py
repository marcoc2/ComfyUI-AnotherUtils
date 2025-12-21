from server import PromptServer
from aiohttp import web
import os
import io
import base64
import json
from PIL import Image

routes = PromptServer.instance.routes

@routes.get("/another_utils/list_images")
async def list_images(request):
    if "directory" not in request.rel_url.query:
        return web.json_response({"error": "Missing directory parameter"}, status=400)
    
    directory = request.rel_url.query["directory"]
    
    if not os.path.isdir(directory):
        return web.json_response({"error": "Directory not found"}, status=404)

    # Structure check: Root has images, 'captions' subfolder has texts
    img_dir = directory
    cap_dir = os.path.join(directory, "captions")
    
    # We allow running without captions folder existing yet (maybe just viewing images), 
    # but strictly user asked for captions. Let's return what we find.
    
    valid_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')
    
    try:
        image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(valid_extensions)]
        image_files.sort()
    except Exception as e:
         return web.json_response({"error": str(e)}, status=500)

    files_data = []

    for filename in image_files:
        basename = os.path.splitext(filename)[0]
        
        # Caption
        caption_text = ""
        cap_path = os.path.join(cap_dir, basename + ".txt")
        if os.path.exists(cap_path):
            try:
                with open(cap_path, 'r', encoding='utf-8') as f:
                    caption_text = f.read().strip()
            except:
                caption_text = "[Error reading caption]"
        
        # Thumbnail (Generate on fly - caching would be better but keep simple for now)
        # For a large folder, this might be slow on first load.
        # We process reasonably fast.
        
        img_path = os.path.join(img_dir, filename)
        thumb_b64 = ""
        try:
            img = Image.open(img_path)
            thumb_size = (128, 128) # Smaller for list
            img.thumbnail(thumb_size, Image.Resampling.LANCZOS)
            
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=70)
            thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode("utf-8")
        except Exception as e:
            print(f"Error thumbing {filename}: {e}")
            continue

        files_data.append({
            "filename": filename,
            "basename": basename,
            "caption": caption_text,
            "thumbnail": thumb_b64
        })

    return web.json_response({"files": files_data})
