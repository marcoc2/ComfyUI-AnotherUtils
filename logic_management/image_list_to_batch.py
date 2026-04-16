import torch
import torch.nn.functional as F
import comfy.utils
import io
from PIL import Image
import numpy as np

class ImageListToBatch:
    """
    Takes a Python List of images (e.g. from ImageListSampler) and converts it 
    into a single Batched Tensor [N, H, W, C] to be compatible with nodes 
    like LTXSequencer ('multi_input').
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "interpolation": (["lanczos", "nearest", "bilinear", "bicubic", "area", "nearest-exact"],),
                "resize_method": (["keep proportion", "stretch", "pad", "crop"],),
                "multiple_of": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "img_compression": ("INT", {"default": 18, "min": 0, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("multi_output", )
    INPUT_IS_LIST = True
    FUNCTION = "convert"
    CATEGORY = "AnotherUtils/logic"

    def resize_image(self, image, width, height, resize_method="keep proportion", interpolation="nearest", multiple_of=0):
        MAX_RESOLUTION = 8192
        _, oh, ow, _ = image.shape
        x = y = x2 = y2 = 0
        pad_left = pad_right = pad_top = pad_bottom = 0

        if multiple_of > 1:
            width = width - (width % multiple_of)
            height = height - (height % multiple_of)

        if resize_method == 'keep proportion' or resize_method == 'pad':
            if width == 0 and oh < height:
                width = MAX_RESOLUTION
            elif width == 0 and oh >= height:
                width = ow

            if height == 0 and ow < width:
                height = MAX_RESOLUTION
            elif height == 0 and ow >= width:
                height = oh

            ratio = min(width / ow, height / oh)
            new_width = round(ow * ratio)
            new_height = round(oh * ratio)

            if resize_method == 'pad':
                pad_left = (width - new_width) // 2
                pad_right = width - new_width - pad_left
                pad_top = (height - new_height) // 2
                pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
            
        elif resize_method == 'crop':
            width = width if width > 0 else ow
            height = height if height > 0 else oh

            ratio = max(width / ow, height / oh)
            new_width = round(ow * ratio)
            new_height = round(oh * ratio)
            x = (new_width - width) // 2
            y = (new_height - height) // 2
            x2 = x + width
            y2 = y + height
            if x2 > new_width:
                x -= (x2 - new_width)
            if x < 0:
                x = 0
            if y2 > new_height:
                y -= (y2 - new_height)
            if y < 0:
                y = 0
            width = new_width
            height = new_height
            
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh

        # Always apply resize logic
        outputs = image.permute(0, 3, 1, 2)

        if interpolation == "lanczos":
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)

        if resize_method == 'pad':
            if pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0:
                outputs = F.pad(outputs, (pad_left, pad_right, pad_top, pad_bottom), value=0)

        outputs = outputs.permute(0, 2, 3, 1)

        if resize_method == 'crop':
            if x > 0 or y > 0 or x2 > 0 or y2 > 0:
                outputs = outputs[:, y:y2, x:x2, :]

        if multiple_of > 1 and (outputs.shape[2] % multiple_of != 0 or outputs.shape[1] % multiple_of != 0):
            width = outputs.shape[2]
            height = outputs.shape[1]
            x = (width % multiple_of) // 2
            y = (height % multiple_of) // 2
            x2 = width - ((width % multiple_of) - x)
            y2 = height - ((height % multiple_of) - y)
            outputs = outputs[:, y:y2, x:x2, :]
        
        outputs = torch.clamp(outputs, 0, 1)
        return outputs

    def convert(self, images, width, height, interpolation, resize_method, multiple_of, img_compression):
        # Unwrap list inputs (INPUT_IS_LIST=True)
        width = width[0] if isinstance(width, list) else width
        height = height[0] if isinstance(height, list) else height
        interpolation = interpolation[0] if isinstance(interpolation, list) else interpolation
        resize_method = resize_method[0] if isinstance(resize_method, list) else resize_method
        multiple_of = multiple_of[0] if isinstance(multiple_of, list) else multiple_of
        img_compression = img_compression[0] if isinstance(img_compression, list) else img_compression
        
        if not images:
            empty = torch.zeros((1, 64, 64, 3))
            return (empty,)
            
        # Unwrap list of lists if passed strangely
        all_frames = []
        for item in images:
            if isinstance(item, torch.Tensor):
                if len(item.shape) == 4:
                    for i in range(item.shape[0]):
                        all_frames.append(item[i:i+1])
                else: 
                    all_frames.append(item.unsqueeze(0) if len(item.shape) == 3 else item)
            elif isinstance(item, list):
                all_frames.extend(item)
                
        if not all_frames:
            empty = torch.zeros((1, 64, 64, 3))
            return (empty,)

        results = []
        for img in all_frames:
            # Apply Advanced Resize
            img = self.resize_image(img, width, height, resize_method, interpolation, multiple_of)

            # Compression (Applied after resize to accurately maintain the effect)
            if img_compression > 0:
                img_np = (img[0].numpy() * 255).clip(0, 255).astype(np.uint8)
                img_pil = Image.fromarray(img_np)
                img_byte_arr = io.BytesIO()
                img_pil.save(img_byte_arr, format="JPEG", quality=max(1, 100 - img_compression))
                img_pil = Image.open(img_byte_arr)
                img = torch.from_numpy(np.array(img_pil).astype(np.float32) / 255.0)[None,]

            results.append(img)

        # Check dimension consistency to form a batch
        first_shape = results[0].shape
        all_same_shape = all(r.shape == first_shape for r in results)
        
        if all_same_shape:
            multi_output = torch.cat(results, dim=0)
        else:
            print("[ImageListToBatch] Warning: Images have different dimensions. Padding batch as zero tensor.")
            multi_output = torch.zeros((1, 64, 64, 3))

        return (multi_output,)



