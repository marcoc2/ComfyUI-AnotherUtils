class ImageListSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "count": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("sampled_images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "sample_images"
    CATEGORY = "AnotherUtils/logic"

    def sample_images(self, images, count):
        if not isinstance(images, list):
            # If it's a batch tensor [B, H, W, C], convert to list because logic expects list input
            # However, if it's already a list of tokens, we just use it.
            # In ComfyUI, if INPUT is simple "IMAGE", it might be a single tensor.
            # But if we were called with a list, 'images' is a list.
            pass
            
        total = len(images)
        if count <= 0:
            return ([],)
        
        if count >= total:
            return (images,)

        # Calculate equally spaced indices
        # Example: total=9, count=3
        # step = 9 / 3 = 3.0
        # i=0 -> 0
        # i=1 -> 3
        # i=2 -> 6
        step = total / count
        indices = [int(i * step) for i in range(count)]
        
        sampled = [images[i] for i in indices]
        
        return (sampled,)
