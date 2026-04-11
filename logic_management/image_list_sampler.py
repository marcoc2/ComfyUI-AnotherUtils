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
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "sample_images"
    CATEGORY = "AnotherUtils/logic"

    def sample_images(self, images, count):
        # When INPUT_IS_LIST is True, all inputs arrive as lists
        # Unwrap parameters that should be single values
        cnt = count[0] if isinstance(count, list) else count
        
        if not images:
            return ([],)
            
        total = len(images)
        if cnt <= 0:
            return ([],)
        
        if cnt >= total:
            return (images,)

        # Calculate equally spaced indices
        # Example: total=9, count=3
        # step = 9 / 3 = 3.0
        # i=0 -> 0
        # i=1 -> 3
        # i=2 -> 6
        step = total / cnt
        indices = [int(i * step) for i in range(cnt)]
        
        sampled = [images[i] for i in indices]
        
        return (sampled,)
