class ImageListSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "count": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("sampled_images", "indices")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "sample_images"
    CATEGORY = "AnotherUtils/logic"

    def sample_images(self, images, count):
        # When INPUT_IS_LIST is True, all inputs arrive as lists
        # Unwrap parameters that should be single values
        cnt = count[0] if isinstance(count, list) else count
        
        if not images:
            return ([],)
            
        # Flatten list/batches into individual frames
        import torch
        all_frames = []
        for item in images:
            if isinstance(item, torch.Tensor):
                if len(item.shape) == 4: # [B, H, W, C]
                    for i in range(item.shape[0]):
                        all_frames.append(item[i:i+1])
                else: 
                    all_frames.append(item.unsqueeze(0) if len(item.shape) == 3 else item)
            elif isinstance(item, list):
                all_frames.extend(item)
        
        total = len(all_frames)
        if cnt <= 0:
            return ([], [])
        
        if cnt >= total:
            return (all_frames, list(range(total)))

        # Calculate equally spaced indices
        step = total / cnt
        indices = [int(i * step) for i in range(cnt)]
        
        sampled = [all_frames[i] for i in indices]
        
        return (sampled, indices)
