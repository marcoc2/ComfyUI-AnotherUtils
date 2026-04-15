class ImageListSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "count": ("INT", {"default": 3, "min": 1, "max": 10000, "step": 1}),
                "target_frames": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1, 
                                          "tooltip": "If > 0, outputs indices scaled to this length. Example: Use target_frames=97 for LTXV video generation."}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("sampled_images", "indices")
    INPUT_IS_LIST = True
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "sample_images"
    CATEGORY = "AnotherUtils/logic"

    def sample_images(self, images, count, target_frames=0):
        # Unwrap parameters that should be single values
        cnt = count[0] if isinstance(count, list) else count
        tgt_frames = target_frames[0] if isinstance(target_frames, list) else target_frames
        
        if not images:
            return ([], [])
            
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
            
        if cnt == 1:
            return ([all_frames[0]], [0])

        # New math: Anchors to edges. 0 to total-1
        step_img = (total - 1) / (cnt - 1) if (cnt - 1) > 0 else 0
        img_indices = [int(round(i * step_img)) for i in range(cnt)]
        
        # Clamp to avoid rounding out-of-bounds just in case
        img_indices = [max(0, min(total - 1, idx)) for idx in img_indices]
        
        # Determine output indices (scaled to target_frames if provided)
        if tgt_frames is not None and tgt_frames > 0:
            step_tgt = (tgt_frames - 1) / (cnt - 1) if (cnt - 1) > 0 else 0
            out_indices = [int(round(i * step_tgt)) for i in range(cnt)]
            out_indices = [max(0, min(tgt_frames - 1, idx)) for idx in out_indices]
        else:
            out_indices = img_indices
            
        sampled = [all_frames[i] for i in img_indices]
        
        return (sampled, out_indices)
