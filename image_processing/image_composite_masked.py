import torch
from comfy_extras.nodes_mask import composite
import node_helpers

class AnotherImageCompositeMasked:
    """
    AnotherImageCompositeMasked
    A memory-safe, batch-aware version of the native ImageCompositeMasked node.
    It loops through batch elements one by one to avoid massive memory spikes 
    (OOM 'not enough memory' allocated to DefaultCPUAllocator) when processing large image lists.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "x": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "y": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 1}),
                "resize_source": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "AnotherUtils/image_processing"

    def execute(self, destination, source, x, y, resize_source, mask=None):
        destination, source = node_helpers.image_alpha_fix(destination, source)
        destination = destination.clone().movedim(-1, 1)
        source = source.movedim(-1, 1)

        batch_size = max(destination.shape[0], source.shape[0])
        
        # Expand mask to batch size if needed
        if mask is not None:
            if mask.shape[0] < batch_size:
                mask = mask.repeat(batch_size // mask.shape[0] + 1, 1, 1)[:batch_size]
        
        # Prepare output tensor
        out = []

        # Iterate one frame at a time to prevent CPU OOM allocation spikes
        for i in range(batch_size):
            dest_frame = destination[i:i+1] # [1, C, H, W]
            
            # handle loop/repeat around if lengths differ
            src_frame = source[i % source.shape[0]:i % source.shape[0] + 1]
            
            mask_frame = None
            if mask is not None:
                mask_frame = mask[i % mask.shape[0]:i % mask.shape[0] + 1]

            # Use native comfy composite, but on a single element
            res_frame = composite(dest_frame, src_frame, x, y, mask_frame, 1, resize_source)
            out.append(res_frame)

        # Re-stack and fix dimension order
        output = torch.cat(out, dim=0).movedim(1, -1)
        
        return (output,)
