import torch


class BatchToImageList:
    """Splits an IMAGE batch [N,H,W,C] into a list of N individual frames.
    ComfyUI will iterate over the list, running each frame through the pipeline
    independently — avoiding RAM spikes from large batches in VAE encode/decode.
    Use ImageListToBatch+ downstream to collect results back into a batch.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "split"
    CATEGORY = "image"
    TITLE = "Batch to Image List"

    def split(self, images):
        # images: [N, H, W, C] → list of N tensors [1, H, W, C]
        return ([images[i:i+1] for i in range(images.shape[0])],)
