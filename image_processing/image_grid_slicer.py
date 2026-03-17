class ImageGridSlicer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grid_x": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
                "grid_y": ("INT", {"default": 2, "min": 1, "max": 100, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_list",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "slice_image"
    CATEGORY = "AnotherUtils/image_processing"

    def slice_image(self, image, grid_x, grid_y):
        # image shape: [B, H, W, C]
        # Handle alpha channel (flatten with white background like LoadGifFrames)
        if image.shape[-1] == 4:
            alpha = image[:, :, :, 3:4]
            rgb = image[:, :, :, :3]
            # Standard alpha blending with white background: result = foreground * alpha + background * (1 - alpha)
            image = rgb * alpha + (1.0 - alpha)

        batch_size, height, width, _ = image.shape

        # Calculate tile dimensions
        tile_width = width // grid_x
        tile_height = height // grid_y
        output_images = []
        for b in range(batch_size):
            img = image[b] # [H, W, C]
            for y in range(grid_y):
                for x in range(grid_x):
                    start_x = x * tile_width
                    start_y = y * tile_height
                    # Ensure we don't go out of bounds if division isn't perfect
                    end_x = start_x + tile_width
                    end_y = start_y + tile_height
                    if x == grid_x - 1:
                        end_x = width
                    if y == grid_y - 1:
                        end_y = height
                    tile = img[start_y:end_y, start_x:end_x, :]
                    # ComfyUI expects [1, H, W, C] for lists of images
                    output_images.append(tile.unsqueeze(0))
        return (output_images,)
