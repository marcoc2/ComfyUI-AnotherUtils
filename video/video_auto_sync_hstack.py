import torch
import torch.nn.functional as F


class VideoAutoSyncHStack:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image1": ("IMAGE",),
                "fps1": (
                    "FLOAT",
                    {"default": 8.0, "min": 0.1, "max": 120.0, "step": 0.01},
                ),
                "image2": ("IMAGE",),
                "fps2": (
                    "FLOAT",
                    {"default": 48.0, "min": 0.1, "max": 120.0, "step": 0.01},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "fps")
    FUNCTION = "sync_and_stack"
    CATEGORY = "AnotherUtils/Video"

    def sync_and_stack(self, image1, fps1, image2, fps2):
        # Image shapes in ComfyUI: [B, H, W, C]
        b1, h1, w1, c1 = image1.shape
        b2, h2, w2, c2 = image2.shape

        target_fps = max(fps1, fps2)

        # Max height, ensuring it's an even number
        target_height = max(h1, h2)
        if target_height % 2 != 0:
            target_height += 1

        # Calculate durations and determine the shortest one
        duration1 = b1 / fps1
        duration2 = b2 / fps2
        target_duration = min(duration1, duration2)

        # Total frames for the final output
        target_b = int(target_duration * target_fps)
        if target_b == 0:
            target_b = 1

        # Determine target widths to match target height maintaining aspect ratio
        target_w1 = int(w1 * (target_height / h1))
        if target_w1 % 2 != 0:
            target_w1 += 1

        target_w2 = int(w2 * (target_height / h2))
        if target_w2 % 2 != 0:
            target_w2 += 1

        # Permute to [B, C, H, W] for interpolation
        img1_t = image1.permute(0, 3, 1, 2)
        img2_t = image2.permute(0, 3, 1, 2)

        # Resize if dimensions differ from target
        if h1 != target_height or w1 != target_w1:
            img1_t = F.interpolate(
                img1_t,
                size=(target_height, target_w1),
                mode="bilinear",
                align_corners=False,
            )
        if h2 != target_height or w2 != target_w2:
            img2_t = F.interpolate(
                img2_t,
                size=(target_height, target_w2),
                mode="bilinear",
                align_corners=False,
            )

        # Precompute indices for FPS matching without re-encoding
        # Similar to fps filter in ffmpeg duplicating/dropping frames
        indices1 = []
        indices2 = []
        for i in range(target_b):
            t = i / target_fps
            idx1 = min(int(t * fps1), b1 - 1)
            idx2 = min(int(t * fps2), b2 - 1)
            indices1.append(idx1)
            indices2.append(idx2)

        # Gather the frames based on the synchronized indices
        synced_img1 = img1_t[indices1]  # [target_b, C, target_height, target_w1]
        synced_img2 = img2_t[indices2]  # [target_b, C, target_height, target_w2]

        # Stack them side by side (Concatenate along Width dimension)
        stacked = torch.cat((synced_img1, synced_img2), dim=3)

        # Permute back to [B, H, W, C]
        result = stacked.permute(0, 2, 3, 1)

        return (result, float(target_fps))
