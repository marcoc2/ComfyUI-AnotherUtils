import os
import hashlib
import json
import numpy as np
import torch
from PIL import Image, ImageSequence
import folder_paths


class LoadGifFrames:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if f.lower().endswith('.gif')]
        return {
            "required": {
                "gif": (sorted(files), {"image_upload": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "INT", "INT")
    RETURN_NAMES = ("unique_frames", "frame_map", "unique_count", "total_frames")
    FUNCTION = "load_gif"
    CATEGORY = "image/loaders"
    TITLE = "Load GIF Frames (Raw)"

    def load_gif(self, gif):
        gif_path = folder_paths.get_annotated_filepath(gif)
        img = Image.open(gif_path)

        all_frames = []
        for frame in ImageSequence.Iterator(img):
            frame_rgba = frame.convert('RGBA')
            background = Image.new('RGB', frame_rgba.size, (255, 255, 255))
            background.paste(frame_rgba, mask=frame_rgba.split()[3])
            frame_np = np.array(background, dtype=np.float32) / 255.0
            all_frames.append(frame_np)

        # Deduplicate: hash each frame to find identical ones
        unique_frames = []
        frame_map = []        # frame_map[i] = index in unique_frames
        hash_to_index = {}

        for frame_np in all_frames:
            h = hashlib.md5(frame_np.tobytes()).hexdigest()
            if h not in hash_to_index:
                hash_to_index[h] = len(unique_frames)
                unique_frames.append(frame_np)
            frame_map.append(hash_to_index[h])

        unique_batch = torch.stack([torch.from_numpy(f) for f in unique_frames])  # [U, H, W, 3]

        return (unique_batch, json.dumps(frame_map), len(unique_frames), len(all_frames))

    @classmethod
    def IS_CHANGED(cls, gif):
        gif_path = folder_paths.get_annotated_filepath(gif)
        m = hashlib.sha256()
        with open(gif_path, 'rb') as f:
            m.update(f.read())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, gif):
        if not folder_paths.exists_annotated_filepath(gif):
            return f"GIF file not found: {gif}"
        return True


class RemapGifFrames:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "processed_frames": ("IMAGE",),
                "frame_map": ("STRING", {"forceInput": True}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("frames", "total_frames")
    FUNCTION = "remap"
    CATEGORY = "image/loaders"
    TITLE = "Remap GIF Frames"

    def remap(self, processed_frames, frame_map):
        mapping = json.loads(frame_map)  # e.g. [0, 0, 1, 2, 2, 1]

        unique_count = processed_frames.shape[0]
        required = max(mapping) + 1
        if unique_count < required:
            raise ValueError(
                f"RemapGifFrames: processed_frames tem {unique_count} frame(s), mas frame_map exige {required}. "
                f"O pipeline entre LoadGifFrames e RemapGifFrames deve preservar todos os {required} unique_frames como batch — "
                f"verifique se algum nó no meio está selecionando ou descartando frames."
            )

        # Reconstruct full sequence by indexing into processed_frames
        reordered = torch.stack([processed_frames[i] for i in mapping])  # [N, H, W, C]

        return (reordered, len(mapping))
