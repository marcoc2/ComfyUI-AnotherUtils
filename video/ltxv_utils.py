import torch

def resolve_frame_indices(indices, positions_str, mode, fps, total_pixel_frames, num_images):
    """Resolve frame indices from either INT list or positions string."""
    # Priority 1: indices input (from ImageListSampler etc.)
    if indices is not None and len(indices) > 0 and indices[0] is not None:
        if mode == "frames":
            return [int(i) for i in indices]
        elif mode == "seconds":
            return [
                -1 if i < 0 else min(round(float(i) / fps * fps), total_pixel_frames - 1)
                if False else int(i)  # indices are already frame numbers
                for i in indices
            ]
        elif mode == "percentage":
            return [
                -1 if i >= total_pixel_frames - 1
                else int(i)
                for i in indices
            ]
        # Default: treat indices as frame numbers directly
        return [int(i) for i in indices]

    # Priority 2: positions string
    if positions_str and positions_str.strip():
        return parse_positions(positions_str, mode, fps, total_pixel_frames)

    # Fallback: distribute evenly
    if num_images == 1:
        return [0]
    step = (total_pixel_frames - 1) / (num_images - 1)
    return [round(i * step) for i in range(num_images)]

def parse_positions(positions_str, mode, fps, total_pixel_frames):
    """Parse position string into frame indices."""
    raw = [s.strip() for s in positions_str.split(",") if s.strip()]
    frame_indices = []

    for val_str in raw:
        val = float(val_str)

        if mode == "frames":
            frame_indices.append(int(val))
        elif mode == "seconds":
            if val < 0:
                frame_indices.append(-1)
            else:
                frame_idx = round(val * fps)
                frame_idx = min(frame_idx, total_pixel_frames - 1)
                frame_indices.append(frame_idx)
        elif mode == "percentage":
            if val < 0 or val >= 100.0:
                frame_indices.append(-1)
            else:
                frame_idx = round(val / 100.0 * (total_pixel_frames - 1))
                frame_indices.append(frame_idx)

    return frame_indices

def parse_strengths(strengths_str, num_images, default_strength):
    """Parse per-image strengths or fill with default."""
    if not strengths_str or not strengths_str.strip():
        return [default_strength] * num_images

    raw = [s.strip() for s in strengths_str.split(",") if s.strip()]
    parsed = [max(0.0, min(1.0, float(s))) for s in raw]

    while len(parsed) < num_images:
        parsed.append(default_strength)

    return parsed[:num_images]

def flatten_images(images):
    """Flatten list or batch of images into a single B,H,W,C tensor."""
    if isinstance(images, list):
        frames = []
        for item in images:
            if isinstance(item, torch.Tensor):
                if len(item.shape) == 4:
                    for j in range(item.shape[0]):
                        frames.append(item[j:j+1])
                else:
                    frames.append(item.unsqueeze(0) if len(item.shape) == 3 else item)
        return torch.cat(frames, dim=0) if frames else images[0]
    return images
