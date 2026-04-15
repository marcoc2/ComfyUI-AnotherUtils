import math
import random
import torch
import torch.nn.functional as F
import node_helpers

# --- Easing Functions ---
def lerp(a, b, t):
    return a + (b - a) * t

def ease_in(t):
    return t * t

def ease_out(t):
    return t * (2 - t)

def ease_in_out(t):
    return t * t * (3.0 - 2.0 * t)

EASING_FUNCTIONS = {
    "linear": lambda t: t,
    "ease-in": ease_in,
    "ease-out": ease_out,
    "ease-in-out": ease_in_out,
}

# --- Direction Vectors (unit) ---
DIRECTIONS_8 = {
    "N":  ( 0.0, -1.0),
    "NE": ( 0.7071, -0.7071),
    "E":  ( 1.0,  0.0),
    "SE": ( 0.7071,  0.7071),
    "S":  ( 0.0,  1.0),
    "SW": (-0.7071,  0.7071),
    "W":  (-1.0,  0.0),
    "NW": (-0.7071, -0.7071),
}

DIRECTIONS_4 = {k: v for k, v in DIRECTIONS_8.items() if k in ("N", "E", "S", "W")}


class AnotherTransformOrchestrator:
    """
    Divides total frames into segments of N frames each.
    For each segment, picks a direction (random or sequential) and generates
    interpolated TRANSFORM_DATA — compatible with AnotherAnimatedCompositeMasked.
    Canvas dimensions are derived from the background image input.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "background": ("IMAGE",),
                "frames_per_segment": ("INT", {"default": 64, "min": 1, "max": 100000, "step": 1}),
                "total_frames": ("INT", {"default": 512, "min": 1, "max": 100000, "step": 1}),
                "travel_percent": ("FLOAT", {"default": 0.1, "min": 0.01, "max": 1.0, "step": 0.01,
                                             "tooltip": "How far from center toward edge. 0.1 = subtle movement, 1.0 = full edge"}),
                "start_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "end_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "direction_mode": (["random_8dir", "random_4dir", "sequential_8dir", "sequential_4dir"],
                                   {"default": "random_8dir"}),
                "direction_flow": (["center_to_edge", "edge_to_center", "random"],
                                   {"default": "center_to_edge",
                                    "tooltip": "center_to_edge: starts center moves out. edge_to_center: starts offset moves to center. random: coin flip per segment"}),
                "easing_mode": (["linear", "ease-in", "ease-out", "ease-in-out"], {"default": "ease-in-out"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
            }
        }

    RETURN_TYPES = ("TRANSFORM_DATA",)
    RETURN_NAMES = ("transform_data",)
    FUNCTION = "generate"
    CATEGORY = "AnotherUtils/animation"

    def generate(self, background, frames_per_segment, total_frames, travel_percent,
                 start_scale, end_scale, direction_mode, direction_flow, easing_mode, seed):
        rng = random.Random(seed)
        ease_fn = EASING_FUNCTIONS.get(easing_mode, lambda t: t)

        canvas_h, canvas_w = background.shape[1], background.shape[2]
        center_x = canvas_w / 2.0
        center_y = canvas_h / 2.0
        max_travel_x = center_x * travel_percent
        max_travel_y = center_y * travel_percent

        is_8dir = "8dir" in direction_mode
        is_sequential = "sequential" in direction_mode
        dir_pool = list(DIRECTIONS_8.values()) if is_8dir else list(DIRECTIONS_4.values())

        num_segments = math.ceil(total_frames / frames_per_segment)
        data = []

        for seg_idx in range(num_segments):
            seg_start = seg_idx * frames_per_segment
            seg_end = min(seg_start + frames_per_segment, total_frames)
            seg_len = seg_end - seg_start

            # Pick direction
            if is_sequential:
                dx, dy = dir_pool[seg_idx % len(dir_pool)]
            else:
                dx, dy = rng.choice(dir_pool)

            edge_x = center_x + dx * max_travel_x
            edge_y = center_y + dy * max_travel_y

            # Pick flow direction
            if direction_flow == "center_to_edge":
                flow_out = True
            elif direction_flow == "edge_to_center":
                flow_out = False
            else:
                flow_out = rng.choice([True, False])

            if flow_out:
                sx, sy = center_x, center_y
                ex, ey = edge_x, edge_y
                s_scale, e_scale = start_scale, end_scale
            else:
                sx, sy = edge_x, edge_y
                ex, ey = center_x, center_y
                s_scale, e_scale = end_scale, start_scale

            for fi in range(seg_len):
                t = fi / (seg_len - 1) if seg_len > 1 else 1.0
                t_eased = ease_fn(t)
                data.append({
                    "x": lerp(sx, ex, t_eased),
                    "y": lerp(sy, ey, t_eased),
                    "scale": lerp(s_scale, e_scale, t_eased),
                })

        return (data,)

class AnotherTransformKeyframes:
    """
    Generates interpolation values (X, Y, Scale) over a specified number of frames.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "start_x": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "end_x": ("INT", {"default": 100, "min": -8192, "max": 8192, "step": 1}),
                "start_y": ("INT", {"default": 0, "min": -8192, "max": 8192, "step": 1}),
                "end_y": ("INT", {"default": 100, "min": -8192, "max": 8192, "step": 1}),
                "start_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "end_scale": ("FLOAT", {"default": 1.0, "min": 0.01, "max": 100.0, "step": 0.01}),
                "frames_count": ("INT", {"default": 16, "min": 1, "max": 100000, "step": 1}),
                "easing_mode": (["linear", "ease-in", "ease-out", "ease-in-out"], {"default": "ease-in-out"}),
            }
        }

    RETURN_TYPES = ("TRANSFORM_DATA",)
    RETURN_NAMES = ("transform_data",)
    FUNCTION = "generate"
    CATEGORY = "AnotherUtils/animation"

    def generate(self, start_x, end_x, start_y, end_y, start_scale, end_scale, frames_count, easing_mode):
        ease_fn = EASING_FUNCTIONS.get(easing_mode, lambda t: t)
        frames_count = max(1, frames_count)
        data = []

        for i in range(frames_count):
            # Normalize t between 0 and 1
            t = i / (frames_count - 1) if frames_count > 1 else 1.0
            t_eased = ease_fn(t)

            cur_x = lerp(start_x, end_x, t_eased)
            cur_y = lerp(start_y, end_y, t_eased)
            cur_scale = lerp(start_scale, end_scale, t_eased)

            data.append({
                "x": cur_x,
                "y": cur_y,
                "scale": cur_scale
            })

        return (data,)


class AnotherAnimatedCompositeMasked:
    """
    Composites a batch of source images onto destination backgrounds dynamically.
    Instead of holding massive tensors in VRAM/CPU memory, it composites frame by frame.
    Also handles dynamic translation and scaling (zooming) injected via transform_data.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "destination": ("IMAGE",),
                "source": ("IMAGE",),
                "anchor_mode": (["top-left", "center"], {"default": "center"}),
            },
            "optional": {
                "mask": ("MASK",),
                "transform_data": ("TRANSFORM_DATA",),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"
    CATEGORY = "AnotherUtils/animation"

    def execute(self, destination, source, anchor_mode, mask=None, transform_data=None):
        destination, source = node_helpers.image_alpha_fix(destination, source)
        
        # move to [B, C, H, W] for standard image operations
        destination = destination.clone().movedim(-1, 1)
        source = source.movedim(-1, 1)

        batch_size = max(destination.shape[0], source.shape[0])
        
        # Ensure mask sizing matches
        if mask is not None:
            if mask.shape[0] < batch_size:
                mask = mask.repeat(batch_size // mask.shape[0] + 1, 1, 1)[:batch_size]

        if transform_data and len(transform_data) < batch_size:
            # Pad the transform data with the last frame if it's shorter than the batch
            last_td = transform_data[-1]
            transform_data.extend([last_td] * (batch_size - len(transform_data)))

        out = []
        dest_h, dest_w = destination.shape[2], destination.shape[3]

        for i in range(batch_size):
            dest_frame = destination[i:i+1].clone() # [1, C, H, W]
            src_frame = source[i % source.shape[0]:i % source.shape[0] + 1] # [1, C, H, W]
            
            mask_frame = None
            if mask is not None:
                mask_frame = mask[i % mask.shape[0]:i % mask.shape[0] + 1]
                if mask_frame.ndim < src_frame.ndim:
                    mask_frame = mask_frame.unsqueeze(1) # [1, 1, H, W]
            else:
                mask_frame = torch.ones((1, 1, src_frame.shape[2], src_frame.shape[3]), dtype=src_frame.dtype, device=src_frame.device)

            # Retrieve coordinates and scale
            if transform_data is not None:
                td = transform_data[i]
                cur_x = int(td["x"])
                cur_y = int(td["y"])
                cur_scale = float(td["scale"])
            else:
                cur_x, cur_y, cur_scale = 0, 0, 1.0

            # Scale source and mask if needed
            if cur_scale != 1.0 and cur_scale > 0:
                new_h = max(1, int(src_frame.shape[2] * cur_scale))
                new_w = max(1, int(src_frame.shape[3] * cur_scale))
                
                # Check for enormous scalings to avoid OOM
                if new_h <= 16384 and new_w <= 16384:
                    src_frame = F.interpolate(src_frame, size=(new_h, new_w), mode="bilinear", align_corners=False)
                    mask_frame = F.interpolate(mask_frame, size=(new_h, new_w), mode="bilinear", align_corners=False)

            src_h, src_w = src_frame.shape[2], src_frame.shape[3]

            # Adjust coords based on anchor mode
            if anchor_mode == "center":
                top = cur_y - src_h // 2
                left = cur_x - src_w // 2
            else:
                top = cur_y
                left = cur_x

            bottom = top + src_h
            right = left + src_w

            # Calculate safe bounds for compositing onto the canvas
            # This ensures we don't try to paste outside the destination bounds
            canvas_top = max(0, top)
            canvas_left = max(0, left)
            canvas_bottom = min(dest_h, bottom)
            canvas_right = min(dest_w, right)

            if canvas_top < canvas_bottom and canvas_left < canvas_right:
                # Source crop bounds (what part of the source gets pasted)
                src_top = max(0, -top)
                src_left = max(0, -left)
                src_bottom = src_top + (canvas_bottom - canvas_top)
                src_right = src_left + (canvas_right - canvas_left)

                sliced_mask = mask_frame[..., src_top:src_bottom, src_left:src_right]
                inverse_mask = 1.0 - sliced_mask
                
                src_portion = src_frame[..., src_top:src_bottom, src_left:src_right] * sliced_mask
                dest_portion = dest_frame[..., canvas_top:canvas_bottom, canvas_left:canvas_right] * inverse_mask
                
                dest_frame[..., canvas_top:canvas_bottom, canvas_left:canvas_right] = src_portion + dest_portion

            out.append(dest_frame)

        output = torch.cat(out, dim=0).movedim(1, -1) # back to [B, H, W, C]
        return (output,)
