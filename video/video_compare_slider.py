import re
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont


class VideoCompareSlider:
    """
    Compares two videos (IMAGE batches in ComfyUI) using a sliding wipe bar.
    Highly optimized and memory-safe: processes frame-by-frame directly into
    a pre-allocated output tensor under torch.no_grad() to prevent memory spikes
    on long video batches (500+ frames).
    Optionally overlays Nvidia-style '{method} OFF' and '{method} ON' badges.
    """

    COLOR_MAP = {
        "white": (1.0, 1.0, 1.0),
        "black": (0.0, 0.0, 0.0),
        "red": (1.0, 0.0, 0.0),
        "green": (0.0, 1.0, 0.0),
        "blue": (0.0, 0.0, 1.0),
        "yellow": (1.0, 1.0, 0.0),
        "cyan": (0.0, 1.0, 1.0),
        "magenta": (1.0, 0.0, 1.0),
    }

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_a": ("IMAGE", {"tooltip": "Base video / 'Before' (IMAGE batch [B, H, W, C])"}),
                "video_b": ("IMAGE", {"tooltip": "Comparison video / 'After' (IMAGE batch [B, H, W, C])"}),
                "pause_count": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Number of times to pause playback and sweep the slider. 0 = continuous sweep across the video without pausing.",
                    },
                ),
                "custom_pause_frames": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Comma or space separated frame indices to pause on (e.g. '15, 30, 45'). Overrides pause_count if provided.",
                    },
                ),
                "pause_duration_frames": (
                    "INT",
                    {
                        "default": 60,
                        "min": 2,
                        "max": 1000,
                        "step": 1,
                        "tooltip": "Number of frames used for the slider sweep during each pause.",
                    },
                ),
                "sweep_mode": (
                    [
                        "ping_pong (0% -> 100% -> 0%)",
                        "ping_pong_rest (rest -> 100% -> 0% -> rest)",
                        "left_to_right (0% -> 100%)",
                        "right_to_left (100% -> 0%)",
                    ],
                    {"default": "ping_pong (0% -> 100% -> 0%)"},
                ),
                "line_width": (
                    "INT",
                    {
                        "default": 4,
                        "min": 0,
                        "max": 50,
                        "step": 1,
                        "tooltip": "Width of the vertical slider separator line in pixels (0 for no line).",
                    },
                ),
                "line_color": (
                    list(cls.COLOR_MAP.keys()),
                    {"default": "white"},
                ),
                "fps": (
                    "FLOAT",
                    {
                        "default": 24.0,
                        "min": 0.1,
                        "max": 120.0,
                        "step": 0.1,
                        "tooltip": "Frames per second for output video encoding.",
                    },
                ),
                "resting_position": (
                    "FLOAT",
                    {
                        "default": 0.5,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "Slider position (0.0=all A, 0.5=split, 1.0=all B) during normal video playback when pauses are used.",
                    },
                ),
                "method": (
                    "STRING",
                    {
                        "default": "",
                        "tooltip": "Method label prefix (e.g. 'RTX', 'DLSS', 'UPSCALER'). When provided, overlays '{method} OFF' in red on Before and '{method} ON' in green on After with white text and black outline.",
                    },
                ),
            },
            "optional": {
                "method_font_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.2,
                        "max": 5.0,
                        "step": 0.1,
                        "tooltip": "Font size multiplier for the method overlay badges.",
                    },
                ),
                "method_position": (
                    ["bottom_corners", "top_corners"],
                    {"default": "bottom_corners", "tooltip": "Placement of the ON/OFF badges."},
                ),
                "pause_frames_input": (
                    "*",
                    {"tooltip": "Optional upstream list or string of frame indices to pause on."},
                ),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    RETURN_NAMES = ("image", "fps")
    FUNCTION = "generate_slider_comparison"
    CATEGORY = "AnotherUtils/Video"

    def _parse_custom_frames(self, custom_str, custom_input, total_frames):
        """
        Parses custom pause frame inputs from string widget or optional upstream connection.
        Returns a sorted list of valid unique frame indices [0, total_frames - 1].
        """
        raw_items = []

        # Check optional upstream connection first
        if custom_input is not None:
            if isinstance(custom_input, (list, tuple, set)):
                raw_items.extend(custom_input)
            elif isinstance(custom_input, (int, float)):
                raw_items.append(int(custom_input))
            elif isinstance(custom_input, str):
                tokens = re.split(r"[,;\s]+", custom_input.strip())
                for t in tokens:
                    if t.strip():
                        raw_items.append(t.strip())

        # Check string widget input
        if custom_str and isinstance(custom_str, str) and custom_str.strip():
            tokens = re.split(r"[,;\s]+", custom_str.strip())
            for t in tokens:
                if t.strip():
                    raw_items.append(t.strip())

        valid_indices = []
        for item in raw_items:
            try:
                val = int(item)
                if 0 <= val < total_frames:
                    valid_indices.append(val)
            except (ValueError, TypeError):
                continue

        return sorted(list(set(valid_indices)))

    def _calculate_pause_frames(self, pause_count, total_frames):
        """
        Calculates equidistant pause frame indices based on pause_count.
        If pause_count = 1 -> index at 1/2 (halfway).
        If pause_count = 2 -> indices at 1/3 and 2/3 (divided into 3 parts).
        If pause_count = k -> indices at i * (total_frames - 1) / (k + 1) for i in 1..k.
        """
        if pause_count <= 0 or total_frames <= 0:
            return []

        parts = pause_count + 1
        indices = []
        for k in range(1, pause_count + 1):
            idx = int(round(k * (total_frames - 1) / parts))
            idx = max(0, min(total_frames - 1, idx))
            indices.append(idx)

        return sorted(list(set(indices)))

    def _get_sweep_progress(self, index, total_len, sweep_mode, resting_position=0.5):
        """
        Computes progress ratio [0.0, 1.0] for a given frame index in a sweep.
        """
        if total_len <= 1:
            return resting_position

        t = index / (total_len - 1)

        if "ping_pong_rest" in sweep_mode:
            # Smooth loop: resting_position -> 1.0 -> 0.0 -> resting_position
            r = max(0.0, min(1.0, resting_position))
            if t < 0.25:
                # [0, 0.25] -> from r to 1.0
                return r + (1.0 - r) * (t / 0.25)
            elif t < 0.75:
                # [0.25, 0.75] -> from 1.0 to 0.0
                return 1.0 - (t - 0.25) / 0.5
            else:
                # [0.75, 1.0] -> from 0.0 to r
                return (t - 0.75) / 0.25 * r
        elif "ping_pong" in sweep_mode:
            # 0.0 -> 1.0 -> 0.0
            return 1.0 - abs(2.0 * t - 1.0)
        elif "left_to_right" in sweep_mode:
            # 0.0 -> 1.0
            return t
        elif "right_to_left" in sweep_mode:
            # 1.0 -> 0.0
            return 1.0 - t
        else:
            return t

    def _get_font(self, font_size):
        """
        Loads a bold font with fallback support across operating systems.
        """
        font_candidates = [
            "arialbd.ttf",
            "arial.ttf",
            "segoeuib.ttf",
            "calibrib.ttf",
            "DejaVuSans-Bold.ttf",
            "DejaVuSans.ttf",
            "C:\\Windows\\Fonts\\arialbd.ttf",
            "C:\\Windows\\Fonts\\arial.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        ]
        for font_name in font_candidates:
            try:
                return ImageFont.truetype(font_name, font_size)
            except Exception:
                continue
        try:
            return ImageFont.load_default()
        except Exception:
            return None

    def _create_text_overlay(self, height, width, method_name, status, position, font_scale, device, dtype):
        """
        Pre-renders a single 2D text overlay (RGB and Alpha) once for all frames.
        Returns: (rgb_tensor [H, W, 3], alpha_tensor [H, W, 1])
        """
        if not method_name or not method_name.strip():
            return None, None

        base_font_size = max(16, int(height * 0.045 * font_scale))
        font = self._get_font(base_font_size)

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        stroke_width = max(1, int(base_font_size * 0.08))
        method_str = f"{method_name.strip()} "
        status_str = status.strip()

        # Measure text sizes
        if hasattr(draw, "textbbox") and font is not None:
            bbox_m = draw.textbbox((0, 0), method_str, font=font, stroke_width=stroke_width)
            m_w = bbox_m[2] - bbox_m[0]
            m_h = bbox_m[3] - bbox_m[1]
            bbox_s = draw.textbbox((0, 0), status_str, font=font, stroke_width=stroke_width)
            s_w = bbox_s[2] - bbox_s[0]
            s_h = bbox_s[3] - bbox_s[1]
        else:
            m_w = len(method_str) * int(base_font_size * 0.6)
            s_w = len(status_str) * int(base_font_size * 0.6)
            m_h = s_h = base_font_size

        total_w = m_w + s_w
        total_h = max(m_h, s_h)

        margin_x = int(width * 0.035)
        margin_y = int(height * 0.045)

        # Positioning: 'ON' (Video B) on the left, 'OFF' (Video A) on the right
        if status == "ON":
            pos_x = margin_x
        else:
            pos_x = max(0, width - margin_x - total_w)

        if "top" in position:
            pos_y = margin_y
        else:
            pos_y = max(0, height - margin_y - total_h)

        # Status color: Apple/Nvidia vibrant green & red
        status_color = (76, 217, 100) if status == "ON" else (255, 59, 48)

        # Draw method name (White with black stroke)
        draw.text(
            (pos_x, pos_y),
            method_str,
            font=font,
            fill=(255, 255, 255),
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0),
        )

        # Draw status (Green / Red with black stroke)
        draw.text(
            (pos_x + m_w, pos_y),
            status_str,
            font=font,
            fill=status_color,
            stroke_width=stroke_width,
            stroke_fill=(0, 0, 0),
        )

        overlay_np = np.array(overlay).astype(np.float32) / 255.0
        overlay_tensor = torch.from_numpy(overlay_np).to(device=device, dtype=dtype)  # [H, W, 4]

        rgb = overlay_tensor[..., :3]     # [H, W, 3]
        alpha = overlay_tensor[..., 3:4]   # [H, W, 1]

        return rgb, alpha

    def _prepare_frame(self, frame, target_h, target_w, overlay_rgb, overlay_alpha):
        """
        Extracts, resizes (if needed), and applies text overlay to a single frame [H, W, C].
        Zero unnecessary memory overhead.
        """
        # Ensure 3 channels
        if frame.shape[-1] == 4:
            f = frame[..., :3]
        else:
            f = frame

        # Resize single frame if dimensions differ
        h, w = f.shape[:2]
        if (h, w) != (target_h, target_w):
            f_perm = f.permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]
            f_res = F.interpolate(f_perm, size=(target_h, target_w), mode="bilinear", align_corners=False)
            f = f_res.squeeze(0).permute(1, 2, 0)     # [target_h, target_w, 3]

        # Apply overlay if present
        if overlay_alpha is not None and overlay_rgb is not None:
            f = f * (1.0 - overlay_alpha) + overlay_rgb * overlay_alpha

        return f

    def _render_split_into(self, out_tensor, out_idx, f_a, f_b, split_ratio, line_width, color_tensor, target_w):
        """
        Writes combined frame directly into pre-allocated output tensor in-place.
        """
        split_x = int(round(split_ratio * target_w))
        split_x = max(0, min(target_w, split_x))

        # Write left side (Video B) and right side (Video A)
        if split_x > 0:
            out_tensor[out_idx, :, :split_x, :] = f_b[:, :split_x, :]
        if split_x < target_w:
            out_tensor[out_idx, :, split_x:, :] = f_a[:, split_x:, :]

        # Draw vertical line
        if line_width > 0:
            half = line_width // 2
            x1 = max(0, split_x - half)
            x2 = min(target_w, split_x + half + (line_width % 2))
            if x2 > x1:
                out_tensor[out_idx, :, x1:x2, :] = color_tensor

    @torch.inference_mode()
    def generate_slider_comparison(
        self,
        video_a,
        video_b,
        pause_count,
        custom_pause_frames,
        pause_duration_frames,
        sweep_mode,
        line_width,
        line_color,
        fps,
        resting_position,
        method="",
        method_font_scale=1.0,
        method_position="bottom_corners",
        pause_frames_input=None,
    ):
        b_a, h_a, w_a, c_a = video_a.shape
        b_b, h_b, w_b, c_b = video_b.shape

        total_frames = min(b_a, b_b)
        if total_frames == 0:
            raise ValueError("Input videos must contain at least 1 frame.")

        target_h, target_w = h_a, w_a

        # Pre-render text overlays once (each is only ~H*W*4 float32, <10MB)
        overlay_a_rgb, overlay_a_alpha = self._create_text_overlay(
            target_h, target_w, method, "OFF", method_position, method_font_scale, video_a.device, video_a.dtype
        )
        overlay_b_rgb, overlay_b_alpha = self._create_text_overlay(
            target_h, target_w, method, "ON", method_position, method_font_scale, video_b.device, video_b.dtype
        )

        # Prepare line color tensor
        rgb_color = self.COLOR_MAP.get(line_color, (1.0, 1.0, 1.0))
        color_tensor = torch.tensor(
            rgb_color,
            dtype=video_a.dtype,
            device=video_a.device,
        ).reshape(1, 1, 3)

        # Determine pause frames
        custom_frames = self._parse_custom_frames(
            custom_pause_frames,
            pause_frames_input,
            total_frames,
        )

        if len(custom_frames) > 0:
            pause_frame_set = set(custom_frames)
        elif pause_count > 0:
            pause_frame_set = set(self._calculate_pause_frames(pause_count, total_frames))
        else:
            pause_frame_set = set()

        # Calculate exact total output frames
        if len(pause_frame_set) == 0:
            total_out_frames = total_frames
        else:
            num_pauses = len(pause_frame_set)
            total_out_frames = (total_frames - num_pauses) + num_pauses * pause_duration_frames

        # Pre-allocate output tensor directly in memory (zero reallocation / zero list duplication)
        output = torch.empty(
            (total_out_frames, target_h, target_w, 3),
            dtype=video_a.dtype,
            device=video_a.device,
        )

        out_idx = 0

        if len(pause_frame_set) == 0:
            # Mode A: Continuous sweep across the entire video duration without pauses
            for f in range(total_frames):
                f_a = self._prepare_frame(video_a[f], target_h, target_w, overlay_a_rgb, overlay_a_alpha)
                f_b = self._prepare_frame(video_b[f], target_h, target_w, overlay_b_rgb, overlay_b_alpha)
                split_ratio = self._get_sweep_progress(f, total_frames, sweep_mode, resting_position)
                self._render_split_into(output, out_idx, f_a, f_b, split_ratio, line_width, color_tensor, target_w)
                out_idx += 1
        else:
            # Mode B: Video playback with pauses where the slider sweeps while holding the frame
            for f in range(total_frames):
                f_a = self._prepare_frame(video_a[f], target_h, target_w, overlay_a_rgb, overlay_a_alpha)
                f_b = self._prepare_frame(video_b[f], target_h, target_w, overlay_b_rgb, overlay_b_alpha)

                if f in pause_frame_set:
                    # Freeze at frame f and sweep the slider
                    for p in range(pause_duration_frames):
                        split_ratio = self._get_sweep_progress(
                            p,
                            pause_duration_frames,
                            sweep_mode,
                            resting_position,
                        )
                        self._render_split_into(output, out_idx, f_a, f_b, split_ratio, line_width, color_tensor, target_w)
                        out_idx += 1
                else:
                    # Normal playback frame with slider at resting_position
                    self._render_split_into(output, out_idx, f_a, f_b, resting_position, line_width, color_tensor, target_w)
                    out_idx += 1

        return (output, float(fps))
