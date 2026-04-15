import random
import torch


class AnotherCameraSwitcher:
    """
    Multi-camera director node. Receives up to 8 video inputs (IMAGE batches)
    and alternates between them in equal segments, simulating camera cuts.
    All input videos must have the same resolution and at least as many frames
    as the longest cut plan requires.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_1": ("IMAGE",),
                "switch_every_seconds": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 300.0, "step": 0.1,
                                                   "tooltip": "Cut to a different camera every N seconds"}),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0, "step": 0.1,
                                  "tooltip": "Frames per second of the input videos"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF, "step": 1}),
                "avoid_repeat": ("BOOLEAN", {"default": True,
                                             "tooltip": "Avoid picking the same camera twice in a row"}),
            },
            "optional": {
                "video_2": ("IMAGE",),
                "video_3": ("IMAGE",),
                "video_4": ("IMAGE",),
                "video_5": ("IMAGE",),
                "video_6": ("IMAGE",),
                "video_7": ("IMAGE",),
                "video_8": ("IMAGE",),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING",)
    RETURN_NAMES = ("video", "cut_log",)
    FUNCTION = "execute"
    CATEGORY = "AnotherUtils/video"

    def execute(self, video_1, switch_every_seconds, fps, seed, avoid_repeat,
                video_2=None, video_3=None, video_4=None,
                video_5=None, video_6=None, video_7=None, video_8=None):

        # Collect all connected videos
        videos = [video_1]
        for v in [video_2, video_3, video_4, video_5, video_6, video_7, video_8]:
            if v is not None:
                videos.append(v)

        num_cameras = len(videos)
        total_frames = video_1.shape[0]
        frames_per_cut = max(1, round(switch_every_seconds * fps))

        # Validate all videos have compatible dimensions
        _, h, w, c = video_1.shape
        for i, v in enumerate(videos):
            if v.shape[1] != h or v.shape[2] != w:
                raise ValueError(
                    f"video_{i+1} has resolution {v.shape[2]}x{v.shape[1]}, "
                    f"but video_1 is {w}x{h}. All videos must match."
                )

        rng = random.Random(seed)
        output_frames = []
        cut_log_lines = []
        frame_idx = 0
        last_camera = -1

        while frame_idx < total_frames:
            seg_end = min(frame_idx + frames_per_cut, total_frames)

            # Pick camera
            if num_cameras == 1:
                cam = 0
            elif avoid_repeat and num_cameras > 1:
                choices = [c for c in range(num_cameras) if c != last_camera]
                cam = rng.choice(choices)
            else:
                cam = rng.randint(0, num_cameras - 1)

            last_camera = cam
            src_video = videos[cam]

            # If source video is shorter than needed, wrap around
            seg_len = seg_end - frame_idx
            src_total = src_video.shape[0]

            indices = [(frame_idx + j) % src_total for j in range(seg_len)]
            segment = src_video[indices]
            output_frames.append(segment)

            time_start = frame_idx / fps
            time_end = seg_end / fps
            cut_log_lines.append(f"[{time_start:6.2f}s - {time_end:6.2f}s] camera {cam+1} (frames {frame_idx}-{seg_end-1})")

            frame_idx = seg_end

        output = torch.cat(output_frames, dim=0)
        cut_log = f"Camera Switcher | {num_cameras} cameras | {frames_per_cut} frames/cut ({switch_every_seconds}s @ {fps}fps)\n"
        cut_log += f"Total: {total_frames} frames ({total_frames/fps:.1f}s)\n"
        cut_log += "-" * 60 + "\n"
        cut_log += "\n".join(cut_log_lines)

        return (output, cut_log,)
