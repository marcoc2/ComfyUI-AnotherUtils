"""
Video Audio Combiner Node
Combines a video file with an audio file using ffmpeg.
Supports both the new IO.Video type and legacy file paths.
"""

import os
import subprocess
import tempfile
import io
import folder_paths

# Try to get ffmpeg path from VHS if available
try:
    from custom_nodes.ComfyUI_VideoHelperSuite.videohelpersuite.utils import ffmpeg_path
except ImportError:
    # Fallback to system ffmpeg
    ffmpeg_path = "ffmpeg"

# Import for new IO.Video type support
try:
    from comfy_api.latest import IO, Input
    from comfy_api.latest._input_impl.video_types import VideoFromFile
    HAS_NEW_VIDEO_API = True
except ImportError:
    HAS_NEW_VIDEO_API = False


class VideoAudioCombiner:
    """
    Combines a video file with an audio track.
    The audio is trimmed/looped to match the video duration.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "placeholder": "path/to/video.mp4",
                    "tooltip": "Path to the input video file (mp4, mkv, avi, etc.)"
                }),
                "audio_start_seconds": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 36000.0,  # 10 hours max
                    "step": 0.1,
                    "tooltip": "Start position in the audio file (seconds)"
                }),
                "loop_audio": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "If audio is shorter than video, loop it to fill the duration"
                }),
                "output_filename": ("STRING", {
                    "default": "output_with_audio",
                    "tooltip": "Output filename (without extension)"
                }),
            },
            "optional": {
                "audio": ("AUDIO", {
                    "tooltip": "Audio input from VHS or other audio nodes"
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "placeholder": "path/to/audio.mp3",
                    "tooltip": "Alternative: direct path to audio file"
                }),
                "vhs_filenames": ("VHS_FILENAMES", {
                    "tooltip": "Alternative: video from VHS Video Combine node"
                }),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "combine"
    CATEGORY = "AnotherUtils/video"

    def combine(self, video_path, audio_start_seconds, loop_audio, output_filename,
                audio=None, audio_path="", vhs_filenames=None):

        # Determine video source
        if vhs_filenames is not None:
            # VHS_FILENAMES is a tuple: (save_output, [list_of_files])
            if isinstance(vhs_filenames, tuple) and len(vhs_filenames) >= 2:
                file_list = vhs_filenames[1]
                if file_list and len(file_list) > 0:
                    # Get the first video file
                    video_path = file_list[0]
                    if not os.path.isabs(video_path):
                        video_path = os.path.join(folder_paths.get_output_directory(), video_path)

        if not video_path or not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        # Determine audio source
        audio_file = None
        temp_audio_file = None

        if audio is not None:
            # AUDIO type is a dict with 'waveform' and 'sample_rate'
            # We need to save it to a temporary file
            import torch
            import numpy as np

            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            # Create temp file for audio
            temp_audio_file = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio_file.name
            temp_audio_file.close()

            # Convert tensor to wav using ffmpeg
            # waveform shape: [batch, channels, samples]
            audio_data = waveform.squeeze(0).numpy()  # [channels, samples]

            # Interleave channels for raw audio
            if audio_data.shape[0] == 2:
                # Stereo: interleave left and right
                interleaved = np.empty((audio_data.shape[1] * 2,), dtype=np.float32)
                interleaved[0::2] = audio_data[0]
                interleaved[1::2] = audio_data[1]
                channels = 2
            else:
                interleaved = audio_data.flatten().astype(np.float32)
                channels = 1

            # Write raw audio and convert with ffmpeg
            raw_temp = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
            raw_temp_path = raw_temp.name
            interleaved.tofile(raw_temp)
            raw_temp.close()

            # Convert raw to wav
            ffmpeg_cmd = [
                ffmpeg_path, '-y',
                '-f', 'f32le',
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-i', raw_temp_path,
                temp_audio_path
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            os.unlink(raw_temp_path)

            audio_file = temp_audio_path

        elif audio_path and os.path.exists(audio_path):
            audio_file = audio_path
        else:
            raise ValueError("No audio source provided. Use either 'audio' input or 'audio_path'.")

        # Get video duration
        probe_cmd = [
            ffmpeg_path, '-i', video_path,
            '-hide_banner'
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True)

        # Parse duration from stderr
        import re
        duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.?\d*)', result.stderr)
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            video_duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            raise ValueError("Could not determine video duration")

        # Prepare output path
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, f"{output_filename}.mp4")

        # Build ffmpeg command
        ffmpeg_cmd = [ffmpeg_path, '-y']

        # Add video input
        ffmpeg_cmd.extend(['-i', video_path])

        # Add audio input with start offset
        if audio_start_seconds > 0:
            ffmpeg_cmd.extend(['-ss', str(audio_start_seconds)])
        ffmpeg_cmd.extend(['-i', audio_file])

        # Build filter for audio
        audio_filter = []

        if loop_audio:
            # Loop audio if needed
            audio_filter.append(f'aloop=loop=-1:size=2e+09')

        # Trim audio to video duration
        audio_filter.append(f'atrim=0:{video_duration}')
        audio_filter.append('asetpts=PTS-STARTPTS')

        if audio_filter:
            ffmpeg_cmd.extend(['-af', ','.join(audio_filter)])

        # Output settings
        ffmpeg_cmd.extend([
            '-map', '0:v:0',      # Take video from first input
            '-map', '1:a:0',      # Take audio from second input
            '-c:v', 'copy',       # Copy video stream (no re-encoding)
            '-c:a', 'aac',        # Encode audio as AAC
            '-b:a', '192k',       # Audio bitrate
            '-shortest',          # End when shortest stream ends
            output_path
        ])

        print(f"[VideoAudioCombiner] Running: {' '.join(ffmpeg_cmd)}")

        try:
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[VideoAudioCombiner] FFmpeg error: {e.stderr}")
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")
        finally:
            # Cleanup temp files
            if temp_audio_file and os.path.exists(temp_audio_path):
                os.unlink(temp_audio_path)

        print(f"[VideoAudioCombiner] Output saved to: {output_path}")

        return (output_path,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")  # Always re-execute


class VideoAudioCombinerSimple:
    """
    Simple version that takes file paths directly.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_path": ("STRING", {
                    "default": "",
                    "placeholder": "C:/path/to/video.mp4",
                }),
                "audio_path": ("STRING", {
                    "default": "",
                    "placeholder": "C:/path/to/audio.mp3",
                }),
                "audio_start_seconds": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 36000.0,
                    "step": 0.1,
                }),
                "loop_audio": ("BOOLEAN", {
                    "default": False,
                }),
                "output_filename": ("STRING", {
                    "default": "output_with_audio",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    OUTPUT_NODE = True
    FUNCTION = "combine"
    CATEGORY = "AnotherUtils/video"

    def combine(self, video_path, audio_path, audio_start_seconds, loop_audio, output_filename):

        if not os.path.exists(video_path):
            raise ValueError(f"Video file not found: {video_path}")

        if not os.path.exists(audio_path):
            raise ValueError(f"Audio file not found: {audio_path}")

        # Get video duration
        probe_cmd = [ffmpeg_path, '-i', video_path, '-hide_banner']
        result = subprocess.run(probe_cmd, capture_output=True, text=True)

        import re
        duration_match = re.search(r'Duration: (\d+):(\d+):(\d+\.?\d*)', result.stderr)
        if duration_match:
            hours, minutes, seconds = duration_match.groups()
            video_duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            raise ValueError("Could not determine video duration")

        # Prepare output path
        output_dir = folder_paths.get_output_directory()
        output_path = os.path.join(output_dir, f"{output_filename}.mp4")

        # Build ffmpeg command
        ffmpeg_cmd = [ffmpeg_path, '-y', '-i', video_path]

        if audio_start_seconds > 0:
            ffmpeg_cmd.extend(['-ss', str(audio_start_seconds)])
        ffmpeg_cmd.extend(['-i', audio_path])

        # Audio filter
        audio_filter = []
        if loop_audio:
            audio_filter.append('aloop=loop=-1:size=2e+09')
        audio_filter.append(f'atrim=0:{video_duration}')
        audio_filter.append('asetpts=PTS-STARTPTS')

        ffmpeg_cmd.extend(['-af', ','.join(audio_filter)])
        ffmpeg_cmd.extend([
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            output_path
        ])

        print(f"[VideoAudioCombinerSimple] Running ffmpeg...")

        try:
            subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"FFmpeg failed: {e.stderr}")

        print(f"[VideoAudioCombinerSimple] Output: {output_path}")

        return (output_path,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")


# New IO.Video-based node (only available if the new API is present)
if HAS_NEW_VIDEO_API:
    class VideoAudioCombinerV3(IO.ComfyNode):
        """
        Mixes video's original audio with a new audio track using the new IO.Video type.
        The new audio is trimmed/looped to match the video duration.
        Both audio tracks can have their volume adjusted independently.
        """

        @classmethod
        def define_schema(cls) -> IO.Schema:
            return IO.Schema(
                node_id="VideoAudioCombinerV3",
                display_name="Video + Audio Combiner (V3)",
                category="AnotherUtils/video",
                description="Mixes video's original audio with a new audio track. Both volumes are adjustable (0=mute, 1=normal, 2=double).",
                inputs=[
                    IO.Video.Input(
                        "video",
                        tooltip="Input video (original audio will be preserved and mixed)"
                    ),
                    IO.Audio.Input(
                        "audio",
                        tooltip="Audio to mix with the video's original audio"
                    ),
                    IO.Float.Input(
                        "original_volume",
                        default=1.0,
                        min=0.0,
                        max=2.0,
                        step=0.1,
                        tooltip="Volume of the video's original audio (0=mute, 1=normal, 2=double)"
                    ),
                    IO.Float.Input(
                        "mix_volume",
                        default=1.0,
                        min=0.0,
                        max=2.0,
                        step=0.1,
                        tooltip="Volume of the mixed audio (0=mute, 1=normal, 2=double)"
                    ),
                    IO.Float.Input(
                        "audio_start_seconds",
                        default=0.0,
                        min=0.0,
                        max=36000.0,
                        step=0.1,
                        tooltip="Start position in the new audio file (seconds)"
                    ),
                    IO.Boolean.Input(
                        "loop_audio",
                        default=False,
                        tooltip="If new audio is shorter than video, loop it to fill the duration"
                    ),
                ],
                outputs=[IO.Video.Output()],
            )

        @classmethod
        def execute(
            cls,
            video: Input.Video,
            audio: dict,
            original_volume: float,
            mix_volume: float,
            audio_start_seconds: float,
            loop_audio: bool,
        ) -> IO.NodeOutput:
            import numpy as np

            # Get video duration using the new API
            video_duration = video.get_duration()

            # Save input video to temp file for ffmpeg processing
            temp_video = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_video_path = temp_video.name
            temp_video.close()
            video.save_to(temp_video_path)

            # Convert AUDIO tensor to temp wav file
            waveform = audio['waveform']
            sample_rate = audio['sample_rate']

            temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
            temp_audio_path = temp_audio.name
            temp_audio.close()

            # Convert tensor to wav
            audio_data = waveform.squeeze(0).numpy()

            if audio_data.shape[0] == 2:
                interleaved = np.empty((audio_data.shape[1] * 2,), dtype=np.float32)
                interleaved[0::2] = audio_data[0]
                interleaved[1::2] = audio_data[1]
                channels = 2
            else:
                interleaved = audio_data.flatten().astype(np.float32)
                channels = 1

            raw_temp = tempfile.NamedTemporaryFile(suffix='.raw', delete=False)
            raw_temp_path = raw_temp.name
            interleaved.tofile(raw_temp)
            raw_temp.close()

            ffmpeg_cmd = [
                ffmpeg_path, '-y',
                '-f', 'f32le',
                '-ar', str(sample_rate),
                '-ac', str(channels),
                '-i', raw_temp_path,
                temp_audio_path
            ]
            subprocess.run(ffmpeg_cmd, capture_output=True, check=True)
            os.unlink(raw_temp_path)

            # Create output temp file
            temp_output = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            temp_output_path = temp_output.name
            temp_output.close()

            # Build ffmpeg command for mixing audio
            ffmpeg_cmd = [ffmpeg_path, '-y', '-i', temp_video_path]

            # Add new audio input with start offset
            if audio_start_seconds > 0:
                ffmpeg_cmd.extend(['-ss', str(audio_start_seconds)])
            ffmpeg_cmd.extend(['-i', temp_audio_path])

            # Build complex filter for audio mixing
            # [0:a] = original video audio, [1:a] = new audio
            filter_parts = []

            # Process new audio: loop if needed, then trim to video duration
            new_audio_filter = "[1:a]"
            if loop_audio:
                new_audio_filter += "aloop=loop=-1:size=2e+09,"
            new_audio_filter += f"atrim=0:{video_duration},asetpts=PTS-STARTPTS"
            if mix_volume != 1.0:
                new_audio_filter += f",volume={mix_volume}"
            new_audio_filter += "[new_audio]"
            filter_parts.append(new_audio_filter)

            # Process original audio with volume
            orig_audio_filter = "[0:a]"
            if original_volume != 1.0:
                orig_audio_filter += f"volume={original_volume}"
            else:
                orig_audio_filter += "acopy"
            orig_audio_filter += "[orig_audio]"
            filter_parts.append(orig_audio_filter)

            # Mix both audio streams
            filter_parts.append("[orig_audio][new_audio]amix=inputs=2:duration=first:dropout_transition=0[mixed_audio]")

            filter_complex = ";".join(filter_parts)

            ffmpeg_cmd.extend([
                '-filter_complex', filter_complex,
                '-map', '0:v:0',           # Take video from first input
                '-map', '[mixed_audio]',   # Take mixed audio
                '-c:v', 'copy',            # Copy video stream (no re-encoding)
                '-c:a', 'aac',             # Encode audio as AAC
                '-b:a', '192k',            # Audio bitrate
                temp_output_path
            ])

            print(f"[VideoAudioCombinerV3] Running ffmpeg to mix audio...")

            try:
                result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, check=True)
            except subprocess.CalledProcessError as e:
                print(f"[VideoAudioCombinerV3] FFmpeg error: {e.stderr}")
                raise RuntimeError(f"FFmpeg failed: {e.stderr}")
            finally:
                # Cleanup input temp files
                if os.path.exists(temp_video_path):
                    os.unlink(temp_video_path)
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)

            print(f"[VideoAudioCombinerV3] Audio mixed successfully")

            # Return as IO.Video using VideoFromFile
            # Note: temp_output_path will be managed by VideoFromFile
            output_video = VideoFromFile(temp_output_path)
            return IO.NodeOutput(output_video)
