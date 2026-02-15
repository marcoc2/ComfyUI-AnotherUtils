import torch
import av
import os
import json
import hashlib
import folder_paths


def f32_pcm(wav: torch.Tensor) -> torch.Tensor:
    if wav.dtype.is_floating_point:
        return wav
    elif wav.dtype == torch.int16:
        return wav.float() / (2 ** 15)
    elif wav.dtype == torch.int32:
        return wav.float() / (2 ** 31)
    raise ValueError(f"Unsupported wav dtype: {wav.dtype}")


def load_audio(filepath: str) -> tuple:
    with av.open(filepath) as af:
        if not af.streams.audio:
            raise ValueError("No audio stream found in the file.")

        stream = af.streams.audio[0]
        sr = stream.codec_context.sample_rate
        n_channels = stream.channels

        frames = []
        for frame in af.decode(streams=stream.index):
            buf = torch.from_numpy(frame.to_ndarray())
            if buf.shape[0] != n_channels:
                buf = buf.view(-1, n_channels).t()
            frames.append(buf)

        if not frames:
            raise ValueError("No audio frames decoded.")

        wav = torch.cat(frames, dim=1)
        wav = f32_pcm(wav)
        return wav, sr


class AudioWaveformSlicer:
    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = folder_paths.filter_files_content_types(
            os.listdir(input_dir), ["audio", "video"]
        )
        return {
            "required": {
                "audio": (sorted(files), {"audio_upload": True}),
                "cut_positions": ("STRING", {"default": "[]", "multiline": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio_slices",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "slice_audio"
    CATEGORY = "audio"

    def slice_audio(self, audio, cut_positions):
        audio_path = folder_paths.get_annotated_filepath(audio)
        waveform, sample_rate = load_audio(audio_path)
        # waveform shape: [C, T]

        total_samples = waveform.shape[1]

        # Parse cut positions (in seconds), sort them
        try:
            cuts = json.loads(cut_positions)
        except (json.JSONDecodeError, TypeError):
            cuts = []

        if not isinstance(cuts, list):
            cuts = []

        # Convert seconds to sample indices, filter valid ones
        cut_samples = []
        for c in cuts:
            try:
                s = int(float(c) * sample_rate)
                if 0 < s < total_samples:
                    cut_samples.append(s)
            except (ValueError, TypeError):
                continue

        cut_samples = sorted(set(cut_samples))

        # Build slice boundaries
        boundaries = [0] + cut_samples + [total_samples]

        slices = []
        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            if end > start:
                slice_waveform = waveform[:, start:end]
                slices.append({
                    "waveform": slice_waveform.unsqueeze(0),  # [1, C, T]
                    "sample_rate": sample_rate,
                })

        if not slices:
            slices.append({
                "waveform": waveform.unsqueeze(0),
                "sample_rate": sample_rate,
            })

        return (slices,)

    @classmethod
    def IS_CHANGED(cls, audio, cut_positions):
        audio_path = folder_paths.get_annotated_filepath(audio)
        m = hashlib.sha256()
        with open(audio_path, "rb") as f:
            m.update(f.read())
        m.update(cut_positions.encode())
        return m.digest().hex()

    @classmethod
    def VALIDATE_INPUTS(cls, audio, **kwargs):
        if not folder_paths.exists_annotated_filepath(audio):
            return "Invalid audio file: {}".format(audio)
        return True
