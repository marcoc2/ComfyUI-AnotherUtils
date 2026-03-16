import torch


class AudioConcatenate:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "inputcount": ("INT", {"default": 2, "min": 2, "max": 64, "step": 1}),
                "audio_1": ("AUDIO",),
            },
            "optional": {
                "audio_2": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "concatenate"
    CATEGORY = "audio"
    DESCRIPTION = "Concatenates multiple audio inputs in order. Use the inputcount widget and click Update Inputs to add/remove slots."

    def concatenate(self, inputcount, **kwargs):
        audios = []
        for i in range(1, inputcount + 1):
            key = f"audio_{i}"
            if key in kwargs and kwargs[key] is not None:
                audios.append(kwargs[key])

        if not audios:
            raise ValueError("No audio inputs connected.")

        sample_rate = audios[0]["sample_rate"]

        waveforms = []
        for audio in audios:
            wf = audio["waveform"]
            # Resample if needed (simple case: just use first sample rate)
            if audio["sample_rate"] != sample_rate:
                # Basic resample by ratio
                ratio = sample_rate / audio["sample_rate"]
                length = int(wf.shape[-1] * ratio)
                wf = torch.nn.functional.interpolate(
                    wf, size=length, mode="linear", align_corners=False
                )
            waveforms.append(wf)

        # Match channel count (pad mono to stereo if needed)
        max_channels = max(wf.shape[1] for wf in waveforms)
        matched = []
        for wf in waveforms:
            if wf.shape[1] < max_channels:
                wf = wf.repeat(1, max_channels, 1)[:, :max_channels, :]
            matched.append(wf)

        concatenated = torch.cat(matched, dim=2)

        return ({"waveform": concatenated, "sample_rate": sample_rate},)
