class AudioSliceSelector:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_list": ("AUDIO",),
                "index": ("INT", {"default": 0, "min": 0, "max": 999}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    INPUT_IS_LIST = True
    FUNCTION = "select"
    CATEGORY = "audio"

    def select(self, audio_list, index):
        # When INPUT_IS_LIST is True, all inputs arrive as lists
        idx = index[0] if isinstance(index, list) else index

        if not audio_list:
            raise ValueError("No audio slices received.")

        # Clamp index
        idx = max(0, min(idx, len(audio_list) - 1))
        return (audio_list[idx],)
