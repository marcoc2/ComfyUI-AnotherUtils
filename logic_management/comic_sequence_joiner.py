"""
Comic Sequence Joiner Node for ComfyUI

Receives a LIST of strings (one per comic panel, from QwenVL responses)
and assembles them into a single structured string using <sequenceN> tags,
ready to be fed as the user_prompt of a second QwenVL call.
"""


class ComicSequenceJoiner:
    """
    Joins N panel description strings into a single structured text block
    using <sequenceN> tags, ready to feed into a video prompt generator.

    Input:
      - panel_texts : LIST of STRING (one per panel, in reading order)
      - preamble    : optional header before the sequences
      - postamble   : optional instruction after the sequences

    Output:
      - structured_text : STRING with all sequences tagged
      - panel_count     : INT
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "panel_texts": ("STRING", {"forceInput": True}),
                "preamble": (
                    "STRING",
                    {
                        "default": (
                            "The following is a structured description of each panel "
                            "in a comic page, in reading order:"
                        ),
                        "multiline": True,
                    },
                ),
                "postamble": (
                    "STRING",
                    {
                        "default": (
                            "Using ALL the sequence descriptions above and the full comic page image provided, "
                            "write a single, cohesive video generation prompt. "
                            "Capture the mood, timing, humor and narrative arc of the whole page. "
                            "The prompt should describe motion, transitions and atmosphere "
                            "as if directing a short animated clip."
                        ),
                        "multiline": True,
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("structured_text", "panel_count")
    OUTPUT_IS_LIST = (False, False)
    INPUT_IS_LIST = True          # receive the full list at once
    FUNCTION = "join_sequences"
    CATEGORY = "AnotherUtils/text"

    def join_sequences(self, panel_texts, preamble, postamble):
        # When INPUT_IS_LIST=True, all inputs arrive as lists
        # preamble and postamble will be single-element lists
        pre = preamble[0].strip() if preamble else ""
        post = postamble[0].strip() if postamble else ""

        parts = []

        if pre:
            parts.append(pre)
            parts.append("")

        for i, text in enumerate(panel_texts, start=1):
            t = text.strip() if isinstance(text, str) else str(text).strip()
            parts.append(f"<sequence{i}>")
            parts.append(t)
            parts.append("")   # blank line between sequences

        if post:
            parts.append(post)

        structured = "\n".join(parts).strip()
        count = len(panel_texts)

        return (structured, count)
