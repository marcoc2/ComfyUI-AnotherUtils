import os
from .trello_utils import TrelloParser

class TrelloPromptListLoader:
    """
    Trello node that returns a list of all prompts from a specific column/list.
    Filters out prompts with less than 50 characters.
    Output compatible with list-processing nodes.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        # Default path relative to this script
        base_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        default_path = os.path.join(base_path, "reference", "jggNQEHi - prompts.json")
        
        # Try to pre-load lists for the dropdown
        list_options = ["All"]
        data = TrelloParser.get_data(default_path)
        if data:
            list_options.extend(data["list_names"])
        else:
            list_options.append("No lists found (check path)")

        return {
            "required": {
                "json_path": ("STRING", {"default": default_path}),
                "list_filter": (list_options, {"default": "All"}),
                "selected_id": ("STRING", {"default": ""}), # Populated by JS UI
                "min_char_count": ("INT", {"default": 50, "min": 0, "max": 1000, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompts",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "load_all"
    CATEGORY = "text/prompt"
    TITLE = "Trello List Loader (Batch)"

    def load_all(self, json_path, list_filter, selected_id, min_char_count):
        data = TrelloParser.get_data(json_path)
        
        if not data:
            print(f"[TrelloListLoader] Error: JSON file not found at {json_path}")
            return ([],)
            
        cards = data["cards"]
        
        # Apply Trello list filter
        if list_filter != "All":
            filtered_cards = [c for c in cards if c["list"] == list_filter]
        else:
            filtered_cards = cards
            
        # Extract prompts and apply character length filter
        prompts = []
        for card in filtered_cards:
            prompt = card["name"]
            if len(prompt) >= min_char_count:
                prompts.append(prompt)
        
        print(f"[TrelloListLoader] Loaded {len(prompts)} prompts from list '{list_filter}' (Filtered by min {min_char_count} chars)")
            
        return (prompts,)

    @classmethod
    def IS_CHANGED(cls, json_path, list_filter, selected_id, min_char_count):
        # Update if file changes or filter/selection changes
        if os.path.exists(json_path):
            mtime = os.path.getmtime(json_path)
        else:
            mtime = 0
        return f"{json_path}_{list_filter}_{selected_id}_{min_char_count}_{mtime}"
