import os
import random
from .trello_utils import TrelloParser

class TrelloPromptLoader:
    """
    A custom node for ComfyUI that loads prompts from a Trello JSON export.
    Simple version with index-based iteration.
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
                "index": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
                "randomize": ("BOOLEAN", {"default": False}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("prompt", "category", "total")
    FUNCTION = "load_prompt"
    CATEGORY = "text/prompt"
    TITLE = "Trello Prompt Loader"

    def load_prompt(self, json_path, list_filter, index, randomize, seed):
        data = TrelloParser.get_data(json_path)
        
        if not data:
            return ("Error: JSON file not found or invalid", "None", 0)
            
        cards = data["cards"]
        
        # Apply filter
        if list_filter != "All":
            filtered_cards = [c for c in cards if c["list"] == list_filter]
        else:
            filtered_cards = cards
            
        total = len(filtered_cards)
        if total == 0:
            return ("No prompts found in this category", list_filter, 0)
            
        # Select prompt
        if randomize:
            # Use seed for deterministic randomness if provided
            random.seed(seed)
            selected_card = random.choice(filtered_cards)
        else:
            # Robustly handle index (ensure it's an int and handle list mapping artifacts)
            true_index = index[0] if isinstance(index, list) else index
            try:
                idx = int(true_index) % total if total > 0 else 0
            except (ValueError, TypeError):
                idx = 0
            selected_card = filtered_cards[idx]
            
        prompt = selected_card["name"]
        category = selected_card["list"]
        
        return (prompt, category, total)

    @classmethod
    def IS_CHANGED(cls, json_path, list_filter, index, randomize, seed):
        return f"{json_path}_{list_filter}_{index}_{randomize}_{seed}"
