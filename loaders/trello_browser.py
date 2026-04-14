import os
from .trello_utils import TrelloParser

class TrelloBrowser:
    """
    Advanced Trello node with a Javascript-based visual browser.
    Allows point-and-click selection of prompts with image previews.
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
                "selected_id": ("STRING", {"default": ""}), # This is populated by the JS UI
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("prompt", "category", "image_url")
    FUNCTION = "load_selected"
    CATEGORY = "text/prompt"
    TITLE = "Trello Browser (Advanced)"

    def load_selected(self, json_path, list_filter, selected_id):
        data = TrelloParser.get_data(json_path)
        
        if not data:
            return ("Error: JSON file not found", "None", "")
            
        if not selected_id:
            return ("No prompt selected in browser", "None", "")
            
        # Find the card by ID
        card = next((c for c in data["cards"] if c["id"] == selected_id), None)
        
        if not card:
            return (f"Error: Card {selected_id} not found", "None", "")
            
        return (card["name"], card["list"], card["image_url"])

    @classmethod
    def IS_CHANGED(cls, json_path, list_filter, selected_id):
        return f"{json_path}_{list_filter}_{selected_id}"
