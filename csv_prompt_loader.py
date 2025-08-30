import csv
import os
import torch

class CSVPromptLoader:
    """
    A custom node for ComfyUI that loads prompts from a CSV file with dropdown selection.
    The CSV file should have columns: name, prompt, negative_prompt
    """
    
    csv_cache = {}  # Class-level cache to share data between instances
    
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "csv_path": ("STRING", {"default": ""}),
                "selected_name": (cls._get_names_from_cache, {"default": ""}),
            },
        }

    @classmethod
    def _get_names_from_cache(cls):
        """Get available names from the cached CSV data"""
        names = []
        for path_data in cls.csv_cache.values():
            names.extend(path_data.keys())
        if not names:
            names = ["No CSV loaded"]
        return names

    @classmethod
    def _load_csv_data(cls, csv_path):
        """Load CSV data and cache it"""
        if not os.path.exists(csv_path):
            return {}
            
        data = {}
        try:
            with open(csv_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if 'name' in row and 'prompt' in row and 'negative_prompt' in row:
                        data[row['name']] = {
                            'prompt': row['prompt'],
                            'negative_prompt': row['negative_prompt']
                        }
        except Exception as e:
            print(f"Error loading CSV: {e}")
            return {}
            
        # Cache the data
        cls.csv_cache[csv_path] = data
        return data

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("prompt", "negative_prompt")
    FUNCTION = "load_prompt"
    CATEGORY = "text/prompt"
    TITLE = "CSV Prompt Loader"

    def load_prompt(self, csv_path, selected_name):
        # Load CSV data if not cached
        if csv_path not in self.csv_cache:
            self._load_csv_data(csv_path)
            
        # Get cached data
        csv_data = self.csv_cache.get(csv_path, {})
        
        # If no data loaded, return empty strings
        if not csv_data:
            return ("", "")
            
        # If selected name exists, return its data
        if selected_name in csv_data:
            data = csv_data[selected_name]
            return (data['prompt'], data['negative_prompt'])
            
        # Return the first entry as fallback
        if csv_data:
            first_name = list(csv_data.keys())[0]
            first_data = csv_data[first_name]
            return (first_data['prompt'], first_data['negative_prompt'])
            
        return ("", "")

    @classmethod
    def IS_CHANGED(cls, csv_path, selected_name):
        return f"{csv_path}_{selected_name}"

    @classmethod
    def VALIDATE_INPUTS(cls, csv_path, selected_name):
        if not csv_path:
            return "CSV path cannot be empty"
        if not os.path.exists(csv_path):
            return "CSV file does not exist"
        return True