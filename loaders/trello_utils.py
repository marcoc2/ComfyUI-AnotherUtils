import json
import os

class TrelloParser:
    """Helper class to parse and cache Trello JSON data."""
    _cache = {}

    @classmethod
    def get_data(cls, json_path):
        if json_path in cls._cache:
            return cls._cache[json_path]
            
        if not os.path.exists(json_path):
            return None
            
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lists = {l['id']: l['name'] for l in data.get('lists', [])}
            
            cards = []
            for card in data.get('cards', []):
                if card.get('closed'): # Skip archived
                    continue
                    
                list_name = lists.get(card['idList'], "Unknown")
                
                # Get image attachments
                image_url = ""
                attachments = card.get('attachments', [])
                for att in attachments:
                    url = att.get('url', '')
                    if any(url.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.webp']):
                        image_url = url
                        break
                
                cards.append({
                    "id": card['id'],
                    "name": card['name'],
                    "desc": card['desc'],
                    "list": list_name,
                    "image_url": image_url
                })
            
            result = {
                "cards": cards,
                "list_names": sorted(list(set(lists.values())))
            }
            cls._cache[json_path] = result
            return result
        except Exception as e:
            print(f"Error parsing Trello JSON: {e}")
            return None
