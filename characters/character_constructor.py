from typing import Dict, List
from .character_generator import CharacterAttributes

class CharacterConstructor:
    """Node that constructs the character prompt from given attributes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "gender": (CharacterAttributes.genders,),
                "fighting_style": (CharacterAttributes.fighting_styles,),
                "nationality": (CharacterAttributes.nationalities,),
                "personality": (CharacterAttributes.personalities,),
                "occupation": (CharacterAttributes.occupations,),
                "special_power": (CharacterAttributes.special_powers,),
                "weapon": (CharacterAttributes.weapons,),
                "clothing": (CharacterAttributes.clothing_styles,),
                "makeup": (CharacterAttributes.makeup,),
                "accessories": (CharacterAttributes.accessories,)
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")  # Prompt and Selected Values summary
    RETURN_NAMES = ("prompt", "selected_values")
    FUNCTION = "construct"
    OUTPUT_NODE = True
    CATEGORY = "prompt/generators"

    def construct(self, gender, fighting_style, nationality, personality,
                 occupation, special_power, weapon, clothing, makeup, accessories):
        # Create character dictionary
        character = {
            "Gender": gender,
            "Fighting Style": fighting_style,
            "Nationality": nationality,
            "Personality": personality,
            "Occupation": occupation,
            "Special Power": special_power,
            "Weapon": weapon,
            "Clothing": clothing,
            "Makeup": makeup,
            "Accessories": accessories
        }
        
        # Create prompt
        prompt = f"""Create a fighting game character: 
A {character['Nationality']} {character['Gender'].lower()} who is a {character['Occupation'].lower()}.
Practitioner of {character['Fighting Style']}, with a {character['Personality'].lower()} personality.
Possesses {character['Special Power'].lower()} as a special power.
Wields {character['Weapon'].lower()} in combat.
Wears {character['Clothing'].lower()}.
Features {character['Makeup'].lower()} on their face.
Adorned with {character['Accessories'].lower()}."""

        # Create a formatted string of selected values
        selected_values = "\n".join([f"{k}: {v}" for k, v in character.items()])
        
        return (prompt, selected_values)