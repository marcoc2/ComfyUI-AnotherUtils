from typing import Dict, List

class CharacterAttributes:
    """Class to store common character attributes used by both nodes"""
    genders = ["Male", "Female"]
    fighting_styles = [
        "Judo", "Karate", "Boxing", "Kung Fu", "Muay Thai", 
        "Capoeira", "Wrestling", "Aikido", "Taekwondo", "Krav Maga"
    ]
    nationalities = [
        "Japanese", "Brazilian", "American", "Chinese", 
        "Thai", "Russian", "Mexican", "Korean", 
        "Indian", "Italian"
    ]
    personalities = [
        "Disciplined and Serious",
        "Arrogant and Competitive",
        "Playful and Easygoing",
        "Mysterious and Reserved",
        "Honorable and Traditional",
        "Rebellious and Impulsive",
        "Calm and Strategic",
        "Vengeful and Determined"
    ]
    occupations = [
        "Martial Arts Student",
        "Traditional Master",
        "Street Fighter",
        "Special Agent",
        "Performance Artist",
        "Mercenary",
        "Martial Arts Teacher",
        "Professional Athlete"
    ]
    special_powers = [
        "Ki Energy",
        "Elemental Manipulation",
        "Superhuman Strength",
        "Extraordinary Speed",
        "Secret Ancestral Techniques",
        "Beast Transformation",
        "Mystical Weapons",
        "Psychic Powers"
    ]
    weapons = [
        "None",
        "Traditional Sword",
        "Fighting Gauntlets",
        "Energy-infused Staff",
        "Chain Weapons",
        "Combat Fans",
        "Ceremonial Daggers",
        "Ancient Relic Weapon"
    ]
    clothing_styles = [
        "Traditional Martial Arts Gi",
        "Modern Street Fashion",
        "Cultural Traditional Attire",
        "Military-inspired Outfit",
        "Ceremonial Combat Robes",
        "Urban Fighting Gear",
        "Ancient Warrior Armor",
        "Customized Fight Uniform"
    ]
    makeup = [
        "None",
        "Traditional Face Paint",
        "Ritual Markings",
        "Modern Stylish",
        "War Paint",
        "Stage Performance",
        "Cultural Symbols",
        "Battle Scars"
    ]
    accessories = [
        "Sacred Beads",
        "Power-limiting Bracers",
        "Family Heirloom",
        "Tech Gadgets",
        "Mystical Ornaments",
        "Traditional Jewelry",
        "Combat Belt",
        "Energy-channeling Crystals"
    ]


class CharacterRandomizer:
    """Node that generates random character attributes"""
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        }
    
    RETURN_TYPES = tuple(["STRING"] * 10)  # One for each attribute
    RETURN_NAMES = (
        "gender", "fighting_style", "nationality", "personality",
        "occupation", "special_power", "weapon", "clothing",
        "makeup", "accessories"
    )
    FUNCTION = "randomize"
    OUTPUT_NODE = True
    CATEGORY = "prompt/generators"

    def randomize(self, seed):
        import random
        random.seed(seed)
        
        return (
            random.choice(CharacterAttributes.genders),
            random.choice(CharacterAttributes.fighting_styles),
            random.choice(CharacterAttributes.nationalities),
            random.choice(CharacterAttributes.personalities),
            random.choice(CharacterAttributes.occupations),
            random.choice(CharacterAttributes.special_powers),
            random.choice(CharacterAttributes.weapons),
            random.choice(CharacterAttributes.clothing_styles),
            random.choice(CharacterAttributes.makeup),
            random.choice(CharacterAttributes.accessories)
        )