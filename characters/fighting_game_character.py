class FightingGameCharacter:
    def __init__(self):
        # Initialize attribute lists as class attributes for easy access
        self.genders = ["Male", "Female"]
        self.fighting_styles = [
            "Judo", "Karate", "Boxing", "Kung Fu", "Muay Thai", 
            "Capoeira", "Wrestling", "Aikido", "Taekwondo", "Krav Maga"
        ]
        self.nationalities = [
            "Japanese", "Brazilian", "American", "Chinese", 
            "Thai", "Russian", "Mexican", "Korean", 
            "Indian", "Italian"
        ]
        self.personalities = [
            "Disciplined and Serious",
            "Arrogant and Competitive",
            "Playful and Easygoing",
            "Mysterious and Reserved",
            "Honorable and Traditional",
            "Rebellious and Impulsive",
            "Calm and Strategic",
            "Vengeful and Determined"
        ]
        self.occupations = [
            "Martial Arts Student",
            "Traditional Master",
            "Street Fighter",
            "Special Agent",
            "Performance Artist",
            "Mercenary",
            "Martial Arts Teacher",
            "Professional Athlete"
        ]
        self.special_powers = [
            "Ki Energy",
            "Elemental Manipulation",
            "Superhuman Strength",
            "Extraordinary Speed",
            "Secret Ancestral Techniques",
            "Beast Transformation",
            "Mystical Weapons",
            "Psychic Powers"
        ]
        self.weapons = [
            "None",
            "Traditional Sword",
            "Fighting Gauntlets",
            "Energy-infused Staff",
            "Chain Weapons",
            "Combat Fans",
            "Ceremonial Daggers",
            "Ancient Relic Weapon"
        ]
        self.clothing_styles = [
            "Traditional Martial Arts Gi",
            "Modern Street Fashion",
            "Cultural Traditional Attire",
            "Military-inspired Outfit",
            "Ceremonial Combat Robes",
            "Urban Fighting Gear",
            "Ancient Warrior Armor",
            "Customized Fight Uniform"
        ]
        self.makeup = [
            "None",
            "Traditional Face Paint",
            "Ritual Markings",
            "Modern Stylish",
            "War Paint",
            "Stage Performance",
            "Cultural Symbols",
            "Battle Scars"
        ]
        self.accessories = [
            "Sacred Beads",
            "Power-limiting Bracers",
            "Family Heirloom",
            "Tech Gadgets",
            "Mystical Ornaments",
            "Traditional Jewelry",
            "Combat Belt",
            "Energy-channeling Crystals"
        ]

    @classmethod
    def INPUT_TYPES(cls):
        """Define input types for the node"""
        return {
            "required": {
                "shuffle_mode": ("BOOLEAN", {"default": True, "label": "Shuffle Mode"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "gender": (cls().genders,),
                "fighting_style": (cls().fighting_styles,),
                "nationality": (cls().nationalities,),
                "personality": (cls().personalities,),
                "occupation": (cls().occupations,),
                "special_power": (cls().special_powers,),
                "weapon": (cls().weapons,),
                "clothing": (cls().clothing_styles,),
                "makeup": (cls().makeup,),
                "accessories": (cls().accessories,)
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")  # Prompt and Selected Values
    RETURN_NAMES = ("prompt", "selected_values")
    FUNCTION = "generate"
    OUTPUT_NODE = True
    CATEGORY = "prompt/generators"

    def generate(self, shuffle_mode, seed, gender=None, fighting_style=None, nationality=None, 
                personality=None, occupation=None, special_power=None, weapon=None, 
                clothing=None, makeup=None, accessories=None):
        import random
        
        # Set seed for reproducibility
        random.seed(seed)
        
        # Dictionary to store final values (either selected or random)
        character = {
            "Gender": gender if not shuffle_mode and gender else random.choice(self.genders),
            "Fighting Style": fighting_style if not shuffle_mode and fighting_style else random.choice(self.fighting_styles),
            "Nationality": nationality if not shuffle_mode and nationality else random.choice(self.nationalities),
            "Personality": personality if not shuffle_mode and personality else random.choice(self.personalities),
            "Occupation": occupation if not shuffle_mode and occupation else random.choice(self.occupations),
            "Special Power": special_power if not shuffle_mode and special_power else random.choice(self.special_powers),
            "Weapon": weapon if not shuffle_mode and weapon else random.choice(self.weapons),
            "Clothing": clothing if not shuffle_mode and clothing else random.choice(self.clothing_styles),
            "Makeup": makeup if not shuffle_mode and makeup else random.choice(self.makeup),
            "Accessories": accessories if not shuffle_mode and accessories else random.choice(self.accessories)
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

