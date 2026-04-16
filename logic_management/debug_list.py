import json

class AnotherShowList:
    """
    A debug node that displays a list of objects (like indices) as a formatted string in the UI.
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_list": ("*", {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "execute"
    CATEGORY = "AnotherUtils/Logic & Management"
    OUTPUT_NODE = True

    def execute(self, input_list):
        # Flatten the list if it's nested (common with INPUT_IS_LIST)
        if len(input_list) == 1 and isinstance(input_list[0], list):
            actual_list = input_list[0]
        else:
            actual_list = input_list

        # Convert list to a readable string
        try:
            result_string = str(actual_list)
        except Exception as e:
            result_string = f"Error converting list: {e}"

        print(f"[AnotherShowList] Displaying: {result_string}")

        # Return the string for the output slot and for the UI
        return {"ui": {"text": [result_string]}, "result": (result_string,)}
