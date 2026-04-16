class IndicesListTo50:
    """
    Takes a Python List of integers (e.g. from ImageListSampler's 'indices') 
    and splits it into 50 individual INT outputs.
    
    This allows you to bypass manual typing in nodes that have 50 individual 
    widgets (like LTXSequencer). Just Right Click their node -> 'Convert Widget to Input'
    and wire these outputs directly.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "indices": ("INT",),
            }
        }

    RETURN_TYPES = ("INT",) * 50
    RETURN_NAMES = tuple(f"index_{i+1}" for i in range(50))
    INPUT_IS_LIST = True
    FUNCTION = "split"
    CATEGORY = "AnotherUtils/logic"

    def split(self, indices):
        if not indices:
            return (0,) * 50
            
        # Unwrap list of lists if passed strangely
        all_indices = []
        for item in indices:
            if isinstance(item, list):
                all_indices.extend(item)
            else:
                all_indices.append(item)
                
        if not all_indices:
            return (0,) * 50

        # Pad individual outputs to exactly 50
        results = all_indices[:50]
        padded_results = results + [0] * (50 - len(results))

        return tuple(padded_results)
