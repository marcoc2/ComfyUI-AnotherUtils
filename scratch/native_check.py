import torch
import sys

output_file = r"f:\AppsCrucial\ComfyUI_phoenix3\ComfyUI\custom_nodes\ComfyUI-AnotherUtils\scratch\native_nested_check.txt"

with open(output_file, "w") as f:
    try:
        # Create a dummy native nested tensor
        v = torch.randn((1, 12, 10, 32, 32))
        a = torch.randn((1, 8, 10, 16))
        # Note: torch.nested_tensor requires same dims usually, but let's see
        try:
            nt = torch.nested.nested_tensor([v, a])
            f.write(f"Native NT created. is_nested: {nt.is_nested}\n")
            f.write(f"Shape: {nt.shape}\n")
        except Exception as e:
            f.write(f"Native NT creation failed: {e}\n")
            
        # Check if samples.is_nested exists on standard tensors
        f.write(f"Standard tensor is_nested: {v.is_nested}\n")
    except Exception as e:
        f.write(f"Error: {e}")
