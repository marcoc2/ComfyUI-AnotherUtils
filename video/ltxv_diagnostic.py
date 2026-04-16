import torch
import logging

logger = logging.getLogger(__name__)

class LTXVDiagnosticNode:
    """
    Diagnostic node to inspect LTX Video latent structures (NestedTensors, shapes, etc.)
    Place this before Audio Decode or Sampler to see what's happening inside.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "node_name": ("STRING", {"default": "LTXV Diagnostic"}),
                "print_to_console": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "info")
    FUNCTION = "inspect"
    CATEGORY = "AnotherUtils/video"

    def inspect(self, latent, node_name, print_to_console):
        info_lines = [f"--- {node_name} Diagnostic ---"]
        
        samples = latent.get("samples", None)
        if samples is None:
            info_lines.append("ERROR: No 'samples' found in latent.")
        else:
            info_lines.append(f"Samples Type: {type(samples)}")
            
            # Check for ComfyUI NestedTensor wrapper
            is_wrapper = False
            try:
                from comfy.nested_tensor import NestedTensor as ComfyNested
                if isinstance(samples, ComfyNested):
                    is_wrapper = True
                    info_lines.append("Format: ComfyUI NestedTensor Wrapper")
                    for i, t in enumerate(samples.tensors):
                        info_lines.append(f"  Inner Tensor {i} Shape: {list(t.shape)} | dtype: {t.dtype}")
            except ImportError:
                pass

            # Check for Native PyTorch Nested Tensor
            if not is_wrapper and hasattr(samples, "is_nested"):
                if samples.is_nested:
                    info_lines.append("Format: Native PyTorch Nested Tensor")
                    try:
                        # Try to unbind to see components
                        unbound = samples.unbind()
                        for i, t in enumerate(unbound):
                            info_lines.append(f"  Unbound Component {i} Shape: {list(t.shape)} | dtype: {t.dtype}")
                    except Exception as e:
                        info_lines.append(f"  Error unbinding native nested tensor: {e}")
                else:
                    info_lines.append(f"Format: Standard Tensor | Shape: {list(samples.shape)} | dtype: {samples.dtype}")
            elif not is_wrapper:
                info_lines.append(f"Format: Standard Tensor | Shape: {list(samples.shape)} | dtype: {samples.dtype}")

        # Check for noise_mask
        mask = latent.get("noise_mask", None)
        if mask is not None:
            info_lines.append(f"Noise Mask Type: {type(mask)}")
            if hasattr(mask, "shape"):
                info_lines.append(f"  Mask Shape: {list(mask.shape)}")
            elif is_wrapper:
                 info_lines.append("  Mask is likely also a Nested Wrapper (checking components...)")
                 # Many LTX implementations wrap mask too
        else:
            info_lines.append("Noise Mask: Not present")

        # Other keys
        other_keys = [k for k in latent.keys() if k not in ["samples", "noise_mask"]]
        if other_keys:
            info_lines.append(f"Other Keys: {other_keys}")

        info_text = "\n".join(info_lines)
        if print_to_console:
            print(info_text)
            
        return (latent, info_text)
