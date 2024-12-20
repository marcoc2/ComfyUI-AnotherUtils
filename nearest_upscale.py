import torch
import numpy as np

class NearestUpscaleNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "scale_factor": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 8,
                    "step": 1
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "upscale"
    CATEGORY = "image/upscaling"

    def upscale(self, image, scale_factor):
        # Debug inicial do tensor de entrada
        print(f"Tipo de entrada: {type(image)}")
        print(f"Shape do tensor de entrada: {image.shape}")
        
        # Converter tensor para numpy array
        if isinstance(image, torch.Tensor):
            image_np = image[0].cpu().numpy()
        else:
            image_np = np.array(image)

        # Debug após conversão para numpy
        print(f"Shape do numpy array: {image_np.shape}")
        height, width = image_np.shape[:2]
        print(f"Altura: {height}, Largura: {width}")
        print(f"Fator de escala: {scale_factor}")
        
        # Calcular novas dimensões
        new_height = height * scale_factor
        new_width = width * scale_factor
        print(f"Nova altura calculada: {new_height}")
        print(f"Nova largura calculada: {new_width}")

        # Criar array de saída com as dimensões exatas
        upscaled = np.zeros((new_height, new_width, 3), dtype=np.float32)
        print(f"Shape do array de saída: {upscaled.shape}")

        # Upscaling usando repeat nativo do numpy
        # Primeiro expandimos na direção vertical
        temp = np.repeat(image_np, scale_factor, axis=0)
        # Depois na horizontal
        upscaled = np.repeat(temp, scale_factor, axis=1)
        
        print(f"Shape final antes do tensor: {upscaled.shape}")
        
        # Converter para tensor mantendo as dimensões exatas
        upscaled_tensor = torch.from_numpy(upscaled).unsqueeze(0)
        print(f"Shape final do tensor: {upscaled_tensor.shape}")
        
        return (upscaled_tensor,)

# Para registrar o nó
NODE_CLASS_MAPPINGS = {
    "NearestUpscale": NearestUpscaleNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "NearestUpscale": "Nearest Neighbor Upscale"
}