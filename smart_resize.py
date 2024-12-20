import torch
import numpy as np
from PIL import Image
from scipy import stats
from collections import Counter

class SmartResizeNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_size": ("INT", {
                    "default": 1024,
                    "min": 64,
                    "max": 8192,
                    "step": 64
                }),
                "border_sample_size": ("INT", {
                    "default": 5,
                    "min": 1,
                    "max": 50,
                    "step": 1,
                    "description": "Pixels to sample for border color"
                }),
                "color_method": (["mean", "mode"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "process_image"
    CATEGORY = "image/processing"

    def get_border_color_mean(self, img_np, edge, sample_size):
        h, w = img_np.shape[:2]
        
        if edge in ['top', 'bottom']:
            rows = slice(0, sample_size) if edge == 'top' else slice(h-sample_size, h)
            return np.mean(img_np[rows, :], axis=(0, 1))
        else:
            cols = slice(0, sample_size) if edge == 'left' else slice(w-sample_size, w)
            return np.mean(img_np[:, cols], axis=(0, 1))

    def get_border_color_mode(self, img_np, edge, sample_size):
        h, w = img_np.shape[:2]
        
        # Extrair a região da borda
        if edge in ['top', 'bottom']:
            rows = slice(0, sample_size) if edge == 'top' else slice(h-sample_size, h)
            border_pixels = img_np[rows, :]
        else:
            cols = slice(0, sample_size) if edge == 'left' else slice(w-sample_size, w)
            border_pixels = img_np[:, cols]
        
        # Reshapear para ter uma lista de pixels
        pixels = border_pixels.reshape(-1, 3)
        
        # Arredondar para reduzir ruído e facilitar encontrar cores iguais
        pixels = np.round(pixels * 255) / 255
        
        # Usar Counter para encontrar a cor mais frequente
        pixel_tuples = [tuple(p) for p in pixels]
        most_common_color = Counter(pixel_tuples).most_common(1)[0][0]
        
        # Converter de volta para array numpy
        return np.array(most_common_color, dtype=np.float32)

    def get_border_color(self, img_np, edge, sample_size, method='mean'):
        if method == 'mean':
            return self.get_border_color_mean(img_np, edge, sample_size)
        else:  # method == 'mode'
            return self.get_border_color_mode(img_np, edge, sample_size)

    def process_image(self, image, target_size, border_sample_size, color_method):
        # Converter tensor para numpy
        if isinstance(image, torch.Tensor):
            image_np = image[0].cpu().numpy()
        else:
            image_np = np.array(image)

        # Garantir que estamos trabalhando com valores entre 0 e 1
        if image_np.max() > 1.0:
            image_np = image_np.astype(np.float32) / 255.0

        height, width = image_np.shape[:2]
        current_image = image_np.copy()

        # Processar largura
        if width > target_size:
            # Cortar o excesso da largura
            start_x = (width - target_size) // 2
            current_image = current_image[:, start_x:start_x + target_size]
        elif width < target_size:
            # Criar nova imagem com a largura correta
            temp_image = np.zeros((height, target_size, 3), dtype=np.float32)
            
            # Calcular padding
            padding_left = (target_size - width) // 2
            
            # Pegar cores das bordas
            left_color = self.get_border_color(current_image, 'left', border_sample_size, color_method)
            right_color = self.get_border_color(current_image, 'right', border_sample_size, color_method)
            
            # Preencher as bordas
            temp_image[:, :padding_left] = left_color
            temp_image[:, padding_left:padding_left + width] = current_image
            temp_image[:, padding_left + width:] = right_color
            
            current_image = temp_image

        # Processar altura
        current_height = current_image.shape[0]
        if current_height > target_size:
            # Cortar o excesso da altura
            start_y = (current_height - target_size) // 2
            current_image = current_image[start_y:start_y + target_size]
        elif current_height < target_size:
            # Criar nova imagem com a altura correta
            temp_image = np.zeros((target_size, target_size, 3), dtype=np.float32)
            
            # Calcular padding
            padding_top = (target_size - current_height) // 2
            
            # Pegar cores das bordas
            top_color = self.get_border_color(current_image, 'top', border_sample_size, color_method)
            bottom_color = self.get_border_color(current_image, 'bottom', border_sample_size, color_method)
            
            # Preencher as bordas
            temp_image[:padding_top] = top_color
            temp_image[padding_top:padding_top + current_height] = current_image
            temp_image[padding_top + current_height:] = bottom_color
            
            current_image = temp_image

        # Converter de volta para tensor
        return (torch.from_numpy(current_image).unsqueeze(0),)