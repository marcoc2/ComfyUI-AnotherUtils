import numpy as np
import torch
from PIL import Image
import cv2
from sklearn.cluster import KMeans

class PixelArtNormalizerNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "block_size": ("INT", {
                    "default": 4,
                    "min": 0,
                    "max": 8,
                    "step": 1,
                    "description": "0 for auto-detection"
                }),
                "n_colors": ("INT", {
                    "default": 32,
                    "min": 0,
                    "max": 256,
                    "step": 1,
                    "description": "0 for auto-detection"
                })
            }
        }
    
    RETURN_TYPES = ("IMAGE", "INT", "IMAGE")  # imagem normal, tamanho do bloco, imagem downscaled
    RETURN_NAMES = ("normalized", "block_size", "downscaled")
    FUNCTION = "normalize_pixel_art"
    CATEGORY = "image/processing"

    def detect_grid(self, image):
        """Detecta o tamanho aproximado dos pixels na grade."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Detecta linhas
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=20, maxLineGap=5)
        
        if lines is None:
            return 2  # valor padrão se não detectar
            
        # Calcula distâncias entre linhas paralelas
        distances = []
        for i in range(len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            for j in range(i + 1, len(lines)):
                x3, y3, x4, y4 = lines[j][0]
                
                angle1 = np.arctan2(y2 - y1, x2 - x1)
                angle2 = np.arctan2(y4 - y3, x4 - x3)
                if abs(angle1 - angle2) < 0.1:
                    dist = abs((y4 - y3) * x1 - (x4 - x3) * y1 + x4 * y3 - y4 * x3) / \
                          np.sqrt((y4 - y3)**2 + (x4 - x3)**2)
                    if dist > 2:
                        distances.append(dist)
        
        if not distances:
            return 2
            
        grid_size = int(np.median(distances))
        return max(2, min(grid_size, 8))  # limita entre 2 e 8 pixels

    def normalize_to_grid(self, image, grid_size):
        h, w = image.shape[:2]
        
        # Ajusta dimensões para serem múltiplos do grid_size
        new_h = ((h + grid_size - 1) // grid_size) * grid_size
        new_w = ((w + grid_size - 1) // grid_size) * grid_size
        
        # Cria nova imagem com padding se necessário
        normalized = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        normalized[:h, :w] = image
        
        # Para cada célula da grade
        for y in range(0, new_h, grid_size):
            for x in range(0, new_w, grid_size):
                # Limita as coordenadas aos limites da imagem original
                y_end = min(y + grid_size, h)
                x_end = min(x + grid_size, w)
                
                # Pega o bloco atual
                block = normalized[y:y_end, x:x_end]
                
                if block.size > 0:
                    # Encontra a cor mais frequente no bloco
                    block_reshaped = block.reshape(-1, 3)
                    unique_colors, counts = np.unique(block_reshaped, axis=0, return_counts=True)
                    dominant_color = unique_colors[counts.argmax()]
                    
                    # Preenche o bloco com a cor dominante
                    normalized[y:y_end, x:x_end] = dominant_color
        
        return normalized[:h, :w]

    def quantize_colors(self, image, n_colors):
        h, w = image.shape[:2]
        pixels = image.reshape(-1, 3)
        
        kmeans = KMeans(n_clusters=n_colors, random_state=42)
        labels = kmeans.fit_predict(pixels)
        palette = kmeans.cluster_centers_.astype(np.uint8)
        
        quantized = palette[labels].reshape(h, w, 3)
        return quantized

    def normalize_pixel_art(self, image, block_size, n_colors):
        # Converter tensor para numpy array
        if isinstance(image, torch.Tensor):
            if image.dim() == 4:
                image_np = image[0].cpu().numpy()
            else:
                image_np = image.cpu().numpy()
        else:
            image_np = np.array(image)

        # Converter para uint8 se estiver normalizado entre 0-1
        if image_np.max() <= 1.0:
            image_np = (image_np * 255).astype(np.uint8)
        
        # Usar detecção automática se block_size ou n_colors forem 0
        if n_colors <= 0:
            n_colors = min(32, max(8, int(np.sqrt(image_np.shape[0] * image_np.shape[1] / 100))))
            print(f"Número de cores detectado automaticamente: {n_colors}")
        
        # Quantizar cores
        quantized = self.quantize_colors(image_np, n_colors)
        
        # Detectar tamanho do grid se block_size for 0
        detected_block_size = self.detect_grid(quantized) if block_size <= 0 else block_size
        print(f"Tamanho do grid: {detected_block_size}px")
        
        # Normalizar para a grade
        normalized = self.normalize_to_grid(quantized, detected_block_size)
        
        # Criar versão downscaled
        h, w = normalized.shape[:2]
        new_h = h // detected_block_size
        new_w = w // detected_block_size
        
        # Usar área de cada bloco para determinar a cor do pixel correspondente
        downscaled = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        for y in range(new_h):
            for x in range(new_w):
                block = normalized[y*detected_block_size:(y+1)*detected_block_size, 
                                x*detected_block_size:(x+1)*detected_block_size]
                downscaled[y, x] = block[0, 0]  # Como o bloco já está normalizado, podemos pegar qualquer pixel
        
        # Converter ambas as imagens para float32 normalizado
        normalized = normalized.astype(np.float32) / 255.0
        downscaled = downscaled.astype(np.float32) / 255.0
        
        # Converter para tensores
        normalized_tensor = torch.from_numpy(normalized).unsqueeze(0)
        downscaled_tensor = torch.from_numpy(downscaled).unsqueeze(0)
        
        return (normalized_tensor, detected_block_size, downscaled_tensor)

# Registrar o nó
NODE_CLASS_MAPPINGS = {
    "PixelArtNormalizer": PixelArtNormalizerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelArtNormalizer": "Pixel Art Normalizer"
}