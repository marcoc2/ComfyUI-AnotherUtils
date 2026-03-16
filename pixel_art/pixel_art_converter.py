"""
Pixel Art Converter Node for ComfyUI
Converts images to pixel art and detects scaling factor
"""

import cv2
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import torch

class PixelArtConverterNode:
    def __init__(self):
        pass
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "palette_type": (["muito_limitada", "limitada", "muitas", "sem"],),
                "min_size": ("INT", {"default": 2, "min": 1, "max": 32}),
                "max_size": ("INT", {"default": 32, "min": 2, "max": 64}),
            },
        }

    RETURN_TYPES = ("IMAGE", "FLOAT")
    FUNCTION = "convert_to_pixel_art"
    CATEGORY = "image/processing"

    def detect_edges(self, image):
        """Detecta bordas usando o algoritmo Canny com pré-processamento Gaussiano."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(blurred, 50, 150, L2gradient=True)
        return edges

    def analyze_edge_spacing(self, edges, min_distance=2):
        """Analisa o espaçamento entre bordas usando autocorrelação e transformada de Hough."""
        distances = []
        for y in range(edges.shape[0]):
            edge_pos = np.where(edges[y, :] > 0)[0]
            if len(edge_pos) > 1:
                d = np.diff(edge_pos)
                d = d[d >= min_distance]
                distances.extend(d.tolist())
        
        if len(distances) < 10:
            return 0
        
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, 
                               minLineLength=20, maxLineGap=10)
        
        if lines is not None:
            line_distances = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                line_distances.append(length)
            
            if line_distances:
                distances.extend(line_distances)
        
        try:
            hist, bins = np.histogram(distances, bins=50)
            peak_idx = np.argmax(hist)
            most_common_distance = (bins[peak_idx] + bins[peak_idx + 1]) / 2
            return float(most_common_distance)
        except:
            return 0

    def detect_color_clusters(self, image, n_samples=1000):
        """Detecta clusters naturais de cores na imagem."""
        pixels = image.reshape(-1, 3)
        
        if len(pixels) > n_samples:
            indices = np.random.choice(len(pixels), n_samples, replace=False)
            pixels = pixels[indices]
        
        pixels_normalized = pixels.astype(np.float32) / 255.0
        
        distortions = []
        K = range(1, 10)
        for k in K:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(pixels_normalized)
            distortions.append(kmeans.inertia_)
        
        diffs = np.diff(distortions)
        elbow = np.argmin(np.abs(diffs - np.mean(diffs))) + 1
        
        return min(elbow + 2, 8)

    def estimate_pixel_size_by_frequency(self, image, min_size=2, max_size=32):
        """Estima tamanho do pixel usando análise de frequência espacial."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        
        rows, cols = magnitude_spectrum.shape
        center_row, center_col = rows//2, cols//2
        
        peak_distances = []
        for radius in range(min_size, min(center_row, center_col), 2):
            y, x = np.ogrid[-center_row:rows-center_row, -center_col:cols-center_col]
            mask = (x*x + y*y <= radius*radius) & (x*x + y*y > (radius-2)**2)
            ring_values = magnitude_spectrum[mask]
            if len(ring_values) > 0:
                peak_distances.append((radius, np.max(ring_values)))
        
        if not peak_distances:
            return None
            
        peak_distances.sort(key=lambda x: x[1], reverse=True)
        estimated_size = peak_distances[0][0]
        
        return int(np.clip(estimated_size, min_size, max_size))

    def adaptive_color_quantization(self, image, palette_type):
        """Quantização adaptativa de cores com dithering opcional."""
        palette_sizes = {
            'muito_limitada': 16,
            'limitada': 32,
            'muitas': 64,
            'sem': None
        }
        
        if palette_type not in palette_sizes or palette_sizes[palette_type] is None:
            return image
        
        n_colors = palette_sizes[palette_type]
        
        if palette_type == 'sem':
            n_colors = self.detect_color_clusters(image)
        
        pixels = image.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 300, 0.001)
        
        best_inertia = float('inf')
        best_centers = None
        best_labels = None
        
        for _ in range(3):
            _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10,
                                          cv2.KMEANS_PP_CENTERS)
            inertia = np.sum((pixels - centers[labels.flatten()])**2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        
        if palette_type != 'sem':
            quantized = best_centers[best_labels.flatten()].reshape(image.shape)
            h, w = image.shape[:2]
            for y in range(h-1):
                for x in range(w-1):
                    old_pixel = quantized[y, x].astype(np.float32)
                    new_pixel = best_centers[best_labels[y*w + x]]
                    error = old_pixel - new_pixel
                    
                    if x < w-1:
                        quantized[y, x+1] = np.clip(quantized[y, x+1] + error * 7/16, 0, 255)
                    if x > 0 and y < h-1:
                        quantized[y+1, x-1] = np.clip(quantized[y+1, x-1] + error * 3/16, 0, 255)
                    if y < h-1:
                        quantized[y+1, x] = np.clip(quantized[y+1, x] + error * 5/16, 0, 255)
                    if x < w-1 and y < h-1:
                        quantized[y+1, x+1] = np.clip(quantized[y+1, x+1] + error * 1/16, 0, 255)
            
            return quantized.astype(np.uint8)
        
        return best_centers[best_labels.flatten()].reshape(image.shape)

    def create_pixel_perfect_art(self, image, block_size, palette_type='limitada'):
        """Cria a arte pixel perfect final com melhor preservação de detalhes."""
        h, w, _ = image.shape
        new_h = h - (h % block_size)
        new_w = w - (w % block_size)
        
        if new_h == 0 or new_w == 0:
            raise ValueError(f"Tamanho de bloco {block_size} incompatível com dimensões {h}x{w}")
        
        cropped = image[:new_h, :new_w]
        out_h = new_h // block_size
        out_w = new_w // block_size
        
        pixel_art = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        
        for i in range(0, new_h, block_size):
            for j in range(0, new_w, block_size):
                block = cropped[i:i+block_size, j:j+block_size]
                
                mean_color = np.mean(block.reshape(-1, 3), axis=0)
                median_color = np.median(block.reshape(-1, 3), axis=0)
                
                block_variance = np.var(block.reshape(-1, 3), axis=0)
                
                if np.mean(block_variance) > 100:
                    final_color = median_color
                else:
                    final_color = mean_color
                    
                pixel_art[i // block_size, j // block_size] = final_color
        
        return self.adaptive_color_quantization(pixel_art, palette_type)

    def estimate_optimal_pixel_size(self, image, min_size=2, max_size=32):
        """Combina múltiplos métodos para estimativa final mais robusta."""
        freq_size = self.estimate_pixel_size_by_frequency(image, min_size, max_size)
        edges = self.detect_edges(image)
        edge_size = self.analyze_edge_spacing(edges)
        
        weights = {
            'freq': 0.4,
            'edge': 0.6
        }
        
        estimates = []
        if freq_size is not None:
            estimates.append((freq_size, weights['freq']))
        if edge_size is not None and edge_size > 0:
            estimates.append((edge_size, weights['edge']))
        
        if not estimates:
            return min_size
        
        weighted_sum = sum(size * weight for size, weight in estimates)
        total_weight = sum(weight for _, weight in estimates)
        
        estimated_size = int(round(weighted_sum / total_weight))
        
        return int(np.clip(estimated_size, min_size, max_size))

    def convert_to_pixel_art(self, image, palette_type, min_size=2, max_size=32):
        """Método principal do node que processa a imagem."""
        # Converte tensor PyTorch para numpy array
        image_np = image.cpu().numpy()[0]  # Pega primeira imagem do batch
        image_np = (image_np * 255).astype(np.uint8)
        
        # Converte de RGB para BGR para processamento OpenCV
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Estima tamanho ótimo do pixel
        optimal_size = self.estimate_optimal_pixel_size(image_np, min_size, max_size)
        
        # Cria pixel art
        pixel_art = self.create_pixel_perfect_art(image_np, optimal_size, palette_type)
        
        # Converte de volta para RGB
        pixel_art = cv2.cvtColor(pixel_art, cv2.COLOR_BGR2RGB)
        
        # Converte para tensor PyTorch
        pixel_art_tensor = torch.from_numpy(pixel_art).float() / 255.0
        pixel_art_tensor = pixel_art_tensor.unsqueeze(0)  # Adiciona dimensão do batch
        
        return (pixel_art_tensor, float(optimal_size))