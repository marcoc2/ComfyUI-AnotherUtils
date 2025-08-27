"""
Pixel Art Converter Node for ComfyUI with thread-based parallelization
"""

import cv2
import numpy as np
from scipy import signal
from sklearn.cluster import KMeans
import torch
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import threading

class PixelArtConverterNodeParallel:
    def __init__(self):
        # Use ThreadPoolExecutor instead of ProcessPoolExecutor
        self.num_workers = min(32, (threading.active_count() + 1) * 2)
        
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
        """Edge detection with parallel Gaussian blur processing."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        def process_blur_chunk(chunk):
            return cv2.GaussianBlur(chunk, (3, 3), 0)
        
        # Split image into chunks for parallel processing
        chunks = np.array_split(gray, self.num_workers)
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            blurred_chunks = list(executor.map(process_blur_chunk, chunks))
        
        blurred = np.concatenate(blurred_chunks)
        edges = cv2.Canny(blurred, 50, 150, L2gradient=True)
        return edges

    def process_edge_chunk(self, chunk, min_distance=2):
        """Process a chunk of edges for edge spacing analysis."""
        try:
            distances = []
            for row in chunk:
                edge_pos = np.where(row > 0)[0]
                if len(edge_pos) > 1:
                    d = np.diff(edge_pos)
                    d = d[d >= min_distance]
                    distances.extend(d.tolist())
            return distances
        except Exception as e:
            print(f"Warning: Edge chunk processing failed: {str(e)}")
            return []

    def analyze_edge_spacing(self, edges, min_distance=2):
        """Thread-based edge spacing analysis."""
        try:
            # Split edges into chunks
            chunks = np.array_split(edges, self.num_workers)
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                futures = [
                    executor.submit(self.process_edge_chunk, chunk, min_distance)
                    for chunk in chunks
                ]
                
                # Collect results with timeout
                distances = []
                for future in futures:
                    try:
                        chunk_distances = future.result(timeout=10)
                        distances.extend(chunk_distances)
                    except Exception as e:
                        print(f"Warning: Edge analysis chunk failed: {str(e)}")
                        continue

            if len(distances) < 10:
                return 0
                
            try:
                hist, bins = np.histogram(distances, bins=50)
                peak_idx = np.argmax(hist)
                most_common_distance = (bins[peak_idx] + bins[peak_idx + 1]) / 2
                return float(most_common_distance)
            except Exception as e:
                print(f"Warning: Histogram analysis failed: {str(e)}")
                return 0
                
        except Exception as e:
            print(f"Error in edge spacing analysis: {str(e)}")
            return 0

    def detect_color_clusters(self, image, n_samples=1000):
        """Thread-safe color cluster detection."""
        try:
            pixels = image.reshape(-1, 3)
            
            if len(pixels) > n_samples:
                indices = np.random.choice(len(pixels), n_samples, replace=False)
                pixels = pixels[indices]
            
            pixels_normalized = pixels.astype(np.float32) / 255.0
            
            def process_k(k):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    kmeans.fit(pixels_normalized)
                    return k, kmeans.inertia_
                except Exception as e:
                    print(f"Warning: K-means clustering failed for k={k}: {str(e)}")
                    return k, float('inf')
            
            with ThreadPoolExecutor(max_workers=min(9, self.num_workers)) as executor:
                results = list(executor.map(process_k, range(1, 10)))
            
            # Filter out failed results
            valid_results = [(k, inertia) for k, inertia in results if inertia != float('inf')]
            if not valid_results:
                return 8  # fallback value
            
            distortions = [inertia for _, inertia in valid_results]
            diffs = np.diff(distortions)
            elbow = np.argmin(np.abs(diffs - np.mean(diffs))) + 1
            
            return min(elbow + 2, 8)
            
        except Exception as e:
            print(f"Error in color cluster detection: {str(e)}")
            return 8  # fallback value

    def process_frequency_chunk(self, chunk, min_size, max_size):
        """Process a chunk for frequency analysis with error handling."""
        try:
            f = np.fft.fft2(chunk)
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
            
            return peak_distances
            
        except Exception as e:
            print(f"Warning: Frequency chunk processing failed: {str(e)}")
            return []

    def estimate_pixel_size_by_frequency(self, image, min_size=2, max_size=32):
        """Thread-based frequency analysis."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            
            # Reduce chunk size for better memory management
            max_chunk_size = 256
            chunk_size = min(max_chunk_size, gray.shape[0])
            overlap = chunk_size // 4
            
            # Process sequentially if image is small
            if gray.shape[0] * gray.shape[1] < 262144:  # 512x512
                peaks = self.process_frequency_chunk(gray, min_size, max_size)
                if peaks:
                    return int(np.clip(peaks[0][0], min_size, max_size))
                return min_size
            
            chunks = []
            for i in range(0, gray.shape[0] - chunk_size + 1, chunk_size - overlap):
                chunk = gray[i:i+chunk_size, :]
                chunks.append(chunk)
            
            with ThreadPoolExecutor(max_workers=min(4, self.num_workers)) as executor:
                futures = [
                    executor.submit(self.process_frequency_chunk, chunk, min_size, max_size)
                    for chunk in chunks
                ]
                
                all_peaks = []
                for future in futures:
                    try:
                        peaks = future.result(timeout=10)
                        all_peaks.extend(peaks)
                    except Exception as e:
                        print(f"Warning: Frequency analysis chunk failed: {str(e)}")
                        continue
            
            if not all_peaks:
                return min_size
            
            all_peaks.sort(key=lambda x: x[1], reverse=True)
            estimated_size = all_peaks[0][0]
            
            return int(np.clip(estimated_size, min_size, max_size))
            
        except Exception as e:
            print(f"Error in frequency analysis: {str(e)}")
            return min_size

    def process_block(self, args):
        """Process a single block for pixel art creation."""
        try:
            block, palette_type = args
            mean_color = np.mean(block.reshape(-1, 3), axis=0)
            median_color = np.median(block.reshape(-1, 3), axis=0)
            block_variance = np.var(block.reshape(-1, 3), axis=0)
            
            if np.mean(block_variance) > 100:
                return median_color
            return mean_color
        except Exception as e:
            print(f"Warning: Block processing failed: {str(e)}")
            return np.array([0, 0, 0])

    def create_pixel_perfect_art(self, image, block_size, palette_type='limitada'):
        """Thread-based pixel art creation."""
        try:
            h, w, _ = image.shape
            new_h = h - (h % block_size)
            new_w = w - (w % block_size)
            
            if new_h == 0 or new_w == 0:
                raise ValueError(f"Block size {block_size} incompatible with dimensions {h}x{w}")
            
            cropped = image[:new_h, :new_w]
            out_h = new_h // block_size
            out_w = new_w // block_size
            
            blocks = []
            for i in range(0, new_h, block_size):
                for j in range(0, new_w, block_size):
                    block = cropped[i:i+block_size, j:j+block_size]
                    blocks.append((block, palette_type))
            
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                colors = list(executor.map(self.process_block, blocks))
            
            pixel_art = np.array(colors).reshape(out_h, out_w, 3)
            return self.adaptive_color_quantization(pixel_art, palette_type)
            
        except Exception as e:
            print(f"Error in pixel art creation: {str(e)}")
            return image

    def adaptive_color_quantization(self, image, palette_type):
        """Thread-safe color quantization with error handling."""
        try:
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
            
            def kmeans_trial():
                try:
                    _, labels, centers = cv2.kmeans(pixels, n_colors, None, criteria, 10,
                                                  cv2.KMEANS_PP_CENTERS)
                    inertia = np.sum((pixels - centers[labels.flatten()])**2)
                    return inertia, labels, centers
                except Exception as e:
                    print(f"Warning: K-means trial failed: {str(e)}")
                    return float('inf'), None, None
            
            # Run K-means trials in parallel
            with ThreadPoolExecutor(max_workers=3) as executor:
                results = list(executor.map(lambda _: kmeans_trial(), range(3)))
            
            # Filter out failed trials
            valid_results = [(i, l, c) for i, l, c in results if i != float('inf')]
            if not valid_results:
                return image
                
            best_result = min(valid_results, key=lambda x: x[0])
            _, best_labels, best_centers = best_result
            
            if palette_type != 'sem':
                quantized = best_centers[best_labels.flatten()].reshape(image.shape)
                h, w = image.shape[:2]
                
                # Sequential dithering as it's difficult to parallelize effectively
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
            
        except Exception as e:
            print(f"Error in color quantization: {str(e)}")
            return image

    def convert_to_pixel_art(self, image, palette_type, min_size=2, max_size=32):
        """Main processing method with comprehensive error handling."""
        try:
            image_np = image.cpu().numpy()[0]
            image_np = (image_np * 255).astype(np.uint8)
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            
            optimal_size = self.estimate_optimal_pixel_size(image_np, min_size, max_size)
            pixel_art = self.create_pixel_perfect_art(image_np, optimal_size, palette_type)
            pixel_art = cv2.cvtColor(pixel_art, cv2.COLOR_BGR2RGB)
            
            pixel_art_tensor = torch.from_numpy(pixel_art).float() / 255.0
            pixel_art_tensor = pixel_art_tensor.unsqueeze(0)
            
            return (pixel_art_tensor, float(optimal_size))
            
        except Exception as e:
            print(f"Error in main conversion: {str(e)}")
            # Return original image on error
            return (image, float(min_size))

    def estimate_optimal_pixel_size(self, image, min_size=2, max_size=32):
        """Thread-safe optimal pixel size estimation."""
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                freq_future = executor.submit(
                    self.estimate_pixel_size_by_frequency, 
                    image, min_size, max_size
                )
                edges_future = executor.submit(self.detect_edges, image)
                
                # Get results with timeout
                try:
                    freq_size = freq_future.result(timeout=30)
                    edges = edges_future.result(timeout=30)
                    edge_size = self.analyze_edge_spacing(edges)
                except Exception as e:
                    print(f"Warning: Size estimation component failed: {str(e)}")
                    return min_size
            
            weights = {'freq': 0.4, 'edge': 0.6}
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
            
        except Exception as e:
            print(f"Error in optimal size estimation: {str(e)}")
            return min_size