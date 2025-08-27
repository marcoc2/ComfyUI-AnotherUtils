import numpy as np
import torch
import json
import cv2
import math
from PIL import Image

class WalkingPoseGenerator:
    def __init__(self):
        # Cores e conexões do OpenPose conforme util.py
        self.colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0],
                      [170, 255, 0], [85, 255, 0], [0, 255, 0], [0, 255, 85],
                      [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255],
                      [0, 0, 255], [85, 0, 255], [170, 0, 255], [255, 0, 255],
                      [255, 0, 170], [255, 0, 85]]
        
        self.limbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9],
                        [9, 10], [10, 11], [2, 12], [12, 13], [13, 14], [2, 1],
                        [1, 15], [15, 17], [1, 16], [16, 18]]
        
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "num_frames": ("INT", {"default": 8, "min": 2, "max": 16}),
                "canvas_width": ("INT", {"default": 512, "min": 256, "max": 2048}),
                "canvas_height": ("INT", {"default": 512, "min": 256, "max": 2048}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "generate_walking_poses"
    CATEGORY = "pose/animation"

    def create_base_pose(self):
        center_x = 256
        base_y = 150
        base_keypoints = []
        
        # Cabeça e pescoço (keypoints 0-1)
        base_keypoints.extend([
            {"x": center_x, "y": base_y, "score": 0.9},      # 0: Nariz
            {"x": center_x, "y": base_y + 30, "score": 0.9}, # 1: Pescoço/Centro
        ])
        
        # Ombros e braços direitos (keypoints 2-4)
        shoulder_width = 50
        base_keypoints.extend([
            {"x": center_x - shoulder_width, "y": base_y + 30, "score": 0.9},  # 2: Ombro direito
            {"x": center_x - shoulder_width, "y": base_y + 80, "score": 0.9},  # 3: Cotovelo direito
            {"x": center_x - shoulder_width, "y": base_y + 130, "score": 0.9}, # 4: Pulso direito
        ])
        
        # Ombros e braços esquerdos (keypoints 5-7)
        base_keypoints.extend([
            {"x": center_x + shoulder_width, "y": base_y + 30, "score": 0.9},  # 5: Ombro esquerdo
            {"x": center_x + shoulder_width, "y": base_y + 80, "score": 0.9},  # 6: Cotovelo esquerdo
            {"x": center_x + shoulder_width, "y": base_y + 130, "score": 0.9}, # 7: Pulso esquerdo
        ])
        
        # Quadril (keypoint 8)
        hip_y = base_y + 130
        base_keypoints.append(
            {"x": center_x, "y": hip_y, "score": 0.9}  # 8: Quadril
        )
        
        # Pernas (keypoints 9-14)
        leg_width = 30
        base_keypoints.extend([
            {"x": center_x - leg_width, "y": hip_y + 50, "score": 0.9},   # 9: Coxa direita
            {"x": center_x - leg_width, "y": hip_y + 110, "score": 0.9},  # 10: Joelho direito
            {"x": center_x - leg_width, "y": hip_y + 170, "score": 0.9},  # 11: Tornozelo direito
            {"x": center_x + leg_width, "y": hip_y + 50, "score": 0.9},   # 12: Coxa esquerda
            {"x": center_x + leg_width, "y": hip_y + 110, "score": 0.9},  # 13: Joelho esquerdo
            {"x": center_x + leg_width, "y": hip_y + 170, "score": 0.9},  # 14: Tornozelo esquerdo
        ])
        
        # Olhos e orelhas (keypoints 15-18)
        eye_width = 15
        eye_height = 10
        base_keypoints.extend([
            {"x": center_x - eye_width, "y": base_y - eye_height, "score": 0.9},  # 15: Olho direito
            {"x": center_x + eye_width, "y": base_y - eye_height, "score": 0.9},  # 16: Olho esquerdo
            {"x": center_x - eye_width*2, "y": base_y, "score": 0.9},            # 17: Orelha direita
            {"x": center_x + eye_width*2, "y": base_y, "score": 0.9},            # 18: Orelha esquerda
        ])
        return base_keypoints

    def animate_pose(self, base_keypoints, frame, total_frames):
        animated_keypoints = []
        phase = (frame / total_frames) * 2 * np.pi
        
        for i, kp in enumerate(base_keypoints):
            new_kp = kp.copy()
            
            # Pernas (keypoints 9-14)
            if 9 <= i <= 14:
                if i <= 11:  # Perna direita
                    leg_phase = phase
                else:  # Perna esquerda
                    leg_phase = phase + np.pi
                    
                if i in [9, 12]:  # Coxas
                    new_kp["x"] += np.sin(leg_phase) * 15
                    new_kp["y"] += -np.abs(np.sin(leg_phase)) * 10
                elif i in [10, 13]:  # Joelhos
                    new_kp["x"] += np.sin(leg_phase) * 25
                    new_kp["y"] += -np.abs(np.sin(leg_phase)) * 20
                elif i in [11, 14]:  # Tornozelos
                    new_kp["x"] += np.sin(leg_phase) * 35
                    new_kp["y"] += -np.abs(np.sin(leg_phase)) * 30
            
            # Braços (keypoints 2-7)
            elif 2 <= i <= 7:
                if i <= 4:  # Braço direito
                    arm_phase = phase + np.pi
                else:  # Braço esquerdo
                    arm_phase = phase
                    
                if i in [2, 5]:  # Ombros
                    new_kp["x"] += np.sin(arm_phase) * 5
                elif i in [3, 6]:  # Cotovelos
                    new_kp["x"] += np.sin(arm_phase) * 10
                    new_kp["y"] += np.cos(arm_phase) * 5
                elif i in [4, 7]:  # Pulsos
                    new_kp["x"] += np.sin(arm_phase) * 15
                    new_kp["y"] += np.cos(arm_phase) * 10
            
            # Ajuste sutil do tronco
            elif i in [1, 8]:  # Pescoço e quadril
                new_kp["x"] += np.sin(phase) * 5
                new_kp["y"] += -np.abs(np.sin(phase) * 3)
                
            animated_keypoints.append(new_kp)
            
        return animated_keypoints

    def draw_pose(self, keypoints, canvas_width, canvas_height):
        # Criar canvas preto (como no OpenPose)
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        stickwidth = 4
        
        # Desenhar conexões entre keypoints com cores do OpenPose
        for (k1_index, k2_index), color in zip(self.limbSeq, self.colors):
            kp1 = keypoints[k1_index - 1]
            kp2 = keypoints[k2_index - 1]

            if kp1 and kp2:
                y = np.array([kp1["x"], kp2["x"]])
                x = np.array([kp1["y"], kp2["y"]])
                mx = np.mean(x)
                my = np.mean(y)
                length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
                angle = math.degrees(math.atan2(x[0] - x[1], y[0] - y[1]))
                polygon = cv2.ellipse2Poly((int(my), int(mx)), 
                                         (int(length/2), stickwidth), 
                                         int(angle), 0, 360, 1)
                cv2.fillConvexPoly(canvas, polygon, [int(float(c)) for c in color])

        # Desenhar pontos dos keypoints com cores do OpenPose
        for kp, color in zip(keypoints, self.colors):
            if kp:
                x, y = int(kp["x"]), int(kp["y"])
                cv2.circle(canvas, (x, y), 4, color, thickness=-1)

        # Converter para tensor
        img_array = canvas.astype(np.float32) / 255.0
        tensor = torch.from_numpy(img_array)[None,]
        return tensor

    def generate_walking_poses(self, num_frames, canvas_width, canvas_height):
        base_pose = self.create_base_pose()
        poses_batch = []
        
        for frame in range(num_frames):
            animated_pose = self.animate_pose(base_pose, frame, num_frames)
            pose_tensor = self.draw_pose(animated_pose, canvas_width, canvas_height)
            poses_batch.append(pose_tensor)
            
        # Combinar todos os frames
        batch = torch.cat(poses_batch, dim=0)
        return (batch,)

NODE_CLASS_MAPPINGS = {
    "WalkingPoseGenerator": WalkingPoseGenerator
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WalkingPoseGenerator": "Walking Pose Generator"
}