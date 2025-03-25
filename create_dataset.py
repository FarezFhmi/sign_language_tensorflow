import os
import pickle
import numpy as np
import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './data'

# Data storage
data = []
labels = []

# Get folders and sort them numerically
folders = [f for f in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, f))]
folders.sort(key=lambda x: int(x))  # Numerical sort

for dir_ in folders:
    dir_path = os.path.join(DATA_DIR, dir_)
    print(f"Processing class {dir_}...")
    
    for img_path in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, img_path))
        if img is None:
            continue
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            x_ = [lm.x for lm in hand_landmarks.landmark]
            y_ = [lm.y for lm in hand_landmarks.landmark]
            
            normalized = []
            for lm in hand_landmarks.landmark:
                normalized.extend([lm.x - min(x_), lm.y - min(y_)])
            
            if len(normalized) == 42:
                data.append(normalized)
                labels.append(int(dir_))  # Store as integer



# # Save dataset
# with open('sign_data.pickle', 'wb') as f:
#     pickle.dump({'data': data, 'labels': labels}, f)

# print("\nDataset created successfully!")