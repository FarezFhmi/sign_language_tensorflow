import os
import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

# Define dataset directory
DATA_DIR = './data'
OUTPUT_FILE = 'dataset.npz'  # Save as NumPy file

data = []
labels = []

# ✅ Auto-detect class labels (folder names in `data/`)
class_labels = sorted([folder for folder in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, folder))])

# Extract hand landmarks and crop images
for label in class_labels:
    class_dir = os.path.join(DATA_DIR, label)
    
    print(f'Processing class: {label}')
    
    for img_name in os.listdir(class_dir)[:1]:
        img_path = os.path.join(class_dir, img_name)
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H, W, _ = img.shape  # Get image dimensions

        results = hands.process(img_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                data_aux = []
                x_ = []
                y_ = []

                # Collect all x, y positions
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                # Calculate bounding box coordinates with small padding
                padding = 23  # Small padding to ensure the entire hand is captured
                x1 = int(min(x_) * W) - padding  # Left boundary
                y1 = int(min(y_) * H) - padding  # Top boundary
                x2 = int(max(x_) * W) + padding  # Right boundary
                y2 = int(max(y_) * H) + padding  # Bottom boundary

                # Ensure the bounding box is within the image dimensions
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(W, x2)
                y2 = min(H, y2)

                # Crop the image to the bounding box
                cropped_img = img_rgb[y1:y2, x1:x2]

                # Resize the cropped image to a fixed size (e.g., 64x64)
                resized_img = cv2.resize(cropped_img, (64, 64))

                # # Display the original image with landmarks and bounding box
                # plt.figure(figsize=(10, 5))
                # plt.subplot(1, 2, 1)
                # plt.imshow(img_rgb)
                # plt.title(f'Original Image: {label}')
                # plt.axis('off')

                # # Display the cropped and resized image
                # plt.subplot(1, 2, 2)
                # plt.imshow(resized_img)
                # plt.title(f'Cropped and Resized Image: {label}')
                # plt.axis('off')

                # plt.show()

                # Normalize the image to [0, 1] range
                normalized_img = resized_img / 255.0

                # Append to dataset
                data.append(normalized_img)
                labels.append(label)

# Convert to NumPy arrays
X = np.array(data, dtype=np.float32)
y = np.array(labels)

# ✅ Save as `.npz` for TensorFlow compatibility
np.savez_compressed(OUTPUT_FILE, data=X, labels=y)
print(f"Dataset saved as {OUTPUT_FILE} ✅")