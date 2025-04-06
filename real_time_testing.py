import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from keras.models import Sequential # Add Sequential
from keras.layers import Dense, Dropout # Add Dense, Dropout
import pickle
import os
import h5py # Import h5py

print("--- Step 5: Real-Time Prediction (Load Weights Method) ---")

# --- Configuration ---
WEIGHTS_PATH = 'sign_language_weights.h5' # <== Path for weights file
LABEL_MAP_FILE = 'label_map.pkl'
NUM_LANDMARKS = 21
INPUT_SHAPE = (NUM_LANDMARKS * 2,) # Should be (42,)

# --- File Existence Checks ---
print(f"Current working directory: {os.getcwd()}")
print(f"Checking for weights file at: {os.path.abspath(WEIGHTS_PATH)}")
if not os.path.exists(WEIGHTS_PATH):
    print(f"FATAL ERROR: Weights file not found at '{WEIGHTS_PATH}'")
    print("Please ensure Step 3 was run correctly and saved the weights file.")
    exit()
print(f"Checking for label map file at: {os.path.abspath(LABEL_MAP_FILE)}")
if not os.path.exists(LABEL_MAP_FILE):
    print(f"FATAL ERROR: Label map file not found at '{LABEL_MAP_FILE}'")
    exit()
# --- End Check ---

# --- NEW: Inspect H5 File Structure ---
print(f"\nInspecting weights file: {WEIGHTS_PATH}")
try:
    with h5py.File(WEIGHTS_PATH, 'r') as f:
        print("Top-level groups (should be layer names):")
        layer_names_in_file = list(f.keys())
        print(layer_names_in_file)
        # Optional: Check for weights within a specific layer group
        if 'dense' in layer_names_in_file: # Check the first dense layer
             print("\nContents of group 'dense':")
             print(list(f['dense'].keys())) # Should show something like ['dense'] or layer name again
             if 'dense' in f['dense'].keys():
                  print("Weights/biases within 'dense/dense':")
                  print(list(f['dense']['dense'].keys())) # Should show ['bias:0', 'kernel:0'] or similar
        else:
             print("\nGroup 'dense' not found in the H5 file.")

except Exception as e:
    print(f"Error inspecting H5 file: {e}")
# --- END NEW SECTION ---

# --- Load Label Map ---
try:
    with open(LABEL_MAP_FILE, 'rb') as f:
        int_to_label = pickle.load(f)
    class_names = [int_to_label[i] for i in range(len(int_to_label))]
    NUM_CLASSES = len(class_names)
    print(f"Label map loaded. Classes ({NUM_CLASSES}): {class_names}")
except Exception as e:
     print(f"FATAL ERROR loading label map: {e}")
     exit()

# --- Build the EXACT SAME Model Architecture as in Step 3 ---
# --- Using default layer names ---
print("Building model architecture...")
model = Sequential([
    Dense(128, activation='relu', input_shape=INPUT_SHAPE), # Default name 'dense'
    Dropout(0.3),                                      # Default name 'dropout'
    Dense(64, activation='relu'),                     # Default name 'dense_1'
    Dropout(0.3),                                      # Default name 'dropout_1'
    Dense(32, activation='relu'),                     # Default name 'dense_2'
    Dense(NUM_CLASSES, activation='softmax')           # Default name 'dense_3'
])
print("\n--- Step 5 Model Summary (Check against Step 3) ---")
model.summary()

# --- Load ONLY the Weights ---
print(f"\nLoading model weights from {WEIGHTS_PATH}...")
try:
    # Use load_weights()
    model.load_weights(WEIGHTS_PATH)
    print("Model weights loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR loading weights: {e}")
    print("Double-check:")
    print("  1. Step 3 saved weights correctly AFTER successful training.")
    print("  2. The architecture defined here EXACTLY matches Step 3 (check summaries).")
    print(f"  3. The file '{WEIGHTS_PATH}' exists and is not empty/corrupt.")
    exit()

# --- MediaPipe Initialization ---
print("\nInitializing MediaPipe...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
print("MediaPipe initialized.")

# --- Webcam Setup & Main Loop ---
print("\nSetting up webcam...")
# ... (rest of your Step 5 code: cap = cv2.VideoCapture(0), main loop, etc.) ...
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
print("Webcam opened. Starting real-time prediction...")
print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        # print("Error: Failed to capture frame.") # Reduce spam
        continue # Skip frame if capture failed

    # --- Frame Processing ---
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    frame_rgb.flags.writeable = True

    prediction_text = "No Hand"
    confidence = 0.0

    # --- Landmark Extraction & Prediction ---
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        try:
            # --- Apply SAME Normalization ---
            all_x = [lm.x for lm in hand_landmarks.landmark]
            all_y = [lm.y for lm in hand_landmarks.landmark]
            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            relative_landmarks = []
            for i in range(NUM_LANDMARKS):
                relative_x = hand_landmarks.landmark[i].x - wrist_x
                relative_y = hand_landmarks.landmark[i].y - wrist_y
                relative_landmarks.extend([relative_x, relative_y])
            scale = np.sqrt((hand_landmarks.landmark[9].x - wrist_x)**2 + (hand_landmarks.landmark[9].y - wrist_y)**2)
            if scale < 1e-6:
                 max_dist = max(max(np.abs(all_x - wrist_x)), max(np.abs(all_y - wrist_y)))
                 scale = max_dist if max_dist > 1e-6 else 1.0
            normalized_landmarks = np.array(relative_landmarks) / scale
            feature_vector = normalized_landmarks.flatten()

            # --- Prediction ---
            input_data = np.expand_dims(feature_vector, axis=0)
            # Use the rebuilt model with loaded weights
            prediction_proba = model.predict(input_data, verbose=0)[0]
            predicted_class_index = np.argmax(prediction_proba)
            confidence = prediction_proba[predicted_class_index]

            # --- Get Label ---
            pred_threshold = 0.6 # Confidence threshold
            if confidence > pred_threshold:
                prediction_text = int_to_label.get(predicted_class_index, "Unknown")
            else:
                 prediction_text = "Uncertain"

            # --- Draw landmarks ---
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
        except Exception as e:
            # print(f"Error processing landmarks: {e}") # Reduce spam
            prediction_text = "ErrorProc" # Short error message
            confidence = 0.0
    else:
        # Keep prediction_text as "No Hand"
        pass

    # --- Display Prediction on Frame ---
    cv2.putText(frame, f"Pred: {prediction_text}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

    # --- Show Frame ---
    cv2.imshow('Sign Language Recognition', frame)

    # --- Exit Condition ---
    if cv2.waitKey(5) & 0xFF == ord('q'):
        print("\n'q' pressed. Quitting...")
        break

# --- Cleanup ---
print("Releasing webcam and destroying windows...")
cap.release()
cv2.destroyAllWindows()
if 'hands' in locals() and hands is not None: # Ensure 'hands' was initialized
    hands.close()
print("Cleanup complete.")