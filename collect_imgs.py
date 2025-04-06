import os
import cv2
import sys
import time
import string
import numpy as np
import mediapipe as mp

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Exclude J and Z from the classes
all_letters = list(string.ascii_uppercase)
excluded_letters = {'J', 'Z'}
classes = [letter for letter in all_letters if letter not in excluded_letters]
number_of_classes = len(classes)
dataset_size = 200   # Total images per class
batch_size = 50      # Number of images per variation/batch
save_interval = 0.1  # Min seconds between saves

# --- MediaPipe Initialization ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
EXPECTED_NUM_LANDMARKS = len(mp_hands.HandLandmark) # Should be 21

hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.6,
                       min_tracking_confidence=0.6)

# Function to initialize the camera (remains the same)
def initialize_camera(frame):
    cv2.putText(frame, "Select Camera:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "1 - Default Webcam", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "2 - Phone Camera", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Q' to Exit", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Collect Image', frame)
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'): return cv2.VideoCapture(0)
        elif key == ord('2'): return cv2.VideoCapture(1)
        elif key == ord('q'): cv2.destroyAllWindows(); sys.exit()

# Initialize the camera
cap = None
while cap is None:
    frame = cv2.imread('black.jpg')
    if frame is None: frame = np.zeros((300, 500, 3), dtype=np.uint8)
    cap = initialize_camera(frame)
    if cap is not None and not cap.isOpened():
        print("Error: Could not open selected camera."); cap = None
print("Camera Initialized Successfully.")

# --- Main Data Collection Loop ---
for j in range(number_of_classes):
    class_letter = classes[j]
    class_dir = os.path.join(DATA_DIR, class_letter)
    if not os.path.exists(class_dir): os.makedirs(class_dir)
    print(f'\nCollecting data for class {class_letter} ({dataset_size} images total, {batch_size} per variation)')

    # --- Prompt Phase ---
    while True:
        ret, frame = cap.read()
        if not ret or frame is None: print("Error: Could not read frame from webcam."); time.sleep(0.5); continue
        cv2.putText(frame, f'Ready for {class_letter}? Position hand & Press "Space"!', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Collect Image', frame)
        key = cv2.waitKey(25)
        if key == ord(' '): print(f"Starting capture for {class_letter}..."); break
        if key == 27 or cv2.getWindowProperty('Collect Image', cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting..."); cap.release(); cv2.destroyAllWindows(); hands.close(); sys.exit()

    # --- Capture Phase with Batches, Auto-Pause/Resume, and Landmark VISIBILITY Check ---
    counter = 0
    batch_counter = 0
    last_save_time = time.time() - save_interval
    is_paused_no_hand = False

    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None: print("Warning: Frame capture failed during collection."); continue

        frame_to_show = frame.copy()

        # --- MediaPipe Hand Detection ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # --- Check for Full Hand Landmarks & Visibility Proxy --- <<< MODIFIED SECTION
        hand_detected_fully_visible = False # New flag name for clarity
        valid_hand_landmarks = None
        num_visible_landmarks = 0 # Track visible landmarks

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            # Check total count first
            if len(hand_landmarks.landmark) == EXPECTED_NUM_LANDMARKS:
                # Now, iterate and check coordinate validity as visibility proxy
                all_landmarks_visible = True # Assume true initially
                visible_count_this_frame = 0
                for lm in hand_landmarks.landmark:
                    # Check if x and y are within the normalized image bounds [0, 1]
                    if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
                        visible_count_this_frame += 1
                    else:
                        all_landmarks_visible = False
                        # Optional: break here if strict all-visible check is desired
                        # break

                # We require ALL landmarks to be within bounds for 'fully visible'
                if all_landmarks_visible:
                    hand_detected_fully_visible = True
                    valid_hand_landmarks = hand_landmarks # Store landmarks for drawing
                    num_visible_landmarks = EXPECTED_NUM_LANDMARKS
                else:
                    # Hand detected, count is 21, but some are out of bounds
                    num_visible_landmarks = visible_count_this_frame
                    # hand_detected_fully_visible remains False
            # else: Total landmark count mismatch (< 21), treat as not fully visible
        # else: No hand detected at all

        # --- Manage Auto-Pause State (Based on full visibility) ---
        if not hand_detected_fully_visible and not is_paused_no_hand:
            is_paused_no_hand = True
            print("  -- Paused: Hand not fully visible (< 21 landmarks in view). Show hand clearly. --")
        elif hand_detected_fully_visible and is_paused_no_hand:
            is_paused_no_hand = False
            last_save_time = time.time() # Reset save timer
            print("  -- Resumed: Hand fully visible (21 landmarks in view). --")

        # --- Draw Landmarks if Detected Fully Visible ---
        if valid_hand_landmarks: # Draw only if we stored valid landmarks
            mp_drawing.draw_landmarks(frame_to_show, valid_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

        # --- Conditional Image Saving (Based on full visibility) ---
        current_time = time.time()
        if hand_detected_fully_visible and not is_paused_no_hand and (current_time - last_save_time >= save_interval):
            img_path = os.path.join(DATA_DIR, class_letter, f'{counter}.jpg')
            cv2.imwrite(img_path, frame) # Save original frame
            counter += 1
            last_save_time = current_time

            # --- Check if a Manual Batch Pause is Needed ---
            if counter < dataset_size and counter % batch_size == 0:
                batch_counter += 1
                print(f"  -- Manual Pause: Batch {batch_counter} complete ({counter}/{dataset_size}). --")
                prompt_message = f"Adjust Angle/Pos for Batch {batch_counter + 1} & Press Space"
                # --- PAUSE LOOP ---
                while True:
                    ret_pause, frame_pause = cap.read()
                    if not ret_pause or frame_pause is None: print("Warning: Frame capture failed during pause."); time.sleep(0.1); continue
                    cv2.putText(frame_pause, prompt_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.imshow('Collect Image', frame_pause)
                    key_pause = cv2.waitKey(10)
                    if key_pause == ord(' '):
                        print(f"  -- Resuming capture for Batch {batch_counter + 1}... --")
                        last_save_time = time.time(); is_paused_no_hand = False; break
                    if key_pause == 27 or cv2.getWindowProperty('Collect Image', cv2.WND_PROP_VISIBLE) < 1:
                         print("Exiting during pause..."); cap.release(); cv2.destroyAllWindows(); hands.close(); sys.exit()
                # --- End Manual Pause Loop ---
        # --- End Conditional Saving & Pause Logic ---

        # --- Display Feedback on Screen (Based on full visibility) ---
        if is_paused_no_hand:
            feedback_text = f"PAUSED - Show Hand Fully ({num_visible_landmarks}/{EXPECTED_NUM_LANDMARKS})" # Show count even when paused
            feedback_color = (0, 0, 255) # Red
        elif hand_detected_fully_visible:
            feedback_text = f"Capturing ({EXPECTED_NUM_LANDMARKS}/{EXPECTED_NUM_LANDMARKS} Visible)"
            feedback_color = (0, 255, 0) # Green
        else: # Hand detected partially or not at all
             feedback_text = f"Position Hand ({num_visible_landmarks}/{EXPECTED_NUM_LANDMARKS} Visible)"
             feedback_color = (0, 165, 255) # Orange/Yellow

        cv2.putText(frame_to_show, feedback_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
        cv2.putText(frame_to_show, f"Saved: {counter}/{dataset_size}", (frame.shape[1] - 200, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow('Collect Image', frame_to_show)

        # --- Exit Check ---
        key = cv2.waitKey(5)
        if key == 27 or cv2.getWindowProperty('Collect Image', cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting..."); cap.release(); cv2.destroyAllWindows(); hands.close(); sys.exit()
    # --- End Capture Loop for the class ---

    print(f"Finished collecting for class {class_letter}.")
    # --- Brief "Done" display ---
    start_time_done = time.time()
    while time.time() - start_time_done < 1.5:
        ret, frame = cap.read()
        if not ret: break
        cv2.putText(frame, 'Done... Next!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Collect Image', frame)
        if cv2.waitKey(1) == 27: sys.exit()

# --- Final Message & Cleanup ---
# ... (Your existing final message and cleanup code) ...
print("\nData collection finished!")
ret, frame = cap.read()
if ret and frame is not None:
    cv2.putText(frame, 'Thank you!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Collect Image', frame)
    cv2.waitKey(2000)
print("Releasing resources..."); cap.release(); cv2.destroyAllWindows(); hands.close(); print("Done.")