import os
import cv2
import sys
import time
import string
import numpy as np
import mediapipe as mp
import glob # Added for easier file deletion pattern matching

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

hands = mp_hands.Hands(static_image_mode=True,
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
    # Try to load a placeholder, create one if it fails
    frame_placeholder_path = 'black.jpg'
    if os.path.exists(frame_placeholder_path):
        frame = cv2.imread(frame_placeholder_path)
    else:
        frame = None

    if frame is None:
        print("Placeholder 'black.jpg' not found, creating a black image.")
        frame = np.zeros((480, 640, 3), dtype=np.uint8) # Default size

    cap = initialize_camera(frame)
    if cap is not None and not cap.isOpened():
        print("Error: Could not open selected camera."); cap = None
print("Camera Initialized Successfully.")


# --- Helper function for resetting class data ---
def reset_class_data(class_letter, class_dir):
    """Deletes all .jpg files in the specified class directory."""
    print(f"Resetting data for class {class_letter}...")
    files_to_delete = glob.glob(os.path.join(class_dir, '*.jpg'))
    if not files_to_delete:
        print("  No image files found to delete.")
        return 0 # Return number of files deleted

    deleted_count = 0
    for file_path in files_to_delete:
        try:
            os.remove(file_path)
            deleted_count += 1
        except OSError as e:
            print(f"  Error deleting file {file_path}: {e}")
    print(f"  Deleted {deleted_count} image(s).")
    return deleted_count

# --- Main Data Collection Loop ---
current_class_index = 0
while current_class_index < number_of_classes: # Changed to while loop for easier reset/restart
    class_letter = classes[current_class_index]
    class_dir = os.path.join(DATA_DIR, class_letter)
    if not os.path.exists(class_dir): os.makedirs(class_dir)
    print(f'\nCollecting data for class {class_letter} ({dataset_size} images total, {batch_size} per variation)')
    print("Press 'R' anytime during collection to reset this class.")

    # --- Prompt Phase ---
    print(f"Get ready to show the sign for '{class_letter}'. Press Space when ready.")
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from webcam during prompt.")
            time.sleep(0.5)
            continue

        frame_display = frame.copy()

        # --- Perform MediaPipe detection IN THE PROMPT LOOP ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        prompt_hand_fully_visible = False
        num_visible_prompt = 0

        # --- Check landmarks and visibility IN PROMPT ---
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            if len(hand_landmarks.landmark) == EXPECTED_NUM_LANDMARKS:
                all_visible_prompt = True
                visible_count_prompt = 0
                for lm in hand_landmarks.landmark:
                    if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
                        visible_count_prompt += 1
                    else:
                        all_visible_prompt = False
                if all_visible_prompt:
                    prompt_hand_fully_visible = True
                    num_visible_prompt = EXPECTED_NUM_LANDMARKS
                else:
                    num_visible_prompt = visible_count_prompt
            mp_drawing.draw_landmarks(
                frame_display, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

        # --- Display Prompt Text and Detailed Feedback ---
        prompt_text = f'Ready for {class_letter}? Position hand & Press "Space"!'
        cv2.putText(frame_display, prompt_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        if prompt_hand_fully_visible:
            feedback = f"Hand OK ({num_visible_prompt}/{EXPECTED_NUM_LANDMARKS}) - Press Space"
            feedback_color = (0, 255, 0)
        elif results.multi_hand_landmarks:
             feedback = f"Move Hand Fully Into Frame ({num_visible_prompt}/{EXPECTED_NUM_LANDMARKS})"
             feedback_color = (0, 0, 255)
        else:
            feedback = "Position Hand In Frame"
            feedback_color = (0, 165, 255)
        cv2.putText(frame_display, feedback, (10, frame_display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)

        # --- Show the frame ---
        cv2.imshow('Collect Image', frame_display)

        # --- Key Handling ---
        key = cv2.waitKey(25) & 0xFF
        if key == ord(' '):
             print(f"Starting capture for {class_letter}...")
             break # Exit prompt phase
        elif key == ord('r'): # Allow reset even in prompt phase
            print(f"\nReset request for class {class_letter} during prompt.")
            confirm_frame = frame_display.copy()
            cv2.putText(confirm_frame, f"RESET {class_letter}? Press 'Y' to confirm.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.imshow('Collect Image', confirm_frame)
            confirm_key = cv2.waitKey(0) & 0xFF
            if confirm_key == ord('y'):
                reset_class_data(class_letter, class_dir)
                print(f"Class {class_letter} reset. Please reposition and press Space.")
                # No need to change counters here as collection hasn't started
            else:
                print("Reset cancelled.")
            # Continue the prompt loop after confirmation/cancellation
        elif key == 27 or key == ord('q') or cv2.getWindowProperty('Collect Image', cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting during prompt..."); cap.release(); cv2.destroyAllWindows(); hands.close(); sys.exit()
    # --- End of Prompt Phase ---

    # --- Capture Phase with Batches, Auto-Pause/Resume, Visibility Check, and RESET ---
    counter = 0
    batch_counter = 0
    last_save_time = time.time() - save_interval
    is_paused_no_hand = False
    reset_requested_for_class = False # Flag to break outer loop if reset happens

    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None: print("Warning: Frame capture failed during collection."); continue

        frame_to_show = frame.copy()

        # --- MediaPipe Hand Detection ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = hands.process(frame_rgb)
        frame_rgb.flags.writeable = True

        # --- Check for Full Hand Landmarks & Visibility Proxy ---
        hand_detected_fully_visible = False
        valid_hand_landmarks = None
        num_visible_landmarks = 0

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            if len(hand_landmarks.landmark) == EXPECTED_NUM_LANDMARKS:
                all_landmarks_visible = True
                visible_count_this_frame = 0
                for lm in hand_landmarks.landmark:
                    if 0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0:
                        visible_count_this_frame += 1
                    else:
                        all_landmarks_visible = False
                if all_landmarks_visible:
                    hand_detected_fully_visible = True
                    valid_hand_landmarks = hand_landmarks
                    num_visible_landmarks = EXPECTED_NUM_LANDMARKS
                else:
                    num_visible_landmarks = visible_count_this_frame
        # --- Manage Auto-Pause State ---
        if not hand_detected_fully_visible and not is_paused_no_hand:
            is_paused_no_hand = True
            print("  -- Paused: Hand not fully visible (< 21 landmarks in view). Show hand clearly. --")
        elif hand_detected_fully_visible and is_paused_no_hand:
            is_paused_no_hand = False
            last_save_time = time.time() # Reset save timer
            print("  -- Resumed: Hand fully visible (21 landmarks in view). --")

        # --- Draw Landmarks if Detected Fully Visible ---
        if valid_hand_landmarks:
            mp_drawing.draw_landmarks(frame_to_show, valid_hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                      mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2))

        # --- Conditional Image Saving ---
        current_time = time.time()
        if hand_detected_fully_visible and not is_paused_no_hand and (current_time - last_save_time >= save_interval):
            img_path = os.path.join(DATA_DIR, class_letter, f'{counter}.jpg')
            cv2.imwrite(img_path, frame)
            counter += 1
            last_save_time = current_time

            # --- Check if a Manual Batch Pause is Needed ---
            if counter < dataset_size and counter > 0 and counter % batch_size == 0:
                batch_counter += 1
                print(f"  -- Manual Pause: Batch {batch_counter} complete ({counter}/{dataset_size}). --")
                prompt_message = f"Adjust Angle/Pos for Batch {batch_counter + 1} & Press Space"

                # --- Change hand on batch 3 ---
                upcoming_batch_number = batch_counter + 1
                if batch_counter == 2: # If Batch 2 just finished, prompt for switch before Batch 3
                    prompt_message = f"SWITCH HAND for Batch {upcoming_batch_number}! Press Space"
                    prompt_color = (0, 255, 255) # Yellow/Cyan to make it stand out
                    print("  ** Please switch to your other hand for the next batch. **") # Console prompt
                else:
                    prompt_message = f"Adjust Angle/Pos for Batch {upcoming_batch_number} & Press Space"
                    prompt_color = (255, 0, 255) # Default magenta

                # --- PAUSE LOOP ---
                while True:
                    ret_pause, frame_pause_orig = cap.read() # Read original frame
                    if not ret_pause or frame_pause_orig is None:
                        print("Warning: Frame capture failed during pause.")
                        time.sleep(0.1)
                        continue

                    frame_pause = frame_pause_orig.copy() # Work with a copy
                    cv2.putText(frame_pause, prompt_message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                    cv2.putText(frame_pause, "Press 'R' to reset this class", (10, frame_pause.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 1)
                    cv2.imshow('Collect Image', frame_pause)
                    key_pause = cv2.waitKey(10) & 0xFF

                    if key_pause == ord(' '):
                        print(f"  -- Resuming capture for Batch {batch_counter + 1}... --")
                        last_save_time = time.time(); is_paused_no_hand = False; break # Break PAUSE loop
                    elif key_pause == ord('r'):
                        print(f"\nReset request for class {class_letter} during manual pause.")
                        confirm_frame_pause = frame_pause.copy() # Use the already annotated pause frame
                        cv2.putText(confirm_frame_pause, f"RESET {class_letter}? Press 'Y' to confirm.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                        cv2.imshow('Collect Image', confirm_frame_pause)
                        confirm_key_pause = cv2.waitKey(0) & 0xFF # Wait indefinitely for Y/N
                        if confirm_key_pause == ord('y'):
                            reset_class_data(class_letter, class_dir)
                            # Reset counters and flag to restart the outer class loop
                            counter = 0
                            batch_counter = 0
                            last_save_time = time.time() - save_interval
                            is_paused_no_hand = False
                            reset_requested_for_class = True # Signal outer loop to restart
                            print(f"Class {class_letter} reset. Restarting collection...")
                            break # Break PAUSE loop
                        else:
                            print("Reset cancelled.")
                            # Continue the PAUSE loop (redisplay normal pause prompt)
                    elif key_pause == 27 or key_pause == ord('q') or cv2.getWindowProperty('Collect Image', cv2.WND_PROP_VISIBLE) < 1:
                         print("Exiting during pause..."); cap.release(); cv2.destroyAllWindows(); hands.close(); sys.exit()
                # --- End Manual Pause Loop ---
                if reset_requested_for_class:
                    break # Break the main capture loop (while counter < dataset_size) to restart class

        # --- Check for Reset Request DURING active capture ---
        key = cv2.waitKey(5) & 0xFF # Use the same key timeout as before

        if key == ord('r'):
            print(f"\nReset request for class {class_letter} during capture.")
            # Pause auto-saving briefly while confirming
            is_paused_confirm = is_paused_no_hand # Store current pause state
            is_paused_no_hand = True # Temporarily pause saving

            confirm_frame = frame_to_show.copy() # Use the current annotated frame
            cv2.rectangle(confirm_frame, (40, 80), (confirm_frame.shape[1]-40, 140), (0,0,0), -1) # Black background for text
            cv2.putText(confirm_frame, f"RESET {class_letter}? Press 'Y' to confirm.", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            cv2.putText(confirm_frame, "Any other key to cancel.", (50, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow('Collect Image', confirm_frame)

            confirm_key = cv2.waitKey(0) & 0xFF # Wait indefinitely for Y/N
            if confirm_key == ord('y'):
                reset_class_data(class_letter, class_dir)
                # Reset counters and flag to restart the outer class loop
                counter = 0
                batch_counter = 0
                last_save_time = time.time() - save_interval
                is_paused_no_hand = False # Fully unpause after reset
                reset_requested_for_class = True # Signal outer loop to restart
                print(f"Class {class_letter} reset. Restarting collection...")
                break # Break the main capture loop (while counter < dataset_size)
            else:
                print("Reset cancelled.")
                is_paused_no_hand = is_paused_confirm # Restore original pause state
                # Continue the main capture loop

        # --- Display Feedback on Screen ---
        if is_paused_no_hand:
            feedback_text = f"PAUSED - Show Hand Fully ({num_visible_landmarks}/{EXPECTED_NUM_LANDMARKS})"
            feedback_color = (0, 0, 255)
        elif hand_detected_fully_visible:
            feedback_text = f"Capturing ({EXPECTED_NUM_LANDMARKS}/{EXPECTED_NUM_LANDMARKS} Visible)"
            feedback_color = (0, 255, 0)
        else:
             feedback_text = f"Position Hand ({num_visible_landmarks}/{EXPECTED_NUM_LANDMARKS} Visible)"
             feedback_color = (0, 165, 255)

        cv2.putText(frame_to_show, feedback_text, (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, feedback_color, 2)
        cv2.putText(frame_to_show, f"Saved: {counter}/{dataset_size}", (frame.shape[1] - 200, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame_to_show, "Press 'R' to Reset Class", (10, frame_to_show.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1) # Reminder

        cv2.imshow('Collect Image', frame_to_show)

        # --- Exit Check ---
        # (Combined with 'R' key check above)
        if key == 27 or key == ord('q') or cv2.getWindowProperty('Collect Image', cv2.WND_PROP_VISIBLE) < 1:
            print("Exiting..."); cap.release(); cv2.destroyAllWindows(); hands.close(); sys.exit()
    # --- End Capture Loop for the class ---

    if reset_requested_for_class:
        # If reset was requested, loop back to the *same* class index
        # The 'continue' keyword jumps to the next iteration of the outer while loop
        continue
    else:
        # Normal completion for the class
        print(f"Finished collecting for class {class_letter}.")
        # --- Brief "Done" display ---
        start_time_done = time.time()
        while time.time() - start_time_done < 1.5:
            ret_done, frame_done = cap.read()
            if not ret_done or frame_done is None: break # Handle potential read error
            # Ensure frame_done is writable if needed, though text overlay usually works
            if not frame_done.flags.writeable:
                frame_done = frame_done.copy()
            cv2.putText(frame_done, 'Done... Next!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Collect Image', frame_done)
            key_done = cv2.waitKey(1) & 0xFF
            if key_done == 27 or key_done == ord('q'):
                 print("Exiting during 'Done' display..."); cap.release(); cv2.destroyAllWindows(); hands.close(); sys.exit()
        # Move to the next class
        current_class_index += 1

# --- Final Message & Cleanup ---
print("\nData collection finished!")
# Add a final check if cap is still valid before reading
if cap.isOpened():
    ret, frame = cap.read()
    if ret and frame is not None:
        # Ensure frame is writable
        if not frame.flags.writeable:
            frame = frame.copy()
        cv2.putText(frame, 'Thank you!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Collect Image', frame)
        cv2.waitKey(2000)

print("Releasing resources...")
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
if 'hands' in locals() and hands: # Check if hands was initialized
    hands.close()
print("Done.")