import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
# Ensure Keras components are imported correctly if needed, depending on your TF version
# from tensorflow import keras  # Or keep as is if using standalone Keras
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Dense, Dropout
import pickle
import os
# import h5py # This was imported but not used, can be removed if truly unused
import time
import sys # For exit
import google.generativeai as genai
import threading # For non-blocking Gemini calls
import queue     # For thread-safe communication

# --- Load Environment Variables ---
from dotenv import load_dotenv
load_dotenv()

# --- Configuration ---
FULL_MODEL_PATH = 'sign_language_model_new.h5'
LABEL_MAP_FILE = 'label_map_new.pkl'
NUM_LANDMARKS = 21
CAMERA_OPTION = 0 # Default camera index
INPUT_SHAPE = (NUM_LANDMARKS * 2,)
HOLD_DURATION = 1.0 # Reduced hold time based on previous code
NO_HAND_TIMEOUT = 3.0 # Reduced timeout based on previous code
PREDICTION_THRESHOLD = 0.7 # Confidence threshold for a prediction
STABLE_CONFIRM_THRESHOLD = 0.8 # Confidence needed DURING hold

# --- Gemini Configuration ---
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        print("---------------------")
        print("WARNING: GEMINI_API_KEY not found in environment variables.")
        print("Gemini features will be disabled.")
        print("Create a .env file with GEMINI_API_KEY=YOUR_KEY to enable them.")
        print("---------------------")
        GEMINI_ENABLED = False
        gemini_model = None 
    else:
        genai.configure(api_key=GEMINI_API_KEY)
        # Use a model suitable for your task, flash is often faster
        gemini_model = genai.GenerativeModel('gemini-1.5-flash')
        print("Gemini configured successfully using API key from environment.")
        GEMINI_ENABLED = True

except Exception as e:
    print(f"WARNING: Could not configure Gemini. Interpretation disabled. Error: {e}")
    GEMINI_ENABLED = False
    gemini_model = None
# --- End Gemini Config ---

# --- File Existence Checks ---
print(f"Current working directory: {os.getcwd()}")
# --- Check for FULL MODEL file ---
print(f"Checking for model file at: {os.path.abspath(FULL_MODEL_PATH)}")
if not os.path.exists(FULL_MODEL_PATH):
    print(f"FATAL ERROR: Model file (.h5) not found at '{FULL_MODEL_PATH}'")
    sys.exit()
# --- Check for Label Map file ---
print(f"Checking for label map file at: {os.path.abspath(LABEL_MAP_FILE)}")
if not os.path.exists(LABEL_MAP_FILE):
    print(f"FATAL ERROR: Label map file not found at '{LABEL_MAP_FILE}'")
    sys.exit()

# --- Load Label Map ---
try:
    with open(LABEL_MAP_FILE, 'rb') as f:
        int_to_label = pickle.load(f)
    class_names = [int_to_label[i] for i in range(len(int_to_label))]
    NUM_CLASSES = len(class_names) # Still useful info
    print(f"Label map loaded. Classes ({NUM_CLASSES}): {class_names}")
except Exception as e:
     print(f"FATAL ERROR loading label map: {e}")
     sys.exit()

# --- Load Full Model (Architecture + Weights) from H5 file ---
print(f"\nLoading full model from {FULL_MODEL_PATH}...")
try:
    # Load the .h5 file, compile=False is still recommended
    model = tf.keras.models.load_model(FULL_MODEL_PATH, compile=False)
    print("Full model loaded successfully from .h5 file.")
except Exception as e:
    print(f"FATAL ERROR loading model (.h5): {e}")
    sys.exit()

# --- MediaPipe Initialization ---
print("\nInitializing MediaPipe...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.6)
print("MediaPipe initialized.")

# --- Webcam Setup ---
print("\nSetting up webcam...")
cap = cv2.VideoCapture(CAMERA_OPTION)
if not cap.isOpened():
    print(f"Error: Could not open webcam (Index: {CAMERA_OPTION}). Trying index 1...")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open webcam on index 0 or 1. Please check connection.")
        sys.exit()
print("Webcam opened successfully.")

# --- Language Selection Function ---
def select_language(capture_device):
    """Displays language options on the camera feed and waits for user selection."""
    selection_window_name = 'Language Selection'
    print("\nPlease look at the 'Language Selection' window.")
    while True:
        ret, frame = capture_device.read()
        if not ret:
            print("Error: Cannot grab frame for language selection.")
            time.sleep(0.5)
            continue

        frame = cv2.flip(frame, 1)

        # --- Draw Selection Options ---
        h, w, _ = frame.shape
        y_pos = 70.0
        dy = 50.0
        font_scale_title = min(w, h) / 700
        font_scale_option = min(w, h) / 900

        cv2.putText(frame, "Select Interpretation Language:", (50, int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (0, 255, 0), 2)
        y_pos += dy * 1.5
        cv2.putText(frame, "1 - English (Predict Word)", (50, int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_option, (255, 255, 255), 2)
        y_pos += dy
        cv2.putText(frame, "2 - Malay (Predict Word)", (50, int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_option, (255, 255, 255), 2)
        y_pos += dy * 1.5
        cv2.putText(frame, "Press 'Q' to Exit", (50, int(y_pos)), cv2.FONT_HERSHEY_SIMPLEX, font_scale_option * 0.9, (200, 200, 200), 2)

        cv2.imshow(selection_window_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):
            print("Selected: English Mode (Word Prediction)")
            cv2.destroyWindow(selection_window_name)
            return 'EN'
        elif key == ord('2'):
            print("Selected: Malay Mode (Word Prediction)")
            cv2.destroyWindow(selection_window_name)
            return 'MY'
        elif key == ord('q'):
            print("Exiting during language selection...")
            cv2.destroyWindow(selection_window_name)
            capture_device.release()
            sys.exit()

# --- Call Language Selection ---
print("Proceeding to language selection...")
current_language = select_language(cap)
print(f"Interpretation language set to: {current_language}")

# --- Post-Selection Setup ---
print("\nStarting real-time prediction...")
print(f"Hold a sign steady for {HOLD_DURATION:.1f}s to add it.")
print(f"Remove hand for {NO_HAND_TIMEOUT:.1f}s to interpret the sequence.")
print("Press 'q' to quit during recognition.")
print("Press 'c' to clear sequence manually.")

# --- State Variables ---
sentence_sequence = []
current_stable_prediction = None
stable_prediction_start_time = None
last_hand_detection_time = time.time()
interpreted_sentence = ""
status_message = ""
# current_language is set by select_language()

# --- Threading Setup ---
gemini_result_queue = queue.Queue()
interpretation_active = False # Define flag in global scope

# --- Function for Gemini Interpretation (Handles EN/MY prediction) ---
def interpret_with_gemini(sign_list, language='EN'):

    if not sign_list:
        # This case should ideally not be reached if called correctly
        print("[Warning interpret_with_gemini] Called with empty sign list.")
        return "[Internal Error: Empty List]"

    base_word = "".join(sign_list)

    if not GEMINI_ENABLED:
        return base_word + " [Gemini N/A]"

    prompt = ""
    task_description = ""

    # --- Select Prompt based on Language ---
    if language == 'MY':
        task_description = "Malay Completion Task"
        prompt = f"""Analyze the following sequence of letters detected from sign language, assuming they might form the beginning of a **Malay** word: {', '.join(sign_list)}.
Your task is to: 1. Concatenate the letters (base sequence: {base_word}). 2. Determine if '{base_word}' is likely an incomplete prefix of a common **Malay** word. 3. If YES and prediction is confident, output the single completed **Malay** word (e.g., ['L','A','P','A'] -> LAPAR). 4. If NO (complete, not Malay prefix, ambiguous), output the original base sequence '{base_word}' (e.g., ['S','A','Y','A'] -> SAYA, ['X','Y','Z'] -> XYZ).
**Important Rules:** Context is **Malay**. Output **ONLY the final resulting word**. No extra text.
Sequence: {', '.join(sign_list)}
Resulting Word:"""

    elif language == 'EN':
        task_description = "English Completion Task"
        prompt = f"""Analyze the following sequence of letters detected from sign language, assuming they might form the beginning of a common **English** word: {', '.join(sign_list)}.
Your task is to: 1. Concatenate the letters (base sequence: {base_word}). 2. Determine if '{base_word}' is likely an incomplete prefix of a common **English** word. 3. If YES and prediction is confident, output the single completed **English** word (e.g., ['L','O','A'] -> LOAN). 4. If NO (complete, not English prefix, ambiguous), output the original base sequence '{base_word}' (e.g., ['C','A','T'] -> CAT, ['X','Y','Z'] -> XYZ).
**Important Rules:** Context is **English**. Output **ONLY the final resulting word**. No extra text.
Sequence: {', '.join(sign_list)}
Resulting Word:"""
    else: # Fallback
         print(f"[Warning interpret_with_gemini] Unknown language '{language}'. Defaulting to English completion.")
         language = 'EN' # Correct the language variable for the rest of the function
         task_description = "English Completion Task (Fallback)"
         # Re-generate English prompt (important if language was truly unknown)
         prompt = f"""Analyze the following sequence of letters detected from sign language, assuming they might form the beginning of a common **English** word: {', '.join(sign_list)}.
Your task is to: 1. Concatenate the letters (base sequence: {base_word}). 2. Determine if '{base_word}' is likely an incomplete prefix of a common **English** word. 3. If YES and prediction is confident, output the single completed **English** word (e.g., ['L','O','A'] -> LOAN). 4. If NO (complete, not English prefix, ambiguous), output the original base sequence '{base_word}' (e.g., ['C','A','T'] -> CAT, ['X','Y','Z'] -> XYZ).
**Important Rules:** Context is **English**. Output **ONLY the final resulting word**. No extra text.
Sequence: {', '.join(sign_list)}
Resulting Word:"""

    try:
        # Added timeout to prevent thread hanging indefinitely
        # Increase timeout if needed, especially for complex prompts or slow networks
        response = gemini_model.generate_content(prompt, request_options={'timeout': 15})

        if response.parts:
            result = response.text.strip()
            # Aggressive cleaning
            prefixes_to_remove = [
                "Resulting Word:", "Resulting Sequence:", "The completed word is",
                "The most likely word is", "Output:", "Combined Word:", ":"
            ]
            result_lower = result.lower()
            for prefix in prefixes_to_remove:
                 if result_lower.startswith(prefix.lower()):
                     result = result[len(prefix):].strip()
                     result_lower = result.lower()

            result = result.strip('"`')

            # Basic Validation (check if starts with base, ignore if result is empty)
            if not result.upper().startswith(base_word.upper()) and len(result) > 0:
                print(f"[Thread Warning] Gemini {language} result '{result}' doesn't start with base '{base_word}'. Falling back.")
                return base_word + " [Suggest Err]"
            elif len(result) == 0:
                 print(f"[Thread Warning] Gemini {language} returned empty result for '{base_word}'. Falling back.")
                 return base_word + " [Empty Resp]"
            else:
                 print(f"Predict: '{result}' from base '{base_word}'")
                 return result.upper() # Return consistent casing

        elif response.prompt_feedback and response.prompt_feedback.block_reason:
             print(f"[Thread Warning] Gemini blocked: {response.prompt_feedback.block_reason}")
             # Log the sign list that caused the block for debugging
             print(f"[Thread Debug] Blocked sequence: {sign_list}")
             return base_word + " [Blocked]"
        else:
             # This case might be hit if response exists but has no 'parts' and no block reason
             print("[Thread Warning] Gemini response has no parts or block reason.")
             return base_word + " [Malformed Resp]"

    except Exception as e:
        print(f"[Thread Error] Error during Gemini API call: {e}")
        # Check for specific common errors
        error_str = str(e).lower()
        if "timeout" in error_str or "deadline exceeded" in error_str:
            return base_word + " [Timeout]"
        elif "api key not valid" in error_str:
             print("[Thread Critical Error] Gemini API Key is invalid!")
             return base_word + " [API Key Err]"
        elif "quota" in error_str:
             print("[Thread Warning] Gemini API quota exceeded.")
             return base_word + " [Quota Err]"
        # General API error
        return base_word + " [API Error]"
# --- End of interpret_with_gemini ---

# --- Worker Function for Thread ---
def call_gemini_in_thread(sign_list_copy, language_copy):
    try:
        # interpret_with_gemini handles its own API errors and returns a string
        result = interpret_with_gemini(sign_list_copy, language_copy)
        gemini_result_queue.put(result) # Put result in the queue
    except Exception as e:
        # This catches errors within this worker function itself (should be rare)
        print(f"FATAL ERROR in Gemini thread function structure: {e}")
        # Put an error message in the queue so the main thread knows
        gemini_result_queue.put(f"[ThreadFail: {e}]")

# --- Main Loop ---
main_window_name = f'Sign Language Recognition ({current_language} Mode)'
while cap.isOpened():
    # --- Check Queue for Results from Previous Interpretation ---
    try:
        # Check if interpretation is active before trying to get from queue
        # This prevents overwriting a valid result if queue check runs faster than expected
        if interpretation_active:
            new_interpretation = gemini_result_queue.get_nowait() # Non-blocking check
            interpreted_sentence = new_interpretation # Update display variable
            interpretation_active = False # Mark interpretation as finished HERE
            if not any(tag in interpreted_sentence for tag in ["[", "]"]):
                 status_message = "Prediction Complete" # Clear "Interpreting..." status on success
            else:
                 status_message = "Prediction Done (w/ Issues)" # Indicate non-perfect result
            print(f"Full sentences: {interpreted_sentence}")
    except queue.Empty:
        # No result waiting in the queue, continue normally
        pass
    except Exception as e:
        print(f"Error getting from queue: {e}")
        interpretation_active = False # Reset flag on error getting from queue


    # --- Grab Frame ---
    ret, frame = cap.read()
    if not ret:
        print("Warning: Failed to grab frame.")
        time.sleep(0.1)
        continue

    # --- Frame Processing & Hand Detection ---
    frame = cv2.flip(frame, 1)
    # Convert color ONCE before processing
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Optimize: Pass non-writable buffer to MediaPipe
    frame_rgb.flags.writeable = False
    results = hands.process(frame_rgb)
    # Make frame writable again for drawing ONLY if needed (usually drawing works fine)
    # frame.flags.writeable = True # Usually not needed for cv2.putText/draw_landmarks

    current_prediction_text = "No Hand"
    confidence = 0.0
    trigger_interpretation = False # Reset flag each loop

    # --- Reset Status Message (If not interpreting or showing transient message) ---
    if not interpretation_active and not any(status_message.startswith(prefix) for prefix in ["Added", "Cleared", "Prediction", "Holding"]):
        status_message = ""
    # Clear transient messages slightly differently - allow them to persist a bit longer maybe?
    # Or just rely on the above condition clearing them when interpretation isn't active.
    # Let's simplify: if not interpreting and message is transient, clear it.
    elif not interpretation_active and status_message.startswith(("Added:", "Cleared", "Prediction Done")):
        status_message = ""


    # --- Process Hand Landmarks ---
    if results.multi_hand_landmarks:
        last_hand_detection_time = time.time()
        hand_landmarks = results.multi_hand_landmarks[0]
        try:
            # --- Normalization & Feature Extraction ---
            # Ensure landmarks list is not empty before accessing index 0
            if not hand_landmarks.landmark:
                raise ValueError("Landmark list is empty")

            all_x = [lm.x for lm in hand_landmarks.landmark]
            all_y = [lm.y for lm in hand_landmarks.landmark]

            wrist_x = hand_landmarks.landmark[0].x
            wrist_y = hand_landmarks.landmark[0].y
            relative_landmarks = []

            # Check if number of landmarks is as expected
            num_detected_landmarks = len(hand_landmarks.landmark)
            if num_detected_landmarks != NUM_LANDMARKS:
                 print(f"Warning: Detected {num_detected_landmarks} landmarks, expected {NUM_LANDMARKS}.")
                 # Option: Skip frame if wrong number detected? Or try to proceed?
                 # Let's try to proceed using min(detected, expected)
                 num_to_process = min(num_detected_landmarks, NUM_LANDMARKS)
                 if num_to_process == 0: raise ValueError("No landmarks to process.")
            else:
                 num_to_process = NUM_LANDMARKS

            for i in range(num_to_process):
                relative_x = hand_landmarks.landmark[i].x - wrist_x
                relative_y = hand_landmarks.landmark[i].y - wrist_y
                relative_landmarks.extend([relative_x, relative_y])

            # If we processed fewer landmarks, pad the vector? Or adjust INPUT_SHAPE?
            # For now, this might lead to ShapeErr later if num_to_process < NUM_LANDMARKS
            # A more robust solution would involve handling missing landmarks better.

            # Robust Scaling
            if len(all_x) > 1:
                 max_abs_x = max(np.abs(np.array(all_x) - wrist_x))
                 max_abs_y = max(np.abs(np.array(all_y) - wrist_y))
                 scale = max(max_abs_x, max_abs_y)
                 if scale < 1e-6: scale = 1.0
            else:
                 scale = 1.0

            normalized_landmarks = np.array(relative_landmarks) / scale
            feature_vector = normalized_landmarks.flatten()

            # --- Prediction ---
            # Adjust check if padding was implemented
            expected_feature_len = NUM_LANDMARKS * 2
            if feature_vector.shape[0] != expected_feature_len:
                 print(f"Warning: Feature vector shape {feature_vector.shape}. Expected ({expected_feature_len},). Landmarks processed: {num_to_process}")
                 current_prediction_text = "ShapeErr"
                 confidence = 0.0
                 # Reset stability if shape error occurs
                 current_stable_prediction = None
                 stable_prediction_start_time = None
                 if status_message.startswith("Holding"): status_message = ""
            else:
                # Ensure no NaN/inf values before predicting
                if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
                    current_prediction_text = "DataErr"
                    confidence = 0.0
                    print("Warning: NaN or Inf detected in feature vector. Skipping prediction.")
                    current_stable_prediction = None
                    stable_prediction_start_time = None
                    if status_message.startswith("Holding"): status_message = ""
                else:
                    input_data = np.expand_dims(feature_vector, axis=0)
                    prediction_proba = model.predict(input_data, verbose=0)[0]
                    predicted_class_index = np.argmax(prediction_proba)
                    confidence = prediction_proba[predicted_class_index]

                    if confidence >= PREDICTION_THRESHOLD:
                        current_prediction_text = int_to_label.get(predicted_class_index, "Unknown")
                    else:
                        current_prediction_text = "Uncertain"

                    # --- Sign Holding Logic ---
                    valid_sign_for_sequence = current_prediction_text not in ["No Hand", "Uncertain", "Unknown", "ShapeErr", "DataErr", "ErrorProc"]

                    if valid_sign_for_sequence and confidence >= STABLE_CONFIRM_THRESHOLD:
                        if current_prediction_text == current_stable_prediction:
                            if stable_prediction_start_time is not None and time.time() - stable_prediction_start_time >= HOLD_DURATION:
                                if not sentence_sequence or sentence_sequence[-1] != current_prediction_text:
                                    sentence_sequence.append(current_prediction_text)
                                    print(f"Confirmed: {current_prediction_text}. Sequence: {sentence_sequence}")
                                    status_message = f"Added: {current_prediction_text}"
                                    interpreted_sentence = "" # Clear previous interpretation on new add
                                stable_prediction_start_time = time.time() # Reset timer after adding
                        else:
                            current_stable_prediction = current_prediction_text
                            stable_prediction_start_time = time.time()
                            if not interpretation_active:
                                 status_message = f"Holding: {current_prediction_text}?"
                    else:
                        # Reset stability if sign invalid or confidence drops
                        if current_stable_prediction is not None:
                            current_stable_prediction = None
                            stable_prediction_start_time = None
                            if status_message.startswith("Holding") and not interpretation_active:
                                 status_message = ""

            # --- Draw landmarks ---
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                mp_drawing.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
            )
        except Exception as e:
            print(f"ERROR processing landmarks/prediction: {e}") # Keep this uncommented!
            current_prediction_text = "ErrorProc"
            confidence = 0.0
            # Reset stability on any processing error
            current_stable_prediction = None
            stable_prediction_start_time = None
            if status_message.startswith("Holding"): status_message = ""


    else:
        # --- No Hand Detected ---
        current_prediction_text = "No Hand"
        confidence = 0.0
        # Reset stability tracking if hand is lost
        if current_stable_prediction is not None:
             current_stable_prediction = None
             stable_prediction_start_time = None
             if status_message.startswith("Holding") and not interpretation_active: status_message = ""

        # --- Check No Hand Timeout for Interpretation Trigger ---
        # Only trigger if not already waiting for an interpretation result
        if not interpretation_active and sentence_sequence and (time.time() - last_hand_detection_time >= NO_HAND_TIMEOUT):
            print(f"No hand detected for {NO_HAND_TIMEOUT:.1f}s.")
            trigger_interpretation = True
            # Status message set below when thread starts
            # Update time AFTER deciding to trigger to prevent rapid re-triggering
            last_hand_detection_time = time.time()
        # If sequence empty or hand just disappeared recently, keep updating time
        # This check prevents last_hand_detection_time from becoming stale if hand is absent long term
        elif not sentence_sequence:
             last_hand_detection_time = time.time()


    # --- START Interpretation Thread If Triggered ---
    if trigger_interpretation and sentence_sequence and not interpretation_active:
        interpretation_active = True # Set flag: interpretation has started
        status_message = f"Interpreting ({current_language})..." # Update status immediately

        # Copy sequence data FOR the thread (critical!)
        sequence_for_thread = list(sentence_sequence)

        # --- Start the Gemini call in a new thread ---
        gemini_thread = threading.Thread(
            target=call_gemini_in_thread,
            args=(sequence_for_thread, current_language), # Passes current_language here
            daemon=True # Allows program to exit even if thread is stuck
        )
        gemini_thread.start()

        # --- Clear the sequence and reset stability in the MAIN thread ---
        sentence_sequence = []
        current_stable_prediction = None
        stable_prediction_start_time = None
        # Let status_message remain "Interpreting..." until result arrives from queue


    # --- Display Information on Frame ---
    # Prediction & Confidence
    cv2.putText(frame, f"Pred: {current_prediction_text} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    # Current Sequence
    sequence_str = "Seq: " + " ".join(sentence_sequence)
    cv2.putText(frame, sequence_str, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
    # Language Mode
    cv2.putText(frame, f"Lang: {current_language}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    # Status Message (Holding, Interpreting, Added, etc.)
    cv2.putText(frame, status_message, (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA) # Orange color
    # Interpreted Sentence / Word (with text wrapping)
    y0, dy = 150, 40
    max_chars_per_line = max(30, int(frame.shape[1] / 12)) # Basic dynamic wrapping width
    lines_to_display = []
    if interpreted_sentence:
        words = interpreted_sentence.split(' ')
        current_line = ""
        for word in words:
             if '\n' in word:
                 parts = word.split('\n')
                 test_line = (current_line + ' ' + parts[0]).strip()
                 if len(test_line) < max_chars_per_line:
                     current_line = test_line
                     lines_to_display.append(current_line)
                 else:
                     if current_line: lines_to_display.append(current_line)
                     lines_to_display.append(parts[0].strip())
                 for p in parts[1:]: lines_to_display.append(p.strip())
                 current_line = ""
             else:
                 test_line = (current_line + ' ' + word).strip()
                 if len(test_line) < max_chars_per_line:
                     current_line = test_line
                 else:
                     if current_line: lines_to_display.append(current_line)
                     current_line = word
        if current_line: lines_to_display.append(current_line)

    # Display the wrapped lines
    for i, line in enumerate(lines_to_display):
         y = y0 + i * dy
         if y < frame.shape[0] - 10:
            cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2, cv2.LINE_AA)


    # --- Show Frame ---
    cv2.imshow(main_window_name, frame)

    # --- Key Press Handling ---
    key = cv2.waitKey(5) & 0xFF # Use small waitKey for responsiveness
    if key == ord('q'):
        print("Quitting...")
        break
    elif key == ord('c'): # Manual clear
         print("Sequence cleared manually.")
         sentence_sequence = []
         interpreted_sentence = ""
         current_stable_prediction = None
         stable_prediction_start_time = None
         status_message = "Cleared Sequence"
         last_hand_detection_time = time.time() # Prevent immediate trigger
         # If interpretation was active, stop waiting for it
         interpretation_active = False
         # Clear the queue in case something was pending
         while not gemini_result_queue.empty():
             try: gemini_result_queue.get_nowait()
             except queue.Empty: break
             except Exception as e: print(f"Error clearing queue: {e}"); break

# --- Cleanup ---
if cap is not None:
    cap.release()
cv2.destroyAllWindows()
if 'hands' in locals() and hands is not None:
    try:
        hands.close()
    except Exception as e:
        print(f"Error closing MediaPipe hands: {e}")