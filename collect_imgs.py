import os
import cv2
import sys
import time
import string
import numpy as np

DATA_DIR = './data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 27
classes = list(string.ascii_uppercase)
dataset_size = 100

# Function to initialize the camera based on user selection
def initialize_camera(frame):
    cv2.putText(frame, "Select Camera:", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, "1 - Default Webcam", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "2 - Phone Camera", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, "Press 'Q' to Exit", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.imshow('Collect Image', frame)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('1'):  # Default webcam
            return cv2.VideoCapture(0)
        elif key == ord('2'):  # Phone camera (if available)
            return cv2.VideoCapture(1)
        elif key == ord('q'):  # Exit
            cv2.destroyAllWindows()
            sys.exit()
        else:
            print("Invalid selection. Please try again.")

# Initialize the camera
cap = None
while cap is None:
    # Create a black frame for the camera selection menu
    frame = cv2.imread('black.jpg')  # You can use any black image or create one dynamically
    if frame is None:
        frame = np.zeros((300, 500, 3), dtype=np.uint8)

    # Initialize the camera based on user selection
    cap = initialize_camera(frame)

    if not cap.isOpened():
        print("Error: Could not open selected camera.")
        cap = None  # Reset and try again

# Rest of your code remains the same
for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {classes[j]}')

    done = False
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Could not read frame from webcam.")
            break

        cv2.putText(frame, f'Ready {classes[j]}, Press "Space"! :)', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Collect Image', frame)

        key = cv2.waitKey(25)
        if key == ord(' '):
            break
        if key == 27 or cv2.getWindowProperty('Collect Image', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

    # Smooth Countdown (without freezing)
    countdown_values = ["3", "2", "1", "Go.."]
    start_time = time.time()

    for count in countdown_values:
        while time.time() - start_time < 1:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            
            (text_width, text_height), _ = cv2.getTextSize(count, cv2.FONT_HERSHEY_SIMPLEX, 3, 5)
            x_position = (frame.shape[1] - text_width) // 2  
            y_position = 50

            cv2.putText(frame, count, (x_position, y_position), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
            cv2.imshow('Collect Image', frame)
            cv2.waitKey(1)

        start_time = time.time()
        
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("Error: Frame capture failed.")
            break

        cv2.imshow('Collect Image', frame)

        key = cv2.waitKey(25)
        if key == 27 or cv2.getWindowProperty('Collect Image', cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            sys.exit()

        cv2.imwrite(os.path.join(DATA_DIR, str(j), '{}.jpg'.format(counter)), frame)
        counter += 1

    for i in range(50):
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        cv2.putText(frame, 'Done... Continue', (100, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
        cv2.imshow('Collect Image', frame)
        cv2.waitKey(1) 

ret, frame = cap.read()
if ret and frame is not None:
    cv2.putText(frame, 'Thank you!', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow('Collect Image', frame)
    cv2.waitKey(2000)

cap.release()
cv2.destroyAllWindows()