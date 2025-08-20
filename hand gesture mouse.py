# Import necessary libraries
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Initialize Mediapipe Hands and Drawing utilities with improved parameters
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
# Define smoothing factor and initialize previous cursor positions
smooth_factor = 4  # Adjust this value for smoother cursor movement
prev_x, prev_y = 0, 0

# Initialize Mediapipe Hands and Drawing utilities with improved parameters
hands = mp_hands.Hands(
    max_num_hands=1  # Ensure only one hand is tracked for better performance
)

# Get screen dimensions
screen_width, screen_height = pyautogui.size()

# Start video capture
cap = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    """Calculate the Euclidean distance between two points."""
    return math.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a mirror-like effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Convert the frame to RGB for Mediapipe processing
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract landmark positions
            landmarks = hand_landmarks.landmark

            # Get coordinates of index tip
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            pinky_tip = landmarks[mp_hands.HandLandmark.PINKY_TIP]

            # Convert normalized coordinates to screen coordinates
            index_x, index_y = int(index_tip.x * w), int(index_tip.y * h)
            screen_x = np.interp(index_x, [0, w], [0, screen_width])
            screen_y = np.interp(index_y, [0, h], [0, screen_height])

            # Smooth cursor movement
            curr_x = prev_x + (screen_x - prev_x) / smooth_factor
            curr_y = prev_y + (screen_y - prev_y) / smooth_factor
            pyautogui.moveTo(curr_x, curr_y)
            prev_x, prev_y = curr_x, curr_y

            # Calculate distances for gestures
            thumb_pinky_distance = calculate_distance(thumb_tip, pinky_tip)
            thumb_index_distance = calculate_distance(thumb_tip, index_tip)

            # Define thresholds for gestures
            click_threshold = 0.05  # Adjust this value for accuracy

            # Perform left click when thumb and pinky are close
            if thumb_pinky_distance < click_threshold:
                pyautogui.click(button='left')

            # Perform right click when thumb and index are close
            if thumb_index_distance < click_threshold:
                pyautogui.click(button='right')

            # Draw hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("Gesture Control Mouse", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
