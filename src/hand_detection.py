'''
STEP 1: 
Start camera connection.
While the camera is open:
    Read a frame.
    Show that frame in a window.
    If 'q' key is pressed, break the loop.
After the loop:
    Release the camera.
    Close all windows.
'''

import cv2
import mediapipe as mp
import time


web_cam = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

start_time = time.time()
duration = 5  # seconds

# (Optional test)
# is_open = web_cam.isOpened()
# print("Camera open:", is_open)

while web_cam.isOpened():
    success, frame = web_cam.read()
    if not success:
        print("Failed to grab frame. Exiting...")
        break
    
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    
    # if results.multi_handedness:
    #     print(results.multi_handedness)


    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, 
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
                )



    cv2.imshow('GuitarCam', frame)


    if time.time() - start_time > duration:
        print("Session time complete")
        break
   


web_cam.release()
cv2.destroyAllWindows()

