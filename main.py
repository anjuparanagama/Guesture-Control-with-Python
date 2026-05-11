"""
Enhanced Smart Gesture Drawing System
====================================

Controls:
☝️ 1 Finger  -> Draw
✌️ 2 Fingers -> Move
🤟 3 Fingers -> Change Color
🖐️ 5 Fingers -> Clear Canvas

Keyboard:
Q -> Quit
C -> Clear
S -> Save Drawing

Install:
pip install opencv-python mediapipe numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import math

# ==========================================
# MediaPipe Setup
# ==========================================

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# ==========================================
# Colors
# ==========================================

COLORS = [
    (0, 255, 255),   # Yellow
    (255, 0, 255),   # Purple
    (255, 255, 0),   # Cyan
    (0, 255, 0),     # Green
    (0, 0, 255),     # Red
]

color_index = 0
DRAW_COLOR = COLORS[color_index]

BG_COLOR = (20, 20, 30)

# ==========================================
# Brush Settings
# ==========================================

brush_size = 6
eraser_size = 40

# ==========================================
# Finger IDs
# ==========================================

TIPS = [4, 8, 12, 16, 20]
PIPS = [3, 6, 10, 14, 18]

# ==========================================
# Utility Functions
# ==========================================

def count_fingers(hand_landmarks, handedness_label):
    lm = hand_landmarks.landmark
    fingers = 0

    # Thumb
    if handedness_label == "Right":
        if lm[TIPS[0]].x < lm[PIPS[0]].x:
            fingers += 1
    else:
        if lm[TIPS[0]].x > lm[PIPS[0]].x:
            fingers += 1

    # Other fingers
    for tip, pip in zip(TIPS[1:], PIPS[1:]):
        if lm[tip].y < lm[pip].y:
            fingers += 1

    return fingers


def get_index_tip(hand_landmarks, w, h):
    tip = hand_landmarks.landmark[8]
    return int(tip.x * w), int(tip.y * h)


def draw_ui(frame, mode, fps):
    overlay = frame.copy()

    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70),
                  (30, 30, 40), -1)

    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Mode text
    cv2.putText(frame, f"MODE: {mode}",
                (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                DRAW_COLOR,
                2)

    # FPS
    cv2.putText(frame, f"FPS: {int(fps)}",
                (1050, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (200, 200, 200),
                2)

    # Color palette
    x = 350
    for i, c in enumerate(COLORS):
        cv2.circle(frame, (x, 35), 15, c, -1)

        if i == color_index:
            cv2.circle(frame, (x, 35), 22, (255, 255, 255), 2)

        x += 50


# ==========================================
# Main
# ==========================================

def main():

    global color_index
    global DRAW_COLOR

    cap = cv2.VideoCapture(0)

    cap.set(3, 1280)
    cap.set(4, 720)

    success, frame = cap.read()

    if not success:
        print("Camera not found")
        return

    h, w = frame.shape[:2]

    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    prev_x, prev_y = None, None

    p_time = time.time()

    with mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as hands:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.flip(frame, 1)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            mode = "MOVE"

            cx, cy = None, None

            if result.multi_hand_landmarks:

                for hand_lm, handedness in zip(
                    result.multi_hand_landmarks,
                    result.multi_handedness
                ):

                    label = handedness.classification[0].label

                    fingers = count_fingers(hand_lm, label)

                    cx, cy = get_index_tip(hand_lm, w, h)

                    # Draw landmarks
                    mp_draw.draw_landmarks(
                        frame,
                        hand_lm,
                        mp_hands.HAND_CONNECTIONS
                    )

                    # ==================================
                    # Gesture Modes
                    # ==================================

                    if fingers == 1:
                        mode = "DRAW"

                    elif fingers == 2:
                        mode = "MOVE"

                    elif fingers == 3:
                        mode = "COLOR"

                    elif fingers == 5:
                        mode = "CLEAR"

                    # ==================================
                    # Drawing
                    # ==================================

                    if mode == "DRAW":

                        if prev_x is None:
                            prev_x, prev_y = cx, cy

                        # Smooth drawing
                        cv2.line(
                            canvas,
                            (prev_x, prev_y),
                            (cx, cy),
                            DRAW_COLOR,
                            brush_size,
                            cv2.LINE_AA
                        )

                        prev_x, prev_y = cx, cy

                    else:
                        prev_x, prev_y = None, None

                    # ==================================
                    # Change Color
                    # ==================================

                    if mode == "COLOR":

                        color_index = (color_index + 1) % len(COLORS)
                        DRAW_COLOR = COLORS[color_index]

                        time.sleep(0.3)

                    # ==================================
                    # Clear Canvas
                    # ==================================

                    if mode == "CLEAR":
                        canvas[:] = 0

            # ==========================================
            # Merge Canvas
            # ==========================================

            gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)

            _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

            mask_inv = cv2.bitwise_not(mask)

            bg = cv2.bitwise_and(frame, frame, mask=mask_inv)

            fg = cv2.bitwise_and(canvas, canvas, mask=mask)

            combined = cv2.add(bg, fg)

            # ==========================================
            # Cursor
            # ==========================================

            if cx is not None:

                cv2.circle(
                    combined,
                    (cx, cy),
                    12,
                    DRAW_COLOR,
                    3
                )

            # ==========================================
            # FPS
            # ==========================================

            c_time = time.time()

            fps = 1 / (c_time - p_time)

            p_time = c_time

            # ==========================================
            # UI
            # ==========================================

            draw_ui(combined, mode, fps)

            cv2.imshow("AI Gesture Drawing", combined)

            key = cv2.waitKey(1)

            # Quit
            if key == ord('q'):
                break

            # Clear
            elif key == ord('c'):
                canvas[:] = 0

            # Save
            elif key == ord('s'):

                filename = f"drawing_{int(time.time())}.png"

                cv2.imwrite(filename, combined)

                print(f"Saved: {filename}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()