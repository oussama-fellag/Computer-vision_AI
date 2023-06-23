# import cv2
# import mediapipe as mp
# import numpy as np
# import math
# import time
# from ctypes import cast, POINTER
# from comtypes import CLSCTX_ALL
# from pycaw.pycaw import *
#
# devices = AudioUtilities.GetSpeakers()
# interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
# vc = cast(interface, POINTER(IAudioEndpointVolume))
# Range = vc.GetVolumeRange()
# minR, maxR = Range[0], Range[1]
#
# mpHands = mp.solutions.hands
# Hands = mpHands.Hands()
# mpDraw = mp.solutions.drawing_utils
# PTime = 0
# vol = 0
# volBar = 400
# volPer = 0
# cap = cv2.VideoCapture(0)
#
# while (cap.isOpened()):
#     lmList = []
#     success, img = cap.read()
#     converted_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     results = Hands.process(converted_image)
#
#     if results.multi_hand_landmarks:
#         for hand_in_frame in results.multi_hand_landmarks:
#             mpDraw.draw_landmarks(img, hand_in_frame, mpHands.HAND_CONNECTIONS)
#         for id, lm in enumerate(results.multi_hand_landmarks[0].landmark):
#             h, w, c = img.shape
#             cx, cy = int(lm.x * w), int(lm.y * h)
#             lmList.append([cx, cy])
#
#         if len(lmList) != 0:
#             x1, y1 = lmList[4][0], lmList[4][1]
#             x2, y2 = lmList[8][0], lmList[8][1]
#
#             cv2.circle(img, (x1, y1), 15, (255, 0, 0), cv2.FILLED)
#             cv2.circle(img, (x2, y2), 15, (255, 0, 0), cv2.FILLED)
#             cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3, cv2.FILLED)
#             length = math.hypot(x2 - x1 - 30, y2 - y1 - 30)
#
#             vol = np.interp(length, [50, 300], [minR, maxR])
#             volBar = np.interp(length, [50, 300], [400, 150])
#             volPer = np.interp(length, [50, 300], [0, 100])
#
#             cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0))
#             cv2.rectangle(img, (50, int(volBar)), (85, 400), (255, 0, 0), cv2.FILLED)
#             cv2.putText(img, f'{int(volPer)} %', (85, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
#             vc.SetMasterVolumeLevel(vol, None)
#
#     CTime = time.time()
#     fps = 1 / (CTime - PTime)
#     PTime = CTime
#     cv2.putText(img, str(int(fps)), (40, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)
#
#     cv2.imshow("Hand Tracking", img)
#
#     if cv2.waitKey(1) == 113:  # 113 - Q
#         break
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import *
import os
import matplotlib.pyplot as plt

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
vc = cast(interface, POINTER(IAudioEndpointVolume))
min_vol, max_vol, vol_step = vc.GetVolumeRange()
vol_range = max_vol - min_vol

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
prev_time = 0
vol = 0
vol_bar = 400
vol_percentage = 0
cap = cv2.VideoCapture(0)

# Initialize lists for plotting
vol_perc_history = []
time_history = []

while cap.isOpened():
    success, img = cap.read()
    converted_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(converted_img)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            x1, y1 = int(hand_landmarks.landmark[4].x * img.shape[1]), int(hand_landmarks.landmark[4].y * img.shape[0])
            x2, y2 = int(hand_landmarks.landmark[8].x * img.shape[1]), int(hand_landmarks.landmark[8].y * img.shape[0])
            length = math.hypot(x2 - x1, y2 - y1)

            vol_percentage = int(np.interp(length, [50, 300], [0, 100]))
            vol = np.interp(length, [50, 300], [min_vol, max_vol])
            vc.SetMasterVolumeLevel(vol, None)

            vol_bar = np.interp(length, [50, 300], [400, 150])
            cv2.rectangle(img, (50, 150), (85, 400), (255, 0, 0))
            cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (255, 0, 0), cv2.FILLED)
            cv2.putText(img, f'{vol_percentage}%', (40, 450), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

            vol_perc_history.append(vol_percentage)
            time_history.append(time.time())

            if vol_percentage == 100:
                plt.plot(time_history, vol_perc_history)
                plt.xlabel('Time (s)')
                plt.ylabel('Volume Percentage (%)')
                plt.savefig('C:/Users/DELL/Desktop/S2/volume_history.png')
                cap.release()
                cv2.destroyAllWindows()
                exit(0)

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Hand Gesture Volume Control", img)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
