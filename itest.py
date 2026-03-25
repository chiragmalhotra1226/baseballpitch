import mediapipe as mp
import numpy as np
import cv2
mp_pose = mp.solutions.pose
img = np.zeros((480,640,3), dtype=np.uint8)
with mp_pose.Pose(static_image_mode=True, model_complexity=0) as pose:
    result = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    print('model_complexity=0 works!')