import math
from typing import Dict, List, Tuple, Optional

import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose


def angle_3d(a, b, c) -> float:
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    c = np.array(c, dtype=float)
    ba = a - b
    bc = c - b
    denom = (np.linalg.norm(ba) * np.linalg.norm(bc)) + 1e-8
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def distance(a, b) -> float:
    return float(np.linalg.norm(np.array(a, dtype=float) - np.array(b, dtype=float)))


def midpoint(a, b):
    return ((np.array(a, dtype=float) + np.array(b, dtype=float)) / 2.0).tolist()


def get_point(landmarks, idx):
    return landmarks[idx][:3]


def extract_landmarks_from_image(image_bgr: np.ndarray) -> Tuple[Optional[List], Optional[object]]:
    with mp_pose.Pose(static_image_mode=True, model_complexity=0, enable_segmentation=False) as pose:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, None
        landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        return landmarks, results.pose_landmarks


def extract_landmarks_from_video(video_path: str, sample_rate: int = 3) -> Tuple[List, List]:
    cap = cv2.VideoCapture(video_path)
    frames, all_landmarks = [], []
    with mp_pose.Pose(static_image_mode=False, model_complexity=0, enable_segmentation=False) as pose:
        idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            if idx % sample_rate == 0:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)
                if results.pose_landmarks:
                    landmarks = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
                    all_landmarks.append(landmarks)
                    frames.append(frame.copy())
            idx += 1
    cap.release()
    return all_landmarks, frames


def compute_pitching_features(landmarks: List) -> Dict[str, float]:
    pose = mp_pose.PoseLandmark
    l_shoulder = get_point(landmarks, pose.LEFT_SHOULDER.value)
    r_shoulder = get_point(landmarks, pose.RIGHT_SHOULDER.value)
    l_elbow = get_point(landmarks, pose.LEFT_ELBOW.value)
    r_elbow = get_point(landmarks, pose.RIGHT_ELBOW.value)
    l_wrist = get_point(landmarks, pose.LEFT_WRIST.value)
    r_wrist = get_point(landmarks, pose.RIGHT_WRIST.value)
    l_hip = get_point(landmarks, pose.LEFT_HIP.value)
    r_hip = get_point(landmarks, pose.RIGHT_HIP.value)
    l_knee = get_point(landmarks, pose.LEFT_KNEE.value)
    r_knee = get_point(landmarks, pose.RIGHT_KNEE.value)
    l_ankle = get_point(landmarks, pose.LEFT_ANKLE.value)
    r_ankle = get_point(landmarks, pose.RIGHT_ANKLE.value)
    nose = get_point(landmarks, pose.NOSE.value)

    shoulder_center = midpoint(l_shoulder, r_shoulder)
    hip_center = midpoint(l_hip, r_hip)
    knee_center = midpoint(l_knee, r_knee)

    features = {}
    features["right_elbow_flexion"] = angle_3d(r_shoulder, r_elbow, r_wrist)
    features["left_elbow_flexion"] = angle_3d(l_shoulder, l_elbow, l_wrist)
    features["right_arm_extension"] = distance(r_shoulder, r_wrist)
    features["left_arm_extension"] = distance(l_shoulder, l_wrist)
    features["right_shoulder_abduction"] = angle_3d(r_elbow, r_shoulder, r_hip)
    features["left_shoulder_abduction"] = angle_3d(l_elbow, l_shoulder, l_hip)
    features["right_wrist_cock"] = angle_3d(r_elbow, r_wrist, [r_wrist[0], r_wrist[1] - 0.10, r_wrist[2]])
    features["left_wrist_cock"] = angle_3d(l_elbow, l_wrist, [l_wrist[0], l_wrist[1] - 0.10, l_wrist[2]])
    features["right_knee_flexion"] = angle_3d(r_hip, r_knee, r_ankle)
    features["left_knee_flexion"] = angle_3d(l_hip, l_knee, l_ankle)
    features["right_knee_valgus_varus"] = abs(180.0 - features["right_knee_flexion"])
    features["left_knee_valgus_varus"] = abs(180.0 - features["left_knee_flexion"])
    features["lumbar_spine_angle"] = angle_3d(shoulder_center, hip_center, knee_center)
    features["head_position_offset"] = distance(nose, shoulder_center)

    shoulder_angle = math.degrees(math.atan2(r_shoulder[1] - l_shoulder[1], r_shoulder[0] - l_shoulder[0]))
    hip_angle = math.degrees(math.atan2(r_hip[1] - l_hip[1], r_hip[0] - l_hip[0]))
    features["shoulder_rotation"] = abs(shoulder_angle)
    features["hip_rotation"] = abs(hip_angle)
    features["hip_shoulder_separation"] = abs(shoulder_angle - hip_angle)
    features["stride_width"] = distance(l_ankle, r_ankle)
    features["lead_leg_block"] = max(features["right_knee_flexion"], features["left_knee_flexion"])
    features["trunk_tilt"] = angle_3d(shoulder_center, hip_center, [hip_center[0], hip_center[1] + 0.25, hip_center[2]])
    return features


def aggregate_sequence_features(sequence_features: List[Dict]) -> Dict[str, float]:
    if not sequence_features:
        return {}
    keys = sequence_features[0].keys()
    aggregated = {}
    for key in keys:
        values = np.array([f[key] for f in sequence_features], dtype=float)
        aggregated[f"{key}_mean"] = float(np.mean(values))
        aggregated[f"{key}_max"] = float(np.max(values))
        aggregated[f"{key}_min"] = float(np.min(values))
        aggregated[f"{key}_std"] = float(np.std(values))
    return aggregated


def get_feature_vector_from_landmarks_sequence(landmarks_sequence: List) -> Tuple[Dict, List]:
    per_frame = [compute_pitching_features(lm) for lm in landmarks_sequence]
    agg = aggregate_sequence_features(per_frame)
    return agg, per_frame