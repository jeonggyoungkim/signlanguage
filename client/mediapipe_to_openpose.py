import mediapipe as mp
import numpy as np
from typing import Optional

#mediapipe pose -> openpose body25 mapping table
mp_to_body25_indices = [
    0,      # 0 Nose
    (11, 12),  # 1 Neck (avg of shoulders)
    12, 14, 16,  # 2–4 Right arm
    11, 13, 15,  # 5–7 Left arm
    (23, 24),   # 8 MidHip (avg of hips)
    24, 26, 28,  # 9–11 Right leg
    23, 25, 27,  # 12–14 Left leg
    5, 2, 8, 7,  # 15–18 Eyes/Ears
    31, None, 29,  # 19–21 Left foot
    32, None, 30   # 22–24 Right foot
]

#mediapipr face -> openpose face mapping table
mp_to_face_indices = [
    356, 447, 401, 288, 397, 365, 378, 377, 152, 176, 150, 136, 172, 58, 132, 93, 127, # 0–16 Jawline
    107, 66, 105, 63, 70,  # 17–21 Right Eyebrow
    336, 296, 334, 293, 300,  # 22–26 Left Eyebrow
    168, 197, 5, 4,# 27–30 Nose bridge
    59, 60, 2, 290, 289,  # 31–35 Nose bottom
    33, 160, 158, 133, 153, 163,  # 36–41 Right eye
    362, 385, 387, 263, 373, 380,  # 42–47 Left eye
    61, 40, 37, 0, 267, 270, 291, 321, 314, 17, 84, 91, #48-59 mouth outer
    78, 81, 13, 311, 308, 402, 14, 178,  # 60–67 mouth inner
    468, 473 #68-69 iris
]

def mediapipe_to_openpose_hand(hands_results,
                        width: int, heigth: int,
                        visibility: Optional[np.ndarray] = None) -> np.ndarray:
    
    hands_landmarks=np.full((42,3), np.nan, dtype=np.float32) #left -> right
    landmarks=hands_results.multi_hand_landmarks.landmark
    points=np.array([[p.x * width, p.y * heigth, 0.0] for p in landmarks], dtype=np.float32)

    for idx in range(len(points)):
        x, y, z = points[idx]
        conf = visibility[idx] if visibility is not None else 1.0
        hands_landmarks[idx]= [x, y, conf]
    return hands_landmarks
    

def mediapipe_to_openpose_body25(pose_results,
                        width: int, heigth: int,
                        visibility: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert MediaPipe Pose landmarks to OpenPose BODY_25 format.
    
    mediapipe_landmarks: (33, 3) → (x, y, z) normalized [0,1]
    visibility: (33,) → confidence/visibility (optional)
    
    return: (25, 3) → (x, y, confidence) in normalized coordinates
    """
    body25 = np.full((25, 3), np.nan, dtype=np.float32)

    landmarks=pose_results.pose_landmarks.landmark
    points=np.array([[p.x * width, p.y * heigth, 0.0] for p in landmarks], dtype=np.float32)


    for body_idx, mp_idx in enumerate(mp_to_body25_indices):
        if mp_idx is None:
            body25[body_idx] = [np.nan, np.nan, 0.0]
            continue

        if isinstance(mp_idx, tuple):  # Neck, MidHip
            pts = [points[i] for i in mp_idx if i is not None]
            vis = [visibility[i] if visibility is not None else 1.0 for i in mp_idx]
            if len(pts) == 2:
                x = (pts[0][0] + pts[1][0]) / 2
                y = (pts[0][1] + pts[1][1]) / 2
                conf = (vis[0] + vis[1]) / 2
            else:
                x, y, conf = np.nan, np.nan, 0.0
        else:
            x, y, _ = points[mp_idx]
            conf = visibility[mp_idx] if visibility is not None else 1.0

        body25[body_idx] = [x, y, conf]

    return body25

def mediapipe_to_openpose_face(face_results,
    width: int, heigth: int,
    visibility: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert MediaPipe Face Mesh landmarks to OpenPose Face 70 format.

    mediapipe_landmarks: shape (468, 3) → (x, y, z)
    visibility: shape (468,) → confidence/visibility (optional)
    return: shape (70, 3) → (x, y, confidence)
    """

    face_landmarks = np.full((70, 3), np.nan, dtype=np.float32)

    landmarks=face_results.multi_face_landmarks[0].landmark
    points=np.array([[p.x * width, p.y * heigth, 0.0] for p in landmarks], dtype=np.float32)

    for op_idx, mp_idx in enumerate(mp_to_face_indices):
        x, y, z = points[mp_idx]
        conf = visibility[mp_idx] if visibility is not None else 1.0
        face_landmarks[op_idx] = [x, y, conf]

    return face_landmarks

def mediapipe_to_openpose_hand(hands_results,
                               width: int, height: int,
                               visibility: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert MediaPipe Hands landmarks to OpenPose 42 format (21 left + 21 right).
    Left hand: 0–20, Right hand: 21–41
    """
    hands_landmarks = np.full((42, 3), np.nan, dtype=np.float32)  # left→right

    if not hands_results.multi_hand_landmarks:
        return hands_landmarks

    for hand_idx, hand_landmarks in enumerate(hands_results.multi_hand_landmarks):
        handedness = hands_results.multi_handedness[hand_idx].classification[0].label.lower()
        base_idx = 0 if handedness == "left" else 21  # 왼손=0~20, 오른손=21~41

        points = np.array(
            [[p.x * width, p.y * height, 0.0] for p in hand_landmarks.landmark],
            dtype=np.float32
        )
        points = np.nan_to_num(points, nan=0.0)  # NaN → 0.0 변환

        for i, (x, y, z) in enumerate(points):
            conf = visibility[base_idx + i] if visibility is not None else 1.0
            hands_landmarks[base_idx + i] = [x, y, conf]

    return hands_landmarks