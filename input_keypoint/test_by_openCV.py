import cv2
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

def mediapipe_to_openpose_body25(mediapipe_landmarks: np.ndarray,
                        visibility: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Convert MediaPipe Pose landmarks to OpenPose BODY_25 format.
    
    mediapipe_landmarks: (33, 3) → (x, y, z) normalized [0,1]
    visibility: (33,) → confidence/visibility (optional)
    
    return: (25, 3) → (x, y, confidence) in normalized coordinates
    """
    body25 = np.full((25, 3), np.nan, dtype=np.float32)

    for body_idx, mp_idx in enumerate(mp_to_body25_indices):
        if mp_idx is None:
            body25[body_idx] = [np.nan, np.nan, 0.0]
            continue

        if isinstance(mp_idx, tuple):  # Neck, MidHip
            pts = [mediapipe_landmarks[i] for i in mp_idx if i is not None]
            vis = [visibility[i] if visibility is not None else 1.0 for i in mp_idx]
            if len(pts) == 2:
                x = (pts[0][0] + pts[1][0]) / 2
                y = (pts[0][1] + pts[1][1]) / 2
                conf = (vis[0] + vis[1]) / 2
            else:
                x, y, conf = np.nan, np.nan, 0.0
        else:
            x, y, _ = mediapipe_landmarks[mp_idx]
            conf = visibility[mp_idx] if visibility is not None else 1.0

        body25[body_idx] = [x, y, conf]

    return body25

def mediapipe_to_openpose_face(
    mediapipe_landmarks: np.ndarray,
    visibility: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Convert MediaPipe Face Mesh landmarks to OpenPose Face 70 format.

    mediapipe_landmarks: shape (468, 3) → (x, y, z)
    visibility: shape (468,) → confidence/visibility (optional)
    return: shape (70, 3) → (x, y, confidence)
    """
    selected_points = []
    for idx in mp_to_face_indices:
        x, y, z = mediapipe_landmarks[idx]
        conf = visibility[idx] if visibility is not None else 1.0
        selected_points.append([x, y, conf])

    return np.array(selected_points, dtype=np.float32)

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_hands = mp.solutions.hands
    mp_face = mp.solutions.face_mesh

    cap = cv2.VideoCapture(0)
    # 포즈/손/얼굴 인스턴스 설정(신뢰도는 0.5, 연속 프레임 신뢰도 0.5)
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
        mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
        mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:
        while cap.isOpened(): 
            ret, frame = cap.read()
            if not ret:
                break
        
            # BGR -> RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
      
            # detect
            pose_results = pose.process(image)
            hands_results = hands.process(image)
            face_results = face.process(image)
    
            # RGB -> BGR
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #image size for coordinate conversion from normalized to pixel coordinates
            h, w = frame.shape[:2]
        
            # hands landmark
            if hands_results.multi_hand_landmarks is not None:
                for hand_landmarks in hands_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,128,255), thickness=2, circle_radius=2)
                    )
        
            # face landmark
            if face_results.multi_face_landmarks is not None:
                # Convert mediapipe face (468) to openpose-like 70 using the mapping function you have
                lm = face_results.multi_face_landmarks[0].landmark
                pts = np.array([[p.x * w, p.y * h, 0.0] for p in lm], dtype=np.float32)
                openpose70 = mediapipe_to_openpose_face(pts[:, :3], None)

                for i, (x, y, c) in enumerate(openpose70):
                    if not (np.isnan(x) or np.isnan(y)):
                        cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1)

                        # pose landmark
            if pose_results.pose_landmarks is not None:
                lm=pose_results.pose_landmarks.landmark
                pts=np.array([[p.x * w, p.y * h, 0.0] for p in lm], dtype=np.float32)
                body25 = mediapipe_to_openpose_body25(pts[:, :3], None)

                # draw subset of keypoints
                for idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]:
                    x, y, c = body25[idx]
                    if c > 0.3 and not (np.isnan(x) or np.isnan(y)):
                        cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                        cv2.putText(frame, str(idx), (int(x)+4, int(y)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
            cv2.imshow('Mediapipe Feed', frame)

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ =="__main__":
    main()