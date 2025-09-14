import cv2
import mediapipe as mp
import numpy as np
from typing import Optional

from typing import Optional
import numpy as np

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



mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0)
with mp_hands.Hands(model_complexity=1, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened(): 
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        
        # BGR -> RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
      
        # 감지하기
        hands_results = hands.process(image)
    
        # RGB -> BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        
        # 손 랜드마크
        if hands_results.multi_hand_landmarks is not None:
           openpose_hands=mediapipe_to_openpose_hand(hands_results, w, h, None)

           for i, point in enumerate(openpose_hands):
                x, y, _ = point
                if not (np.isnan(x) or np.isnan(y)):
                    cv2.circle(image, (int(x), int(y)), 4, (0, 255, 0), -1)
        
        cv2.imshow('Mediapipe Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()