import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import json
from typing import Optional
from client.mediapipe_to_openpose import (
    mediapipe_to_openpose_face,
    mediapipe_to_openpose_body25,
    mediapipe_to_openpose_hand
)

async def mediapipe_keypoints():
    uri = "ws://localhost:8000/ws"  # 서버주소

    async with websockets.connect(uri) as websocket:
        mp_drawing = mp.solutions.drawing_utils
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_mesh

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)  # FPS 30 설정

        # 포즈/손/얼굴 인스턴스 설정
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
             mp_hands.Hands(model_complexity=1, max_num_hands=2,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
             mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:

            # 프레임 버퍼 (180프레임 저장)
            buffer = []

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

                # image size for coordinate conversion
                h, w = frame.shape[:2]

                body25, openpose_face, openpose_hands = None, None, None

                # pose landmark
                if pose_results.pose_landmarks is not None:
                    body25 = mediapipe_to_openpose_body25(pose_results, w, h, None)

                    # draw subset of keypoints
                    for idx, (x, y, c) in enumerate(body25):
                        if c > 0.3 and not (np.isnan(x) or np.isnan(y)):
                            cv2.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)
                            cv2.putText(frame, str(idx), (int(x)+4, int(y)-4),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
                    body25 = np.nan_to_num(body25, nan=0.0)

                # face landmark
                if face_results.multi_face_landmarks is not None:
                    openpose_face = mediapipe_to_openpose_face(face_results, w, h, None)

                    for x, y, c in openpose_face:
                        if not (np.isnan(x) or np.isnan(y)):
                            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

                    openpose_face = np.nan_to_num(openpose_face, nan=0.0)

                # hands landmark
                if hands_results.multi_hand_landmarks is not None:
                    openpose_hands = mediapipe_to_openpose_hand(hands_results, w, h, None)

                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 128, 255), thickness=2, circle_radius=2)
                        )
                    openpose_hands = np.nan_to_num(openpose_hands, nan=0.0)

                # 세 가지 다 감지된 경우만 처리
                if body25 is not None and openpose_face is not None and openpose_hands is not None:
                    # x , y 추출
                    body25_xy = body25[:, :2]
                    openpose_face_xy = openpose_face[:, :2]
                    openpose_hands_xy = openpose_hands[:, :2]

                    # 통합
                    one_frame_keypoints = np.concatenate((body25_xy, openpose_face_xy, openpose_hands_xy))

                    # 정규화
                    one_frame_keypoints[:, 0] /= w
                    one_frame_keypoints[:, 1] /= h

                    # 코(0번 인덱스) 기준 상대 좌표 변환
                    one_frame_keypoints[:, :2] -= one_frame_k
