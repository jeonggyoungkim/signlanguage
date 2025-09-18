import cv2
import mediapipe as mp
import numpy as np
import asyncio
import websockets
import json
from mediapipe_to_openpose import (
    mediapipe_to_openpose_face,
    mediapipe_to_openpose_body25,
    mediapipe_to_openpose_hand
)

async def send_keypoints():
    uri = "ws://localhost:8000/ws"   # 서버 주소

    async with websockets.connect(uri) as websocket:
        # Mediapipe 솔루션 초기화
        mp_pose = mp.solutions.pose
        mp_hands = mp.solutions.hands
        mp_face = mp.solutions.face_mesh
        mp_drawing = mp.solutions.drawing_utils

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FPS, 30)  # FPS 30 설정

        frame_id = 0
        buffer = []  # 180 프레임 쌓는 버퍼

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose, \
             mp_hands.Hands(model_complexity=1, max_num_hands=2,
                            min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands, \
             mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                              min_detection_confidence=0.5, min_tracking_confidence=0.5) as face:

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                cnt = 0
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False

                pose_results = pose.process(image)
                hands_results = hands.process(image)
                face_results = face.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                h, w = frame.shape[:2]
                body25, openpose_face, openpose_hands = None, None, None

                # pose landmark
                if pose_results.pose_landmarks is not None:
                    cnt += 1
                    body25 = mediapipe_to_openpose_body25(pose_results, w, h, None)
                    body25 = np.nan_to_num(body25, nan=0.0)

                     # 시각화
                    mp_drawing.draw_landmarks(
                        frame,
                        pose_results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(0,0,255), thickness=2, circle_radius=2)
                    )

                # face landmark
                if face_results.multi_face_landmarks is not None:
                    cnt += 1
                    openpose_face = mediapipe_to_openpose_face(face_results, w, h, None)
                    openpose_face = np.nan_to_num(openpose_face, nan=0.0)

                    for face_landmarks in face_results.multi_face_landmarks:
                        for lm in face_landmarks.landmark:
                            x, y = int(lm.x * w), int(lm.y * h)
                            cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)



                # hands landmark
                if hands_results.multi_hand_landmarks is not None:
                    cnt += 1
                    openpose_hands = mediapipe_to_openpose_hand(hands_results, w, h, None)
                    openpose_hands = np.nan_to_num(openpose_hands, nan=0.0)

                    for hand_landmarks in hands_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0,128,255), thickness=2, circle_radius=2)
                        )

                if cnt == 3:
                    # 좌표 추출
                    body25_xy = body25[:, :2]
                    openpose_face_xy = openpose_face[:, :2]
                    openpose_hands_xy = openpose_hands[:, :2]

                    # 통합
                    one_frame_keypoints = np.concatenate(
                        (body25_xy, openpose_face_xy, openpose_hands_xy)
                    )

                    # 정규화
                    one_frame_keypoints[:, 0] /= w
                    one_frame_keypoints[:, 1] /= h

                    # 코 기준 상대 좌표
                    one_frame_keypoints[:, :2] -= one_frame_keypoints[0, :2]

                    # 1차원 배열
                    one_frame_keypoints = one_frame_keypoints.reshape(-1)
                    # 디버깅 출력
                    print("[DEBUG] one_frame_keypoints shape:", one_frame_keypoints.shape)

                    # 버퍼에 저장
                    buffer.append(one_frame_keypoints)

                # ---- 180프레임 모이면 서버 전송 ----
                if len(buffer) == 180:
                    sequence = np.vstack(buffer)

                    data = {
                        "frame_id": frame_id,
                        "sequence": sequence.tolist()
                    }
                    
                    print("[DEBUG] sending frame_id:", frame_id, "sequence shape:", sequence.shape) # 로그 테스트 
                    await websocket.send(json.dumps(data))
                    print(f"[클라이언트] 180프레임 전송 완료 (frame_id={frame_id})")
                    print("[DEBUG] send success") # 로그 테스트 

                    # 서버 응답 (예측 결과)
                    response = await websocket.recv()
                    print("[서버 예측 결과]:", response)

                    buffer = []  # 버퍼 초기화
                    frame_id += 1

                cv2.imshow('camera test', frame)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(send_keypoints())
