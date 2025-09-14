import cv2
import mediaipipe as mp
import asyncio
import websockets
import json

async def send_keypoints():
    uri = "ws://localhost:8000/ws"   # 서버 주소

    async with websockets.connect(uri) as websocket:
        cap = cv2.VideoCapture(0)
        mp_pose = mp.solutions.pose.Pose()

        frame_id = 0
        while cap.is0pened():
            ret, frame = cap.read()
            if not ret:
                break
            
            results = mp_pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        # 임시 json 데이터
        json_data = {
            "frame_id" : 1,
            "keypoints" : [
                [0.1, 0.2, 0.95],
                [0.3, 0.4, 0.90],
                [0.5, 0.6, 0.85]
            ]
        }
        # json 문자열로 변환하여 전송
        await websocket.send(json.dumps(json_data)) # dump(데이터, 파일객체) 파일 저장 함수 / 현재는 문자열 dumps
        print("보낸 데이터:", json_data)

        # 서버 응답 받기
        response = await websocket.recv()
        print("서버 응답:", response)

# 실행
asyncio.run(test_ws())
