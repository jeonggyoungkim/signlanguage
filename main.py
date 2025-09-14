from fastapi import FastAPI, WebSocket
import json

app = FastAPI()

# /ws 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 연결 허용
    while True:
        data = await websocket.receive_text()   # 클라이언트에서 메시지 받기
        print(f"클라이언트에서 받은 데이터: {data}")

        # 클라이언트로 응답 보내기
        await websocket.send_text(f"서버 응답: {data}")
