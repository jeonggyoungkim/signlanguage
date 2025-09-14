from fastapi import FastAPI, WebSocket
import json

app = FastAPI()

# /ws 엔드포인트
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()  # 연결 허용
    while True:
        try:
            data = await websocket.receive_text()
            parsed = json.loads(data)

            # frame_id, sequence 길이 확인
            frame_id = parsed["frame_id"]
            seq_len = len(parsed["sequence"])

            print(f"[서버] frame_id={frame_id}, sequence_len={seq_len}")

            # 그대로 클라이언트로 돌려주기
            await websocket.send_text(
                json.dumps({
                    "received_frame_id": frame_id,
                    "received_seq_len": seq_len
                })
            )

        except Exception as e:
            print("에러 발생:", e)
            break