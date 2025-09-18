from fastapi import FastAPI, WebSocket
import json
import torch
import numpy as np
import yaml

from signjoey.model import build_model, SignModel
from signjoey.vocabulary import Vocabulary

ckpt_path = "C:\\KSLT\\model_files\\300.ckpt" # ckpt 경로
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
print(ckpt.keys())

# config.yaml 읽어오기
with open(r"C:\\KSLT\\model_files\\config.yaml", "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# vocab은 ckpt에서 가져오기
trg_vocab = ckpt["trg_vocab"]

# 모델 빌드
model = build_model(cfg, trg_vocab)
model.load_state_dict(ckpt["model_state"])
model.to(device)
model.eval()

id2label = {i: tok for i, tok in enumerate(trg_vocab.itos)}

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
            sequence = np.array(parsed["sequence"], dtype=np.float32).reshape(180, 274)

            # 모델 입력 준비
            input_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 180, 274)
            src_length = torch.tensor([180], dtype=torch.long).to(device)
            src_mask = torch.ones((1, 1, 180), dtype=torch.bool).to(device)
            
            # 디버깅 출력
            print("[SERVER DEBUG] input_tensor:", input_tensor.shape)
            print("[SERVER DEBUG] src_length:", src_length.shape, src_length)
            print("[SERVER DEBUG] src_mask:", src_mask.shape)

            # 추론
            with torch.no_grad():
                _, logits = model(
                    src=input_tensor,
                    src_length=src_length,
                    src_mask=src_mask
                )
                pred = torch.argmax(logits, dim=-1).item()
                label = id2label.get(pred, "unknown")

            result = {
                "frame_id": frame_id,
                "prediction": label
            }

            await websocket.send_text(json.dumps(result))
           
        except Exception as e:
            import traceback
            print("에러 발생:", e)
            traceback.print_exc()
            break