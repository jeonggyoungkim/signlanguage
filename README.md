## 수어 모델 카메라 연결
```
KSLT/
 ├── client/
 │    ├── webcam_client.py         # 웹캠 캡처 + Mediapipe 키포인트 추출 + OpenPose 변환 + 서버 전송
 │    └── utils/mediapipe_to_openpose.py  # 변환 함수 따로 관리
 │
 ├── server/
 │    ├── main.py                  # FastAPI + WebSocket 서버
 │    ├── inference.py             # 모델 불러오기 + 예측 함수
 │    └── utils/preprocess.py      # 입력 전처리 
 └── model_files/
      └── trained_model.ckpt       # 학습된 모델 가중치

```
