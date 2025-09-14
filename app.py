import torch
import yaml
import pandas as pd
import numpy as np
import os

# signjoey 폴더가 현재 경로에 있으므로, 관련 모듈을 가져올 수 있습니다.
from signjoey.builders import build_model
from signjoey.vocabulary import Vocabulary

def load_model_and_vocab(config_path, ckpt_path, vocab_path):
    """
    설정 파일, 체크포인트, 어휘 사전을 로드하여 추론 준비가 된 모델을 반환합니다.
    """
    print("1. 설정 파일 로딩...")
    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
    
    model_config = config["model"]
    
    print("2. 어휘 사전 로딩...")
    vocab = Vocabulary(file=vocab_path)
    
    # 디바이스 설정 (GPU가 있으면 사용, 없으면 CPU 사용)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"   > 사용 디바이스: {device}")

    print("3. 모델 아키텍처 빌드...")
    model = build_model(
        cfg=model_config,
        gls_vocab=vocab,
        sgn_dim=150, # 설정 파일에 따라 고정된 값
        do_recognition=False,
        do_translation=False
    )

    print("4. 학습된 가중치 로딩...")
    # 체크포인트 파일 로드
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    # 모델의 state_dict를 체크포인트의 것으로 업데이트
    model.load_state_dict(checkpoint['model_state'])
    
    # 모델을 device로 이동
    model.to(device)
    
    # 모델을 추론 모드(evaluation mode)로 설정
    # 드롭아웃 등 학습 시에만 필요한 기능들을 비활성화합니다.
    model.eval()
    
    print("   > 모델 로딩 완료!")
    return model, vocab, device

def preprocess_keypoints(csv_path):
    """
    Keypoint가 저장된 CSV 파일을 모델 입력에 맞게 전처리합니다.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    print(f"5. 입력 데이터 전처리... ({os.path.basename(csv_path)})")
    # CSV 파일을 pandas DataFrame으로 읽기
    df = pd.read_csv(csv_path)
    
    # 필요한 모든 좌표(x, y, z, visibility) 값을 numpy 배열로 변환
    keypoints = df.values.flatten().astype(np.float32)
    
    # PyTorch 텐서로 변환
    tensor = torch.tensor(keypoints)
    
    # 모델은 (batch_size, sequence_length, feature_dim) 형태의 입력을 기대하므로
    # 현재 데이터를 (1, num_frames, num_keypoints * num_coords) 형태로 맞춰줘야 합니다.
    # 이 모델은 (batch_size, sequence_length) 형태를 사용하므로 2D로 만듭니다.
    # [1, sequence_length] 형태로 변환 (배치 크기 1 추가)
    tensor = tensor.unsqueeze(0)
    
    print("   > 전처리 완료!")
    return tensor

def predict(model, data_tensor, vocab, device):
    """
    전처리된 데이터를 모델에 입력하여 예측 결과를 반환합니다.
    """
    print("6. 모델 추론 수행...")
    with torch.no_grad(): # 기울기 계산을 비활성화하여 메모리 사용량을 줄이고 속도를 높임
        # 데이터를 모델과 동일한 디바이스로 이동
        data_tensor = data_tensor.to(device)
        
        # 소스 마스크 생성 (입력 시퀀스의 모든 타임스텝이 유효하다고 가정)
        src_mask = model.make_src_mask(data_tensor)

        # 모델 포워드 패스 실행
        # 이 모델의 Classifier는 src와 src_mask만 필요합니다.
        output, _ = model.forward(
            src=data_tensor,
            trg=None,
            src_mask=src_mask,
            trg_mask=None
        )

        # 결과 해석
        # output은 각 단어(클래스)에 대한 로짓(logits) 값입니다.
        # 가장 높은 값을 가진 인덱스를 찾습니다.
        predicted_index = torch.argmax(output, dim=1).item()
        
        # 인덱스를 실제 단어로 변환
        predicted_word = vocab.itos[predicted_index]
        
    print("   > 추론 완료!")
    return predicted_word

# --- 메인 실행 ---
if __name__ == "__main__":
    # 프로젝트 구조에 맞게 파일 경로 설정
    CONFIG_PATH = "model_files/config.yaml"
    CKPT_PATH = "model_files/240.ckpt"
    VOCAB_PATH = "data/auto_generated_vocab.txt"
    
    # 테스트할 샘플 데이터 (사전 준비 단계에서 복사한 파일)
    SAMPLE_CSV_PATH = "NIA_SL_WORD0009_REAL01_F_keypoints.csv"
    
    print("="*30)
    print("      수어 인식 모델 추론 시작")
    print("="*30)
    
    try:
        # 1-4단계: 모델 및 필요 데이터 로딩
        model, vocab, device = load_model_and_vocab(CONFIG_PATH, CKPT_PATH, VOCAB_PATH)
        
        # 5단계: 추론할 데이터 전처리
        input_tensor = preprocess_keypoints(SAMPLE_CSV_PATH)
        
        # 6단계: 예측 수행
        prediction = predict(model, input_tensor, vocab, device)
        
        print("\n" + "="*30)
        print(f" 최종 예측 결과: {prediction}")
        print("="*30)
        
    except FileNotFoundError as e:
        print(f"\n[오류] 파일