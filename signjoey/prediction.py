#!/usr/bin/env python
# coding: utf-8 # 파일 인코딩을 UTF-8로 명시 (주석 번역에 중요)
import torch

torch.backends.cudnn.deterministic = True # cuDNN의 결정론적 연산 사용 설정 (재현성을 위해)

import logging
import numpy as np
import pickle as pickle # Python 객체 직렬화를 위한 pickle 모듈 (사용되지 않는다면 제거 가능)
import time # 시간 관련 함수 사용 (예: 처리 시간 측정)
import torch.nn as nn
import os # 운영체제 관련 기능 (파일 경로, 디렉토리 생성 등)
import json # JSON 파일 처리

from typing import List
from torch.utils.data import Dataset # torch.utils.data의 Dataset 사용
# from signjoey.loss import XentLoss # 번역 모델용 손실 함수 (현재 사용 안 함)
from signjoey.helpers import (
    load_config, # 설정 파일 로드 함수
    get_latest_checkpoint, # 가장 최근 체크포인트 가져오는 함수
    load_checkpoint, # 체크포인트 로드 함수
)
# from signjoey.metrics import wer_list # 단어 오류율(WER) 계산 함수 <- 주석 처리
from signjoey.model import build_model, SignModel # 모델 빌드 함수 및 모델 클래스
from signjoey.batch import Batch # 배치 데이터 처리를 위한 클래스
from signjoey.data import load_data, make_data_iter # 데이터 로드 및 반복자 생성 함수
from signjoey.vocabulary import PAD_TOKEN, SIL_TOKEN # 특수 토큰 (패딩, 침묵/공백)

# pylint 경고 비활성화: 인자 너무 많음, 지역 변수 너무 많음, 멤버 없음 (개발 편의를 위함)
# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(
    model: SignModel, # 평가할 모델 객체
    data: Dataset, # 평가용 데이터셋 (torch.utils.data.Dataset)
    batch_size: int, # 배치 크기
    use_cuda: bool, # CUDA 사용 여부
    sgn_dim: int, # 수어 키포인트 특성(feature)의 차원 수
    do_recognition: bool, # 수어 인식(gloss 예측) 수행 여부 (항상 True여야 함)
    recognition_loss_function: torch.nn.Module, # 인식 손실 함수 (예: CTCLoss)
    recognition_loss_weight: float, # 인식 손실 가중치
    level: str, # Gloss 처리 수준 ('word', 'char' 등)
    recognition_beam_size: int = 1, # CTC 디코딩 시 빔 크기 (1이면 greedy decoding)
    batch_type: str = "sentence", # 배치 유형 ('sentence' 또는 'token')
    frame_subsampling_ratio: int = None, # 프레임 서브샘플링 비율 (데이터 증강용, 사용 안 할 경우 None)
) -> dict: # 결과 딕셔너리 반환
    """
    주어진 데이터로 수어 인식 모델을 검증합니다.
    인식 손실과 단어 오류율(WER)을 계산합니다.

    :param model: 모델 모듈
    :param data: 검증용 데이터셋
    :param batch_size: 검증 배치 크기
    :param use_cuda: True이면 CUDA 사용
    :param sgn_dim: 수어 프레임의 특성 차원
    :param do_recognition: Gloss 예측 플래그 (True여야 함)
    :param recognition_loss_function: 인식 손실 함수 (CTC)
    :param recognition_loss_weight: CTC 손실 가중치
    :param level: Gloss 분절 수준 ('word', 'char' 등)
    :param recognition_beam_size: CTC 디코딩 시 빔 크기
    :param batch_type: 검증 배치 유형 ('sentence' 또는 'token')
    :param frame_subsampling_ratio: 프레임 서브샘플링 비율
    :return: 검증 점수와 손실을 담은 딕셔너리
    """
    valid_iter = make_data_iter( # 검증용 데이터 반복자 생성
        dataset=data,
        batch_size=batch_size,
        batch_type=batch_type,
        shuffle=False, # 검증 시에는 섞지 않음
        train=False, # 학습 모드 아님
    )

    model.eval() # 모델을 평가 모드로 설정 (dropout 등 비활성화)
    with torch.no_grad(): # 그래디언트 계산 비활성화 (메모리 절약 및 속도 향상)
        all_gls_outputs_indices = [] # 모든 Gloss 예측 결과(인덱스 또는 문자열)를 저장할 리스트
        total_recognition_loss = 0 # 총 인식 손실
        total_num_gls_tokens = 0 # 실제 Gloss 시퀀스의 총 토큰 수 (Loss 정규화 등에 사용될 수 있으나 현재는 사용 안 함)
        total_num_seqs = 0 # 총 시퀀스 수 (배치 수)
        
        for valid_batch in iter(valid_iter):
            # 현재 배치 데이터를 Batch 객체로 변환
            batch = Batch(
                is_train=False, # 학습 중 아님
                torch_batch=valid_batch,
                sgn_dim=sgn_dim,
                use_cuda=use_cuda,
                frame_subsampling_ratio=frame_subsampling_ratio,
            )
            sort_reverse_index = batch.sort_by_sgn_lengths() # 수어 시퀀스 길이에 따라 정렬하고, 원래 순서로 되돌릴 인덱스 저장

            # 현재 배치에 대한 인식 손실 계산
            batch_recognition_loss = model.get_loss_for_batch(
                batch=batch,
                recognition_loss_function=recognition_loss_function,
                recognition_loss_weight=recognition_loss_weight,
            )
            
            if batch_recognition_loss is not None: # 손실이 유효한 경우
                 total_recognition_loss += batch_recognition_loss.item() # .item()으로 Python 숫자로 변환하여 누적
            if batch.num_gls_tokens is not None:
                 total_num_gls_tokens += batch.num_gls_tokens # 현재 배치 내 Gloss 토큰 수 누적
            total_num_seqs += batch.num_seqs # 시퀀스(샘플) 수 누적

            # 현재 배치에 대한 Gloss 예측 실행 (디코딩)
            batch_gls_predictions = model.run_batch(
                batch=batch,
                recognition_beam_size=recognition_beam_size,
            )
            
            if batch_gls_predictions is not None: # 예측 결과가 있는 경우
                # 예측 결과를 원래 순서대로 정렬하여 저장
                # batch_gls_predictions는 디코딩된 문자열 리스트일 것으로 예상
                 all_gls_outputs_indices.extend(
                    [batch_gls_predictions[sri] for sri in sort_reverse_index]
                )

        results = {} # 결과 저장용 딕셔너리
        if do_recognition: # 인식(gloss 예측)을 수행한 경우
            # 가설(예측) 시퀀스 수와 실제 데이터 예제 수가 같은지 확인
            assert len(all_gls_outputs_indices) == len(data.examples), \
                f"가설 수 ({len(all_gls_outputs_indices)}) != 예제 수 ({len(data.examples)})"

            # 평균 인식 손실 계산
            if recognition_loss_function is not None and recognition_loss_weight > 0 and total_num_seqs > 0:
                # CTCLoss의 reduction 설정에 따라 정규화 방식이 달라질 수 있음.
                # 현재는 배치 전체의 손실 합을 시퀀스 수로 나누어 평균 손실을 계산한다고 가정.
                valid_recognition_loss = total_recognition_loss / total_num_seqs if total_num_seqs > 0 else -1
            else:
                valid_recognition_loss = -1 # 손실 계산 조건이 안 맞으면 -1

            # all_gls_outputs_indices는 run_batch에서 반환된 예측된 gloss 문자열 리스트임
            gls_hyp = all_gls_outputs_indices 

            # 참조(정답) Gloss 시퀀스 준비
            # data.gls는 토큰 리스트 또는 문자열 리스트를 포함할 수 있음
            gls_ref_sequences = []
            for ex_gls in data.gls: # 데이터셋의 각 예제에서 정답 gloss를 가져옴
                if isinstance(ex_gls, list):
                    gls_ref_sequences.append(" ".join(ex_gls)) # 토큰 리스트면 공백으로 합쳐 문자열로 만듦
                elif isinstance(ex_gls, str):
                    gls_ref_sequences.append(ex_gls) # 이미 문자열이면 그대로 사용
                else:
                    gls_ref_sequences.append("") # 알 수 없는 타입이면 빈 문자열로 처리 (오류 방지)
            
            # 가설(예측) Gloss 시퀀스도 문자열 형태로 통일 (run_batch가 문자열 리스트를 반환해야 함)
            gls_hyp_sequences = [" ".join(h) if isinstance(h, list) else str(h) for h in gls_hyp]

            assert len(gls_ref_sequences) == len(gls_hyp_sequences) # 정답과 예측 시퀀스 수가 같은지 확인

            # WER 계산
            # gls_wer_score = wer_list(hypotheses=gls_hyp_sequences, references=gls_ref_sequences) <- 주석 처리
            # valid_scores = {"wer": gls_wer_score["wer"], "wer_scores": gls_wer_score} # WER 점수 저장 <- 주석 처리
            valid_scores = {} # 임시로 빈 딕셔너리로 설정 또는 분류 메트릭으로 대체 필요
            
            results["valid_recognition_loss"] = valid_recognition_loss # 검증 인식 손실
            results["decoded_gls"] = gls_hyp_sequences # 디코딩된(예측된) gloss 시퀀스 (처리된 형태)
            results["gls_ref"] = gls_ref_sequences # 참조(정답) gloss 시퀀스
            results["gls_hyp"] = gls_hyp_sequences # 가설(예측) gloss 시퀀스 (decoded_gls와 동일할 수 있으나 일관성을 위해 저장)
        
        results["valid_scores"] = valid_scores # 최종 검증 점수 (WER 등)

    return results


# pylint 경고 비활성화: 너무 많은 인자 로깅 (개발 편의)
# pylint: disable-msg=logging-too-many-args
def test(
    cfg_file: str, # 설정 파일 경로
    ckpt: str, # 테스트할 모델 체크포인트 경로
    output_path: str = None, # 결과를 저장할 디렉토리 경로 (예: 가설 파일)
    logger: logging.Logger = None # 로깅을 위한 로거 객체
) -> None:
    """
    수어 인식(Sign Language Recognition)을 위한 메인 테스트 함수입니다.
    체크포인트로부터 모델을 로드하고, gloss 예측을 생성하며, 이를 저장하고 WER을 계산합니다.

    :param cfg_file: 설정 파일 경로
    :param ckpt: 로드할 체크포인트 경로
    :param output_path: 결과 저장 디렉토리 경로
    :param logger: 로깅 객체 (지정 안 하면 새로 생성)
    """

    if logger is None: # 로거가 없으면 새로 생성
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            FORMAT = "%(asctime)-15s - %(message)s"
            logging.basicConfig(format=FORMAT)
            logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file) # 설정 파일 로드
    model_cfg = cfg["model"] # 모델 설정
    data_cfg = cfg["data"] # 데이터 설정
    test_cfg = cfg.get("testing", {}) # 테스트 관련 설정 (없으면 빈 딕셔너리)

    # 테스트 데이터가 설정 파일이나 annotation_file에 명시되어 있는지 확인
    if "test" not in data_cfg.keys() and "test" not in data_cfg.get("annotation_file", {}):
        try:
            # annotation_file (JSON) 내에 test 스플릿이 있는지 확인
            with open(data_cfg['annotation_file'], 'r', encoding='utf-8') as f:
                splits_in_file = json.load(f)
            if "test" not in splits_in_file or not splits_in_file["test"]:
                 raise ValueError("테스트 데이터가 설정 파일이나 annotation_file에 명시되어야 합니다.")
        except (FileNotFoundError, KeyError, json.JSONDecodeError):
             raise ValueError("테스트 데이터가 설정 파일이나 annotation_file에 명시되어야 하며, annotation_file은 유효한 JSON이어야 합니다.")

    if ckpt is None: # 체크포인트 경로가 없으면, 모델 디렉토리에서 가장 최근 것 사용
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError(
                f"{model_dir} 디렉토리에서 체크포인트를 찾을 수 없습니다."
            )
    logger.info("%s 에서 체크포인트를 로드합니다.", ckpt)

    # 테스트 시 사용할 배치 크기 및 유형, CUDA 사용 여부, gloss 레벨, 특성 차원 등 설정값 가져오기
    batch_size = test_cfg.get("batch_size", cfg["training"]["batch_size"])
    batch_type = test_cfg.get("batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    level = data_cfg.get("level", "word") # Gloss 처리 수준
    sgn_dim = data_cfg["feature_size"]
    
    # 테스트 시 사용할 인식 빔 크기 (리스트 형태 가능)
    recognition_beam_sizes = test_cfg.get("recognition_beam_sizes", [1])
    if not isinstance(recognition_beam_sizes, list): recognition_beam_sizes = [recognition_beam_sizes]

    # 데이터 로드: 현재는 test_data와 gls_vocab만 사용
    _, _, test_data, gls_vocab, _ = load_data(data_cfg=data_cfg)

    if test_data is None or len(test_data.examples) == 0:
        logger.error("테스트 데이터가 비어있습니다. 테스트를 건너뛰었습니다.")
        return

    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda) # 체크포인트 로드

    # 모델 빌드 및 로드된 가중치 적용
    model = build_model(
        cfg=cfg, # 전체 설정
        gls_vocab=gls_vocab, # Gloss 어휘 사전
        sgn_dim=sgn_dim, # 수어 특성 차원
        do_recognition=True, # 인식 모드 강제
    )
    model.load_state_dict(model_checkpoint["model_state"]) # 모델 상태(가중치) 로드

    if use_cuda:
        model.cuda() # 모델을 CUDA 장치로 이동

    model.eval() # 모델을 평가 모드로 설정
    
    all_final_refs = [] # 모든 참조(정답) Gloss 시퀀스를 저장할 리스트
    # 테스트 데이터에서 참조 Gloss를 미리 준비 (빔 크기별로 반복하지 않도록)
    for ex_gls in test_data.gls:
        if isinstance(ex_gls, list):
            all_final_refs.append(" ".join(ex_gls))
        elif isinstance(ex_gls, str):
            all_final_refs.append(ex_gls)
        else:
            all_final_refs.append("")

    # 설정된 각 빔 크기에 대해 테스트 실행
    for beam_size in recognition_beam_sizes:
        logger.info("=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=")
        logger.info("인식 빔 크기 %d 로 테스트 시작", beam_size)
        
        test_iter = make_data_iter( # 테스트 데이터 반복자 생성
            dataset=test_data,
            batch_size=batch_size,
            batch_type=batch_type,
            shuffle=False,
            train=False,
        )
        
        all_hypotheses_for_beam = [] # 현재 빔 크기에 대한 모든 가설(예측) 저장 리스트
        with torch.no_grad(): # 그래디언트 계산 비활성화
            for batch in iter(test_iter):
                current_batch = Batch( # 현재 배치 데이터를 Batch 객체로 변환
                    is_train=False,
                    torch_batch=batch,
                    sgn_dim=sgn_dim,
                    use_cuda=use_cuda,
                    # 테스트 시에는 기본적으로 프레임 서브샘플링/마스킹 사용 안 함
                )
                sort_reverse_index = current_batch.sort_by_sgn_lengths() # 길이로 정렬
                
                # 현재 배치에 대한 Gloss 예측 실행
                batch_gls_predictions = model.run_batch(
                    batch=current_batch,
                    recognition_beam_size=beam_size,
                )
                
                if batch_gls_predictions is not None:
                    # batch_gls_predictions는 run_batch에서 반환된 문자열 리스트여야 함
                    all_hypotheses_for_beam.extend(
                        [batch_gls_predictions[sri] for sri in sort_reverse_index] # 원래 순서로 복원하여 추가
                    )
        
        assert len(all_hypotheses_for_beam) == len(test_data.examples) # 예측 수와 예제 수 일치 확인
        
        # 최종 가설(예측) 시퀀스들을 문자열로 통일
        final_hypotheses = [" ".join(h) if isinstance(h, list) else str(h) for h in all_hypotheses_for_beam]

        if output_path is not None: # 출력 경로가 지정된 경우
            if not os.path.exists(output_path):
                os.makedirs(output_path) # 출력 디렉토리 생성
            # 가설(예측) 파일 저장
            output_file_hyp = os.path.join(output_path, f"test.beam{beam_size}.hyp.gls")
            _write_to_file(output_file_hyp, test_data.sequence, final_hypotheses) # sequence는 파일명 등의 ID 정보
            logger.info("가설 저장 위치: %s", output_file_hyp)
            
            if beam_size == recognition_beam_sizes[0]: # 첫 번째 빔 크기일 때만 참조 파일 저장 (중복 방지)
                output_file_ref = os.path.join(output_path, "test.ref.gls")
                _write_to_file(output_file_ref, test_data.sequence, all_final_refs)
                logger.info("참조 저장 위치: %s", output_file_ref)

        # WER 계산
        # current_wer_score = wer_list(hypotheses=final_hypotheses, references=all_final_refs) <- 주석 처리
        logger.info(
            "빔 크기 %d 에 대한 평가 결과: WER: %.2f%% (DEL: %.2f%%, INS: %.2f%%, SUB: %.2f%%)",
            beam_size,
            # current_wer_score["wer"],
            # current_wer_score["del_rate"],
            # current_wer_score["ins_rate"],
            # current_wer_score["sub_rate"],
        )
        logger.info("=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=")

# 가설/참조 시퀀스를 파일에 저장하는 헬퍼 함수
def _write_to_file(file_path: str, sequence_ids: List[str], sequences: List[str]):
    with open(file_path, "w", encoding='utf-8') as f:
        for seq_id, seq in zip(sequence_ids, sequences):
            f.write(f"{seq_id}|{seq}\n") # 시퀀스 ID와 시퀀스를 | 구분자로 저장

# 이 파일이 직접 실행될 경우 (예: `python signjoey/prediction.py ...`)
# 아래의 main 실행 블록은 training.py나 별도의 테스트 스크립트에서 test() 함수를 호출하는 방식으로 대체되었으므로 주석 처리.
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser("Joey-NMT Tester")
#     parser.add_argument("config", type=str, help="Training configuration file (yaml).")
#     parser.add_argument("--ckpt", type=str, help="Checkpoint to test.", default=None)
#     parser.add_argument(
#         "--output_path", type=str, help="Path for saving hypotheses.", default=None
#     )
#     parser.add_argument(
#         "--gpu_id", type=str, default="0", help="gpu to run your job on"
#     )
#     args = parser.parse_args()
#     os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id # 사용할 GPU ID 설정
#     test(cfg_file=args.config, ckpt=args.ckpt, output_path=args.output_path)
