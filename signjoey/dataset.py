# coding: utf-8
"""
데이터 모듈 (최신 PyTorch Dataset 방식)
"""
from typing import List, Dict, Any, Optional
from tqdm import tqdm   
import numpy as np
import pickle
import gzip
import torch
import json
import glob
import pandas as pd
import os
from torch.utils.data import Dataset
from signjoey.vocabulary import Vocabulary
import logging
import re # 정규 표현식 모듈 임포트

def load_dataset_file(filename):
    """데이터셋 파일을 로드합니다."""
    with gzip.open(filename, "rb") as f:
        loaded_object = pickle.load(f)
        return loaded_object

class SignRecognitionDataset(Dataset):
    """수어 인식을 위한 데이터셋 (CSV 키포인트 사용, 최신 방식)"""
    def __init__(
        self,
        cfg: Dict[str, Any],
        path: str,
        split: List[str],
        trg_vocab: Vocabulary,
        training: bool = False,
        device: str = "cpu",  # 디바이스 파라미터 추가
    ):
        self.trg_vocab = trg_vocab
        self.unk_index = trg_vocab.stoi.get(trg_vocab.unk_token, 0)
        self.device = torch.device(device) if isinstance(device, str) else device
        data_root_path = cfg.get("data_path", "./data")
        self.samples = []
        self.logger = logging.getLogger(__name__) # 로거 가져오기

        # 디바이스 정보 로깅
        self.logger.info(f"Creating dataset with device: {self.device}")

        loaded_count = 0
        unk_count = 0
        
        # 데이터 로딩 최적화: 병렬 처리와 배치 로딩
        print(f"Loading {len(split)} files to {self.device}...")
        
        # 파일들을 배치로 나누어 처리 (메모리 사용량 제어)
        batch_size = 50  # 한 번에 처리할 파일 수
        
        for batch_start in range(0, len(split), batch_size):
            batch_end = min(batch_start + batch_size, len(split))
            batch_files = split[batch_start:batch_end]
            
            print(f"Processing batch {batch_start//batch_size + 1}/{(len(split) + batch_size - 1)//batch_size}")
            
            batch_samples = []
            
            for filename_with_ext in batch_files:
                file_path = os.path.join(data_root_path, filename_with_ext)
                
                self.logger.debug(f"Processing file: {file_path}")

                # 레이블 추출 시에는 전체 상대 경로를 전달하여 폴더명(클래스명)을 사용하도록 함
                label_str_extracted = self._extract_label_from_filename(filename_with_ext) # Pass the full relative path
                self.logger.debug(f"Extracted label_str: '{label_str_extracted}' from filename_with_ext: '{filename_with_ext}'")

                if label_str_extracted is None:
                    unk_count += 1
                    continue

                label_str_to_check = label_str_extracted.strip()

                # 어휘 사전에 레이블이 있는지 확인
                if label_str_to_check not in self.trg_vocab.stoi:
                    self.logger.warning(f"Label '{label_str_to_check}' (original: '{label_str_extracted}') not in trg_vocab. Known vocab: {list(self.trg_vocab.stoi.keys())}")
                    label_idx = self.unk_index
                else:
                    label_idx = self.trg_vocab.stoi[label_str_to_check]
                
                self.logger.debug(f"Label_str_to_check: '{label_str_to_check}', Mapped label_idx: {label_idx}")

                try:
                    # CSV 읽기 최적화: 필요한 컬럼만 읽기
                    df = pd.read_csv(file_path, low_memory=False)
                    keypoint_cols = [col for col in df.columns if col.startswith("keypoint_")]
                    if not keypoint_cols:
                        continue
                    
                    # 키포인트 데이터만 추출
                    keypoint_data = df[keypoint_cols].values.astype(np.float32)
                    expected_feature_size = cfg.get("feature_size")
                    if keypoint_data.shape[1] != expected_feature_size:
                        continue
                    
                    # 배치에 추가 (아직 텐서로 변환하지 않음)
                    sequence_id = filename_with_ext
                    signer_id = cfg.get("default_signer", "unknown_signer")
                    gls_data = ""
                    if "gloss" in df.columns:
                        gloss_series = df["gloss"].dropna()
                        if not gloss_series.empty:
                            gls_data = gloss_series.iloc[0]
                            if isinstance(gls_data, (list, np.ndarray)):
                                gls_data = " ".join(gls_data)
                            gls_data = str(gls_data)
                    
                    batch_samples.append({
                        "sequence": sequence_id,
                        "signer": signer_id,
                        "keypoint_data": keypoint_data,  # numpy 배열로 보관
                        "gls": gls_data,
                        "trg_label": label_idx,
                    })
                    loaded_count += 1
                except Exception as e:
                    self.logger.debug(f"Error loading {file_path}: {e}")
                    continue
            
            # 배치 단위로 GPU 텐서 생성
            if batch_samples and self.device.type == "cuda":
                print(f"Converting batch of {len(batch_samples)} samples to GPU tensors...")
                
                # CUDA 스트림을 사용하여 비동기 전송 최적화
                with torch.cuda.device(self.device):
                    for sample in batch_samples:
                        # GPU에서 직접 텐서 생성 (비동기 전송)
                        sgn_features = torch.from_numpy(sample["keypoint_data"]).cuda(device=self.device, non_blocking=True)
                        sample["sgn"] = sgn_features
                        del sample["keypoint_data"]  # numpy 배열 삭제로 메모리 절약
                        
                # GPU 메모리 정리
                torch.cuda.empty_cache()
            else:
                # CPU 텐서 생성
                for sample in batch_samples:
                    sgn_features = torch.from_numpy(sample["keypoint_data"])
                    sample["sgn"] = sgn_features
                    del sample["keypoint_data"]
            
            # 샘플들을 메인 리스트에 추가
            self.samples.extend(batch_samples)
            
            # GPU 메모리 상태 출력
            if self.device.type == "cuda":
                current_memory = torch.cuda.memory_allocated(0) / 1024**3
                print(f"Current GPU memory: {current_memory:.3f} GB")

        if unk_count > 0:
            self.logger.info(
                f"{unk_count}개의 샘플에 대해 파일명에서 추출한 레이블이 타겟 어휘에 없어 <unk>로 처리되었습니다."
            )

        if not self.samples:
            print(f"경고: 분할에 대한 예제가 로드되지 않았습니다. 데이터 경로와 파일 내용을 확인하세요. 데이터 루트: {data_root_path}")
        
        print(f"Dataset loading complete: {len(self.samples)} samples loaded to {self.device}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def _extract_label_from_filename(self, filename_with_ext: str) -> Optional[str]:
        """
        파일 경로에서 클래스 레이블을 추출합니다.
        Auto-discover 모드에서는 폴더명이 클래스 이름입니다.
        예: "급하다/NIA_SL_WORD0026_REAL01_F_keypoints.csv" -> "급하다"
        """
        # filename_with_ext가 "클래스명/파일명.csv" 형태인지 확인
        if '/' in filename_with_ext or '\\' in filename_with_ext:
            # 경로에서 첫 번째 디렉토리 이름을 클래스 이름으로 사용
            class_name = os.path.dirname(filename_with_ext)
            # Windows 경로 구분자도 처리
            class_name = class_name.replace('\\', '/').split('/')[0]
            return class_name
        
        # 레거시: 파일명에서 'WORDXXXX' 형식의 레이블 코드를 추출 (이전 방식)
        match = re.search(r'(WORD\d{4})', filename_with_ext)
        if match:
            return match.group(1)
        
        self.logger.warning(f"Could not extract class label from filename: {filename_with_ext}")
        return None