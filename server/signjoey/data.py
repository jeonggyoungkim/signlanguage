# coding: utf-8
"""
데이터 로딩 및 처리를 담당하는 모듈입니다.
주로 설정 파일에 정의된 경로와 옵션에 따라 학습, 검증, 테스트 데이터를 로드하고,
어휘 사전을 구축하며, 데이터 반복자(iterator)를 생성하는 기능을 포함합니다.
"""
import os
import sys
import random
import json # JSON 형식의 데이터 분할 정보 파일을 읽기 위해 사용
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import socket # 현재 코드에서는 사용되지 않는 것으로 보임 (제거 고려 가능)
from signjoey.dataset import SignRecognitionDataset # SignRecognitionDataset으로 클래스 이름 사용
from signjoey.vocabulary import (
    build_vocab, # 어휘 사전 구축 함수
    Vocabulary, # 기본 어휘 사전 클래스
    UNK_TOKEN, # 특수 토큰들
    EOS_TOKEN,
    BOS_TOKEN,
    PAD_TOKEN,
)
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
import logging

logger = logging.getLogger(__name__)

# === 기존 MultiClassKeypointDataset, KeypointDataset 등은 주석 처리 ===
# class MultiClassKeypointDataset(torch.utils.data.Dataset):
#     ...
# class KeypointDataset(Dataset):
#     ...

# collate_fn은 dict 기반 샘플을 처리하므로 그대로 사용

def load_data(cfg, device: str = None):
    """
    최신 SignRecognitionDataset을 사용하여 train/test 데이터셋과 DataLoader를 생성합니다.
    `device`가 지정되지 않으면 사용 가능한 경우 GPU를 자동으로 사용합니다.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Device not specified, automatically selected: {device}")

    data_cfg = cfg["data"]
    data_path = data_cfg["data_path"]
    feature_size = data_cfg["feature_size"]
    annotation_file = data_cfg.get("annotation_file", None)
    trg_vocab_file = data_cfg.get("trg_vocab_file", None)
    auto_discover = data_cfg.get("auto_discover", False)
    auto_generate_vocab = data_cfg.get("auto_generate_vocab", False)
    train_files = data_cfg.get("train_files", 15)
    test_files = data_cfg.get("test_files", 3)

    # Auto-discover 모드: 폴더 기반으로 클래스를 자동 발견
    if auto_discover:
        print(f"Auto-discover mode: scanning {data_path} for class directories...")
        
        # 클래스 디렉토리 자동 발견
        class_dirs = []
        if os.path.exists(data_path):
            for item in os.listdir(data_path):
                item_path = os.path.join(data_path, item)
                if os.path.isdir(item_path):
                    class_dirs.append(item)
        
        if not class_dirs:
            raise ValueError(f"No class directories found in {data_path}")
        
        class_dirs.sort()  # 일관된 순서를 위해 정렬
        print(f"Found classes: {class_dirs}")
        
        # 자동 vocabulary 생성
        if auto_generate_vocab:
            print("Auto-generating vocabulary...")
            trg_vocab = Vocabulary()
            
            # 실제 클래스들만 vocabulary에 추가 (특수 토큰 제외)
            trg_vocab._from_list(class_dirs)
            
            # vocabulary 파일 저장 (선택사항)
            if trg_vocab_file:
                os.makedirs(os.path.dirname(trg_vocab_file), exist_ok=True)
                with open(trg_vocab_file, 'w', encoding='utf-8') as f:
                    for token in trg_vocab.itos:
                        f.write(f"{token}\n")
                print(f"Vocabulary saved to {trg_vocab_file}")
        else:
            # 타겟 어휘 사전 구축
            if not trg_vocab_file or not os.path.exists(trg_vocab_file):
                raise ValueError(f"Target vocabulary file 'data.vocab_file' ({trg_vocab_file}) is not specified or does not exist.")
            
            # 기존 vocabulary 파일 로드 후 특수 토큰 필터링
            temp_vocab = Vocabulary()
            temp_vocab._from_file(trg_vocab_file)
            
            # 특수 토큰들을 제외한 실제 클래스들만 필터링
            special_tokens = {UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
            filtered_tokens = [token for token in temp_vocab.itos if token not in special_tokens]
            
            # 새로운 vocabulary 생성 (특수 토큰 없이)
            trg_vocab = Vocabulary()
            trg_vocab._from_list(filtered_tokens)
        
        # 각 클래스별로 파일들을 train/test로 분할
        train_split = []
        test_split = []
        
        for class_name in class_dirs:
            class_path = os.path.join(data_path, class_name)
            csv_files = [f for f in os.listdir(class_path) if f.endswith('.csv')]
            csv_files.sort()  # 일관된 순서
            
            if len(csv_files) < train_files + test_files:
                print(f"Warning: Class '{class_name}' has only {len(csv_files)} files, "
                      f"but {train_files + test_files} files are needed. Using all available files.")
                actual_train_files = max(1, len(csv_files) * train_files // (train_files + test_files))
                actual_test_files = len(csv_files) - actual_train_files
            else:
                actual_train_files = train_files
                actual_test_files = test_files
            
            # 파일들을 섞어서 train/test로 분할
            random.shuffle(csv_files)
            train_files_for_class = csv_files[:actual_train_files]
            test_files_for_class = csv_files[actual_train_files:actual_train_files + actual_test_files]
            
            # 상대 경로로 split에 추가 (SignRecognitionDataset이 기대하는 형식)
            for train_file in train_files_for_class:
                train_split.append(os.path.join(class_name, train_file))
            
            for test_file in test_files_for_class:
                test_split.append(os.path.join(class_name, test_file))
            
            print(f"Class '{class_name}': {len(train_files_for_class)} train files, {len(test_files_for_class)} test files")
    
    else:
        # 기존 annotation_file 기반 로딩
        if annotation_file is not None and os.path.exists(annotation_file):
            with open(annotation_file, 'r', encoding='utf-8') as f:
                splits = json.load(f)
            train_split = splits.get("train", [])
            test_split = splits.get("test", [])
        else:
            raise ValueError("annotation_file이 없거나 경로가 잘못되었습니다.")

        # 타겟 어휘 사전 구축
        if not trg_vocab_file or not os.path.exists(trg_vocab_file):
            raise ValueError(f"Target vocabulary file 'data.vocab_file' ({trg_vocab_file}) is not specified or does not exist.")
        
        # 기존 vocabulary 파일 로드 후 특수 토큰 필터링
        temp_vocab = Vocabulary()
        temp_vocab._from_file(trg_vocab_file)
        
        # 특수 토큰들을 제외한 실제 클래스들만 필터링
        special_tokens = {UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN}
        filtered_tokens = [token for token in temp_vocab.itos if token not in special_tokens]
        
        # 새로운 vocabulary 생성 (특수 토큰 없이)
        trg_vocab = Vocabulary()
        trg_vocab._from_list(filtered_tokens)

    print(f"Total training samples: {len(train_split)}")
    print(f"Total test samples: {len(test_split)}")
    print(f"Vocabulary size: {len(trg_vocab)}")

    # 최신 SignRecognitionDataset 사용
    print(f"Attempting to load dataset to device: {device}")
    train_data = SignRecognitionDataset(data_cfg, data_path, train_split, trg_vocab, training=True, device=device)
    test_data = SignRecognitionDataset(data_cfg, data_path, test_split, trg_vocab, training=False, device=device)

    # DataLoader 생성
    # GPU 사용 시 num_workers=0, pin_memory=False 권장
    num_workers_setting = 0 if device != "cpu" else data_cfg.get("num_workers", 0) # CPU면 설정값, GPU면 0
    pin_memory_setting = False if device != "cpu" else data_cfg.get("pin_memory", False) # CPU면 설정값, GPU면 False

    train_iter = make_data_iter(
        train_data,
        batch_size=data_cfg.get("batch_size", 8),
        train=True,
        shuffle=True,
        num_workers=num_workers_setting,
        pin_memory=pin_memory_setting
    )
    test_iter = make_data_iter(
        test_data,
        batch_size=data_cfg.get("batch_size", 8),
        train=False,
        shuffle=False,
        num_workers=num_workers_setting,
        pin_memory=pin_memory_setting
    )

    vocab_info = {
        "sgn_vocab": None,
        "txt_vocab": trg_vocab,
        "feature_size": feature_size,
        "class_names": trg_vocab.itos,  # 이미 특수 토큰이 제거된 상태
        "num_classes": len(trg_vocab)  # 이미 특수 토큰이 제거된 상태
    }
    return train_iter, test_iter, vocab_info

def collate_fn_multi_class(batch):
    """Collate function for DataLoader with multiple classes"""
    sgn_sequences = [item['sgn'] for item in batch]
    trg_labels = torch.tensor([item['trg_label'] for item in batch], dtype=torch.long)
    
    # sgn_sequences가 GPU 텐서인지 확인하고 같은 디바이스에서 처리
    if sgn_sequences and sgn_sequences[0].is_cuda:
        # GPU 텐서들이 들어오면 최적화된 처리
        device = sgn_sequences[0].device
        
        # GPU에서 직접 trg_labels 생성
        trg_labels = torch.tensor([item['trg_label'] for item in batch], 
                                dtype=torch.long, device=device)
        
        # GPU에서 pad_sequence 실행 (같은 디바이스에서 실행)
        with torch.cuda.device(device):
            sgn_padded = pad_sequence(sgn_sequences, batch_first=True, padding_value=0.0)
            sgn_lengths = torch.tensor([len(seq) for seq in sgn_sequences], device=device)
    else:
        # CPU 텐서 처리
        sgn_padded = pad_sequence(sgn_sequences, batch_first=True, padding_value=0.0)
        sgn_lengths = torch.tensor([len(seq) for seq in sgn_sequences])
    
    return {
        'sgn': sgn_padded,
        'txt': trg_labels,
        'sgn_lengths': sgn_lengths
    }

def make_data_iter_multi_class(dataset, batch_size=8, shuffle=True, num_workers=0):
    """Create data iterator for multi-class dataset"""
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn_multi_class,
        num_workers=num_workers
    )

def make_data_iter(
    dataset: Dataset, 
    batch_size: int, 
    train: bool = False, 
    shuffle: bool = False,
    num_workers: int = 0, 
    pin_memory: bool = False,
    prefetch_factor: int = 2  # num_workers > 0 일 때 PyTorch 기본값
) -> DataLoader:
    """
    주어진 데이터셋에 대한 데이터 반복자(iterator)를 생성합니다.
    기본 collate_fn으로 collate_fn_multi_class를 사용합니다.
    """
    # DataLoader는 num_workers=0일 경우 prefetch_factor를 무시합니다.
    # 명시적으로 None을 전달하여 PyTorch의 기본 동작을 따르도록 할 수 있습니다.
    # 또는 PyTorch 1.9+ 에서는 num_workers=0 이어도 prefetch_factor=None (또는 생략)으로 두면 경고 없이 잘 동작합니다.
    effective_prefetch_factor = prefetch_factor if num_workers > 0 else None

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle if train else False, # 학습 시에만 셔플
        collate_fn=collate_fn_multi_class,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=effective_prefetch_factor
    )
