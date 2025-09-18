# coding: utf-8
import math
import random
import torch
import numpy as np


class Batch:
    """학습 중 마스크와 함께 데이터 배치를 보유하기 위한 객체입니다.
    입력은 torchtext 반복자로부터의 배치입니다.
    이 버전은 수어 단어 분류를 위해 수정되었습니다.
    """

    def __init__(
        self,
        torch_batch, # DataLoader로부터의 배치 객체
        sgn_dim: int, # 수어 특성(키포인트)의 차원
        is_train: bool = False, # 이 배치가 학습용인지 여부 (데이터 증강에 영향)
        use_cuda: bool = False, # 텐서를 GPU로 옮길지 여부
        frame_subsampling_ratio: int = None, # 프레임 서브샘플링 비율
        random_frame_subsampling: bool = None, # 프레임 서브샘플링 시작점을 랜덤화할지 여부
        random_frame_masking_ratio: float = None, # 랜덤 프레임 마스킹 비율
    ):
        """
        torch 배치로부터 단어 분류를 위한 새 배치를 생성합니다.
        이 배치는 수어(sgn) 데이터, 길이, 마스크 및 타겟 단어 레이블(trg_label)을 처리합니다.

        :param torch_batch: DataLoader의 배치 객체
        :param sgn_dim: 수어 특성의 차원.
        :param is_train: 이 배치가 학습용인지 여부 (증강에 영향).
        :param use_cuda: 텐서를 GPU로 옮길지 여부.
        :param frame_subsampling_ratio: 프레임 서브샘플링 비율.
        :param random_frame_subsampling: 프레임 서브샘플링 시작을 랜덤화할지 여부.
        :param random_frame_masking_ratio: 랜덤 프레임 마스킹 비율.
        """

        # 시퀀스 정보 (메타데이터)
        self.sequence = torch_batch.sequence # 원본 시퀀스 식별자 (예: 파일명)
        self.signer = torch_batch.signer # 화자(signer) 정보
        
        # 수어 데이터 (Sign Data)
        self.sgn, self.sgn_lengths = torch_batch.sgn # sgn: (N, T_sgn, F_sgn), sgn_lengths: (N,)
        self.sgn_dim = sgn_dim # 수어 특성 차원 저장

        # 프레임 증강 (Frame Augmentations): 서브샘플링 및 마스킹
        # 프레임 서브샘플링 (학습 시에만 적용되도록 is_train 조건 확인이 중요)
        if frame_subsampling_ratio and frame_subsampling_ratio > 1 and is_train:
            # 프레임 서브샘플링 적용 (원본 코드 로직 기반)
            # 이 로직은 더 효율적으로 만들거나 Dataset 클래스의 일부로 옮길 수 있습니다.
            # 주의: self.sgn과 self.sgn_lengths를 사실상 현재 위치에서 수정합니다.
            # 여기서는 안전과 명확성을 위해 새 텐서를 생성합니다.
            new_sgn_list = [] # 서브샘플링된 특성 시퀀스들을 저장할 리스트
            new_sgn_lengths_list = [] # 서브샘플링된 시퀀스들의 길이를 저장할 리스트
            max_len_after_subsample = 0 # 서브샘플링 후 최대 시퀀스 길이

            for idx, length in enumerate(self.sgn_lengths):
                features = self.sgn[idx, :length.long(), :].clone() # 현재 시퀀스의 실제 길이만큼만 복제
                if random_frame_subsampling: # 랜덤 시작점 사용 여부 (학습 시 권장)
                    init_frame = random.randint(0, (frame_subsampling_ratio - 1))
                else: # 고정된 중간 지점에서 시작
                    init_frame = math.floor((frame_subsampling_ratio - 1) / 2)
                
                subsampled_features = features[init_frame::frame_subsampling_ratio] # 서브샘플링
                new_sgn_list.append(subsampled_features)
                new_len = subsampled_features.shape[0]
                new_sgn_lengths_list.append(new_len)
                if new_len > max_len_after_subsample:
                    max_len_after_subsample = new_len
            
            # 서브샘플링된 시퀀스들을 새 최대 길이에 맞춰 패딩합니다.
            # sgn_dim은 특징 차원이라고 가정합니다.
            padded_sgn = torch.zeros(len(new_sgn_list), max_len_after_subsample, self.sgn_dim, device=self.sgn.device, dtype=self.sgn.dtype)
            for i, seq in enumerate(new_sgn_list):
                padded_sgn[i, :seq.shape[0], :] = seq
            
            self.sgn = padded_sgn # 서브샘플링 및 패딩된 sgn으로 업데이트
            self.sgn_lengths = torch.tensor(new_sgn_lengths_list, device=self.sgn_lengths.device, dtype=self.sgn_lengths.dtype)

        # 랜덤 프레임 마스킹 (학습 시에만 적용)
        if random_frame_masking_ratio and random_frame_masking_ratio > 0 and is_train:
            # 랜덤 프레임 마스킹 적용 (원본 코드 로직 기반)
            # 주의: 복제하지 않으면 self.sgn을 현재 위치에서 수정합니다.
            # num_mask_frames = (self.sgn_lengths * random_frame_masking_ratio).floor().long()
            # num_mask_frames에 대해 torch.floor를 사용한 후 long으로 변환합니다.
            # numpy를 사용하는 경우 sgn_lengths가 CPU에 있는지 확인하거나, torch 연산을 사용합니다.
            # 이 연산은 텐서 장치 관련 문제를 피하기 위해 신중하게 수행되어야 합니다.
            
            # 배치의 각 시퀀스에 대해 반복
            for idx in range(self.sgn.size(0)):
                current_length = self.sgn_lengths[idx].item() # Python int로 가져오기
                if current_length == 0: continue # 시퀀스가 비어있으면 건너뛰기

                num_frames_to_mask = int(math.floor(current_length * random_frame_masking_ratio))
                if num_frames_to_mask == 0: continue # 마스킹할 프레임이 없으면 건너뛰기

                # current_length까지의 프레임 인덱스를 랜덤하게 섞어 num_frames_to_mask개 선택
                # torch.randperm 사용 시 올바른 장치에 있는지 확인
                # 또는 current_length가 작고 CPU에 있다면 numpy.random.permutation 사용
                mask_frame_indices = torch.randperm(current_length, device=self.sgn.device)[:num_frames_to_mask]
                
                # self.sgn이 다른 곳에서 사용되거나 여러 증강이 연쇄되는 경우 수정 전에 복제하는 것이 더 안전합니다.
                # 단순성을 위해 여기서 이것이 유일한 수정이라면 현재 위치에서 수정합니다.
                self.sgn[idx, mask_frame_indices, :] = 1e-8 # 작은 값으로 설정하여 마스킹 (0으로 하면 패딩과 구분 안될 수 있음)

        # sgn 및 sgn_lengths에 대한 모든 잠재적 수정 후 sgn_mask 생성
        # 패딩이 0인 경우 마스크는 실제 0이 아닌 특징을 기반으로 해야 합니다.
        # 또는 패딩이 다른 것이거나 특징이 0일 수 있는 경우 sgn_lengths를 기반으로 해야 합니다.
        # 원본 마스크: (self.sgn != torch.zeros(sgn_dim))[..., 0].unsqueeze(1)
        # 이는 패딩이 정확히 torch.zeros(sgn_dim)이라고 가정합니다.
        # 길이를 기반으로 한 더 견고한 마스크:
        max_len = self.sgn.size(1) # 현재 배치의 최대 시퀀스 길이 (패딩 포함)
        self.sgn_mask = torch.arange(max_len, device=self.sgn.device)[None, :] < self.sgn_lengths[:, None]
        self.sgn_mask = self.sgn_mask.unsqueeze(1) # Shape: (N, 1, T_sgn) - 어텐션 등에 사용될 수 있는 형태

        # 타겟 단어 레이블 (Target word label)
        self.trg_label = None # (N,) 형태의 단일 정수 레이블 텐서
        if hasattr(torch_batch, "gls") and torch_batch.gls is not None:
            # gls_label_field에서 sequential=False로 설정했으므로, torch_batch.gls는 (N,) 형태의 텐서가 됨.
            # 만약 (N, 1) 형태로 온다면 squeeze 필요.
            self.trg_label = torch_batch.gls 
            if self.trg_label.ndim > 1: # (N, 1) 같은 경우 (N,)로 squeeze
                 self.trg_label = self.trg_label.squeeze(-1)

        self.use_cuda = use_cuda # CUDA 사용 여부 저장
        self.num_seqs = self.sgn.size(0) # 배치 내 시퀀스 수 (배치 크기)

        if use_cuda: # CUDA 사용이 True이면
            self._make_cuda() # 텐서를 GPU로 이동

    def _make_cuda(self):
        """
        배치를 GPU로 이동시킵니다.
        """
        self.sgn = self.sgn.cuda()
        self.sgn_mask = self.sgn_mask.cuda()
        self.sgn_lengths = self.sgn_lengths.cuda() # sgn_lengths도 이동

        if self.trg_label is not None:
            self.trg_label = self.trg_label.cuda()
        
        # txt, txt_mask, txt_input의 CUDA 전송 로직 제거됨

    def sort_by_sgn_lengths(self):
        """
        수어 시퀀스 길이(sgn_length)로 정렬하고 (내림차순),
        원래 순서로 되돌리기 위한 인덱스를 반환합니다.
        RNN 패킹/패딩 효율성을 위해 사용됩니다.
        """
        # sgn_lengths를 기준으로 내림차순 정렬하고, 정렬된 인덱스(perm_index)를 얻음
        _, perm_index = self.sgn_lengths.sort(0, descending=True)
        # 원래 순서로 되돌리기 위한 역 인덱스(rev_index) 생성
        rev_index = torch.zeros_like(perm_index, device=perm_index.device)
        # perm_index가 연속적이거나 0부터 시작하지 않을 수 있으므로 더 견고한 방법으로 rev_index 생성
        for new_pos, old_pos in enumerate(perm_index.cpu().tolist()): # CPU로 옮기고 리스트로 변환하여 반복
            rev_index[old_pos] = new_pos

        # perm_index를 사용하여 모든 관련 텐서 및 리스트를 정렬
        self.sgn = self.sgn[perm_index]
        self.sgn_mask = self.sgn_mask[perm_index]
        self.sgn_lengths = self.sgn_lengths[perm_index]

        # sequence와 signer 메타 정보도 정렬
        # 이 정보들은 리스트일 수 있으므로, 리스트인지 확인 후 리스트 컴프리헨션 사용
        if isinstance(self.signer, list):
            self.signer = [self.signer[pi] for pi in perm_index.cpu().tolist()]
        if isinstance(self.sequence, list):
            self.sequence = [self.sequence[pi] for pi in perm_index.cpu().tolist()]

        if self.trg_label is not None:
            self.trg_label = self.trg_label[perm_index]

        # 텐서가 이미 대상 장치에 있고 perm_index도 같은 장치에 있거나 연산이 장치에 구애받지 않는 경우
        # 여기서 _make_cuda()를 다시 호출할 필요는 없습니다.
        # 그러나 perm_index가 정렬을 위해 CPU에 있었고 텐서가 CUDA인 경우 인덱싱 후 CPU가 됩니다.
        # use_cuda가 true이면 필요한 경우 다시 이동해야 합니다.
        # 원본 코드는 _make_cuda()를 호출했습니다. perm_index가 CPU일 수 있는 경우를 대비해 유지합니다.
        # 더 나은 접근 방식: 인덱싱 전에 perm_index가 텐서와 동일한 장치에 있는지 확인합니다.
        if self.use_cuda:
             # perm_index가 정렬을 위해 CPU에 있었고 텐서가 인덱싱 후 CPU가 된 경우:
             self._make_cuda() # use_cuda가 True이면 다시 CUDA로 이동하도록 보장합니다.

        return rev_index # 원래 순서로 되돌리기 위한 인덱스 반환
