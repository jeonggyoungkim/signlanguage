# coding: utf-8
"""
학습 손실 함수 구현 모듈
"""

import torch
from torch import nn, Tensor
from torch.autograd import Variable


class XentLoss(nn.Module):
    """
    선택적 레이블 스무딩이 있는 교차 엔트로피 손실
    """

    def __init__(self, pad_index: int, smoothing: float = 0.0):
        super(XentLoss, self).__init__()
        self.smoothing = smoothing
        self.pad_index = pad_index
        if self.smoothing <= 0.0:
            # 표준 교차 엔트로피 손실
            self.criterion = nn.NLLLoss(ignore_index=self.pad_index, reduction="sum")
        else:
            # KL 발산 손실로 계산된 커스텀 레이블 스무딩 손실
            self.criterion = nn.KLDivLoss(reduction="sum")

    def _smooth_targets(self, targets: Tensor, vocab_size: int):
        """
        타겟 분포를 스무딩합니다. 모든 비참조 단어는 "smoothing"에 따라
        균일한 확률 질량을 가집니다.

        :param targets: 타겟 인덱스, batch*seq_len
        :param vocab_size: 출력 어휘의 크기
        :return: 스무딩된 타겟 분포, batch*seq_len x vocab_size
        """
        # batch*seq_len x vocab_size
        smooth_dist = targets.new_zeros((targets.size(0), vocab_size)).float()
        # 스무딩으로 균일하게 분포 채우기
        smooth_dist.fill_(self.smoothing / (vocab_size - 2))
        # 참 레이블에 1-smoothing의 확률 할당 ("신뢰도")
        smooth_dist.scatter_(1, targets.unsqueeze(1).data, 1.0 - self.smoothing)
        # 패딩에 0의 확률 할당
        smooth_dist[:, self.pad_index] = 0
        # 패딩 영역 마스킹 (패딩 영역의 확률 합 = 0)
        padding_positions = torch.nonzero(targets.data == self.pad_index)
        # pylint: disable=len-as-condition
        if len(padding_positions) > 0:
            smooth_dist.index_fill_(0, padding_positions.squeeze(), 0.0)
        return Variable(smooth_dist, requires_grad=False)

    # pylint: disable=arguments-differ
    def forward(self, log_probs, targets):
        """
        로짓과 타겟 사이의 교차 엔트로피를 계산합니다.

        레이블 스무딩이 사용되는 경우, 타겟 분포는 원-핫이 아니며,
        올바른 타겟 토큰에 대해 "1-smoothing"이고 나머지 확률 질량은
        다른 토큰들에 균일하게 분포됩니다.

        :param log_probs: 모델이 예측한 로그 확률
        :param targets: 타겟 인덱스
        :return:
        """
        if self.smoothing > 0:
            targets = self._smooth_targets(
                targets=targets.contiguous().view(-1), vocab_size=log_probs.size(-1)
            )
            # targets: batch*seq_len x vocab_size 크기의 분포
            assert (
                log_probs.contiguous().view(-1, log_probs.size(-1)).shape
                == targets.shape
            )
        else:
            # targets: batch*seq_len 크기의 인덱스
            targets = targets.contiguous().view(-1)
        loss = self.criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), targets
        )
        return loss
