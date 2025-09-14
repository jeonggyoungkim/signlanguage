# coding: utf-8
"""
어텐션 모듈
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


class AttentionMechanism(nn.Module):
    """
    기본 어텐션 클래스
    """

    def forward(self, *inputs):
        raise NotImplementedError("이 메서드를 구현해야 합니다.")


class BahdanauAttention(AttentionMechanism):
    """
    Bahdanau (MLP) 어텐션 구현

    https://arxiv.org/pdf/1409.0473.pdf의 A.1.2 섹션 참조
    """

    def __init__(self, hidden_size=1, key_size=1, query_size=1):
        """
        어텐션 메커니즘을 생성합니다.

        :param hidden_size: 쿼리와 키를 위한 투영(projection) 크기
        :param key_size: 어텐션 입력 키의 크기
        :param query_size: 쿼리의 크기
        """

        super(BahdanauAttention, self).__init__()

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.proj_keys = None  # 투영된 키를 저장
        self.proj_query = None  # 투영된 쿼리

    # pylint: disable=arguments-differ
    def forward(self, query: Tensor = None, mask: Tensor = None, values: Tensor = None):
        """
        Bahdanau MLP 어텐션 순전파

        :param query: 키/메모리와 비교할 항목(디코더 상태),
            크기 (batch_size, 1, decoder.hidden_size)
        :param mask: 키 위치를 마스킹 (유효하지 않은 위치는 0, 나머지는 1),
            크기 (batch_size, 1, sgn_length)
        :param values: 값(인코더 상태),
            크기 (batch_size, sgn_length, encoder.hidden_size)
        :return: 크기 (batch_size, 1, value_size)의 컨텍스트 벡터,
            크기 (batch_size, 1, sgn_length)의 어텐션 확률
        """
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert mask is not None, "마스크가 필요합니다"
        assert self.proj_keys is not None, "투영된 키가 미리 계산되어 있어야 합니다"

        # 먼저 쿼리(디코더 상태)를 투영합니다.
        # 투영된 키(인코더 상태)는 이미 미리 계산되어 있습니다.
        self.compute_proj_query(query)

        # 점수를 계산합니다.
        # proj_keys: batch x sgn_len x hidden_size
        # proj_query: batch x 1 x hidden_size
        scores = self.energy_layer(torch.tanh(self.proj_query + self.proj_keys))
        # scores: batch x sgn_len x 1

        scores = scores.squeeze(2).unsqueeze(1)
        # scores: batch x 1 x time

        # 마스크된 위치를 -inf로 채워서 유효하지 않은 위치를 마스킹
        scores = torch.where(mask, scores, scores.new_full([1], float("-inf")))

        # 점수를 확률로 변환
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x time

        # 컨텍스트 벡터는 값들의 가중 합
        context = alphas @ values  # batch x 1 x value_size

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        """
        키의 투영을 계산합니다.
        개별 쿼리를 받기 전에 미리 계산하면 효율적입니다.

        :param keys: 투영할 키
        :return: 없음
        """
        self.proj_keys = self.key_layer(keys)

    def compute_proj_query(self, query: Tensor):
        """
        쿼리의 투영을 계산합니다.

        :param query: 투영할 쿼리
        :return: 없음
        """
        self.proj_query = self.query_layer(query)

    def _check_input_shapes_forward(
        self, query: torch.Tensor, mask: torch.Tensor, values: torch.Tensor
    ):
        """
        `self.forward`에 입력되는 텐서들의 크기가 올바른지 확인합니다.
        `self.forward`와 동일한 입력 의미를 가집니다.

        :param query: 쿼리 텐서
        :param mask: 마스크 텐서
        :param values: 값 텐서
        :return: 없음
        """
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.query_layer.in_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "BahdanauAttention"


class LuongAttention(AttentionMechanism):
    """
    Luong (이중선형/곱셈) 어텐션 구현

    http://aclweb.org/anthology/D15-1166의 식 8("general") 참조
    """

    def __init__(self, hidden_size: int = 1, key_size: int = 1):
        """
        어텐션 메커니즘을 생성합니다.

        :param hidden_size: 키 투영 레이어의 크기, 디코더 히든 크기와 같아야 함
        :param key_size: 어텐션 입력 키의 크기
        """

        super(LuongAttention, self).__init__()
        self.key_layer = nn.Linear(
            in_features=key_size, out_features=hidden_size, bias=False
        )
        self.proj_keys = None  # 투영된 키

    # pylint: disable=arguments-differ
    def forward(
        self,
        query: torch.Tensor = None,
        mask: torch.Tensor = None,
        values: torch.Tensor = None,
    ):
        """
        Luong (곱셈/이중선형) 어텐션 순전파
        주어진 쿼리와 모든 마스킹된 값에 대해 컨텍스트 벡터와 어텐션 점수를 계산하고 반환합니다.

        :param query: 키/메모리와 비교할 항목(디코더 상태),
            크기 (batch_size, 1, decoder.hidden_size)
        :param mask: 키 위치를 마스킹 (유효하지 않은 위치는 0, 나머지는 1),
            크기 (batch_size, 1, sgn_length)
        :param values: 값(인코더 상태),
            크기 (batch_size, sgn_length, encoder.hidden_size)
        :return: 크기 (batch_size, 1, value_size)의 컨텍스트 벡터,
            크기 (batch_size, 1, sgn_length)의 어텐션 확률
        """
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert self.proj_keys is not None, "투영된 키가 미리 계산되어 있어야 합니다"
        assert mask is not None, "마스크가 필요합니다"

        # scores: batch_size x 1 x sgn_length
        scores = query @ self.proj_keys.transpose(1, 2)

        # 마스크된 위치를 -inf로 채워서 유효하지 않은 위치를 마스킹
        scores = torch.where(mask, scores, scores.new_full([1], float("-inf")))

        # 점수를 확률로 변환
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x sgn_len

        # 컨텍스트 벡터는 값들의 가중 합
        context = alphas @ values  # batch x 1 x values_size

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        """
        키의 투영을 계산하고 `self.proj_keys`에 할당합니다.
        이 사전 계산은 개별 쿼리를 받기 전에 모든 키에 대해 효율적으로 수행됩니다.

        :param keys: 크기 (batch_size, sgn_length, encoder.hidden_size)
        """
        # proj_keys: batch x sgn_len x hidden_size
        self.proj_keys = self.key_layer(keys)

    def _check_input_shapes_forward(
        self, query: torch.Tensor, mask: torch.Tensor, values: torch.Tensor
    ):
        """
        `self.forward`에 입력되는 텐서들의 크기가 올바른지 확인합니다.
        `self.forward`와 동일한 입력 의미를 가집니다.

        :param query: 쿼리 텐서
        :param mask: 마스크 텐서
        :param values: 값 텐서
        :return: 없음
        """
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.key_layer.out_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "LuongAttention"
