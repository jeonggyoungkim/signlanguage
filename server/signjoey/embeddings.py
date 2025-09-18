# coding: utf-8
"""
임베딩 모듈
"""
import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from signjoey.helpers import freeze_params


def get_activation(activation_type):
    """활성화 함수를 반환합니다."""
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "celu":
        return nn.CELU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softplus":
        return nn.Softplus()
    elif activation_type == "softshrink":
        return nn.Softshrink()
    elif activation_type == "softsign":
        return nn.Softsign()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("알 수 없는 활성화 함수 유형 {}".format(activation_type))


class MaskedNorm(nn.Module):
    """
    원본 코드 출처:
    https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("지원하지 않는 정규화 레이어")

        self.num_features = num_features

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            reshaped_mask = mask.reshape([-1, 1]) > 0
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )
            batch_normed = self.norm(selected)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])


# TODO (Cihan): Spatial과 Word 임베딩은 거의 동일합니다.
#       단일 모듈 클래스로 변환하는 것이 좋을 것 같습니다.
#       유일한 차이점은 lut와 linear 레이어입니다.
class Embeddings(nn.Module):
    """
    단어 임베딩 클래스
    """

    def __init__(
        self,
        embedding_dim: int,
        scale: bool = True,
        vocab_size: int = 0,
        padding_idx: int = 1,
        freeze: bool = False,
        **kwargs,
    ):
        """
        임베딩을 초기화합니다.

        :param embedding_dim: 임베딩 차원
        :param scale: 스케일링 여부
        :param vocab_size: 어휘 크기
        :param padding_idx: 패딩 인덱스
        :param freeze: 고정 여부
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)

        self.scale = scale
        if scale:
            self._scale = math.sqrt(embedding_dim)

        if freeze:
            freeze_params(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        순전파를 수행합니다.

        :param x: 입력 텐서
        :return: 임베딩된 텐서
        """
        if self.scale:
            return self.lut(x) * self._scale
        return self.lut(x)

    def __repr__(self):
        """
        임베딩의 문자열 표현을 반환합니다.
        """
        return f"{self.__class__.__name__}(embedding_dim={self.embedding_dim}, vocab_size={self.vocab_size})"


class SpatialEmbeddings(nn.Module):

    """
    간단한 선형 투영 레이어
    (인코더 출력을 gloss로 예측하기 위한)
    """

    # pylint: disable=unused-argument
    def __init__(
        self,
        data_cfg,
        embedding_dim: int,
        input_size: int,
        num_heads: int,
        freeze: bool = False,
        norm_type: str = None,
        activation_type: str = None,
        scale: bool = False,
        scale_factor: float = None,
        **kwargs
    ):
        """
        어휘에 대한 새로운 임베딩을 생성합니다.
        Transformer를 위해 스케일링을 사용합니다.

        :param embedding_dim: 임베딩 차원
        :param input_size: 입력 크기
        :param freeze: 학습 중 임베딩 고정 여부
        """
        super().__init__()

        self.embedding_dim = embedding_dim
        self.input_size = input_size
        self.ln = nn.Linear(self.input_size, self.embedding_dim)

        self.norm_type = norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=norm_type, num_groups=num_heads, num_features=embedding_dim
            )

        self.activation_type = activation_type
        if self.activation_type:
            self.activation = get_activation(activation_type)

        self.scale = scale
        if self.scale:
            if scale_factor:
                self.scale_factor = scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        입력 `x`를 임베딩 공간으로 투영합니다.

        :param mask: 토큰 마스크
        :param x: 입력 텐서
        :return: 임베딩된 표현
        """
        x = self.ln(x)

        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, input_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_size,
        )


def freeze_params(module: nn.Module) -> None:
    """
    모듈의 파라미터를 고정합니다.

    :param module: 고정할 모듈
    """
    for _, p in module.named_parameters():
        p.requires_grad = False
