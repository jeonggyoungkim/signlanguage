# coding: utf-8

"""
커스텀 초기화 구현
"""

import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.init import _calculate_fan_in_and_fan_out


def orthogonal_rnn_init_(cell: nn.RNNBase, gain: float = 1.0):
    """
    순환 가중치의 직교 초기화
    RNN 파라미터는 하나의 파라미터에 3개 또는 4개의 행렬을 포함하므로, 이를 분할합니다.
    """
    with torch.no_grad():
        for _, hh, _, _ in cell.all_weights:
            for i in range(0, hh.size(0), cell.hidden_size):
                nn.init.orthogonal_(hh.data[i : i + cell.hidden_size], gain=gain)


def lstm_forget_gate_init_(cell: nn.RNNBase, value: float = 1.0) -> None:
    """
    LSTM 망각 게이트를 `value`로 초기화합니다.

    :param cell: LSTM 셀
    :param value: 초기값, 기본값: 1
    """
    with torch.no_grad():
        for _, _, ih_b, hh_b in cell.all_weights:
            l = len(ih_b)
            ih_b.data[l // 4 : l // 2].fill_(value)
            hh_b.data[l // 4 : l // 2].fill_(value)


def xavier_uniform_n_(w: Tensor, gain: float = 1.0, n: int = 4) -> None:
    """
    효율성을 위해 하나의 파라미터에 여러 행렬을 결합하는 파라미터를 위한 Xavier 초기화 함수입니다.
    이는 예를 들어 GRU와 LSTM 파라미터에 사용되며, 모든 게이트가 하나의 큰 행렬로 동시에 계산됩니다.

    :param w: 파라미터
    :param gain: 기본값 1
    :param n: 기본값 4
    """
    with torch.no_grad():
        fan_in, fan_out = _calculate_fan_in_and_fan_out(w)
        assert fan_out % n == 0, "fan_out should be divisible by n"
        fan_out //= n
        std = gain * math.sqrt(2.0 / (fan_in + fan_out))
        a = math.sqrt(3.0) * std
        nn.init.uniform_(w, -a, a)


# pylint: disable=too-many-branches
def initialize_model(model: nn.Module, cfg: dict, txt_padding_idx: int) -> None:
    """
    제공된 설정에 따라 모델을 초기화합니다.

    모든 초기화 설정은 설정 파일의 `model` 섹션에 포함됩니다.
    예시는 `https://github.com/joeynmt/joeynmt/
    blob/master/configs/iwslt_envi_xnmt.yaml#L47`를 참조하세요.

    주요 초기화 함수는 `initializer` 키를 사용하여 설정됩니다.
    가능한 값은 `xavier`, `uniform`, `normal` 또는 `zeros`입니다.
    (`xavier`가 기본값입니다).

    초기화 함수가 `uniform`으로 설정된 경우, `init_weight`는 값의 범위를 설정합니다
    (-init_weight, init_weight).

    초기화 함수가 `normal`로 설정된 경우, `init_weight`는 가중치의 표준 편차를 설정합니다
    (평균 0).

    단어 임베딩 초기화 함수는 `embed_initializer`를 사용하여 설정되며 동일한 값을 사용합니다.
    기본값은 `normal`이며 `embed_init_weight = 0.01`입니다.

    편향은 `bias_initializer`를 사용하여 별도로 초기화됩니다.
    기본값은 `zeros`이지만, 주요 초기화 함수와 동일한 초기화 함수를 사용할 수 있습니다.

    RNN 직교 초기화를 원하는 경우 `init_rnn_orthogonal`을 True로 설정하세요
    (순환 행렬용). 기본값은 False입니다.

    `lstm_forget_gate`는 LSTM 망각 게이트의 초기화 방식을 제어합니다.
    기본값은 `1`입니다.

    :param model: 초기화할 모델
    :param cfg: 모델 설정
    :param txt_padding_idx: 구어 텍스트 패딩 토큰의 인덱스
    """

    # 기본값: xavier, 임베딩: normal 0.01, 편향: zeros, 직교 없음
    gain = float(cfg.get("init_gain", 1.0))  # xavier용
    init = cfg.get("initializer", "xavier")
    init_weight = float(cfg.get("init_weight", 0.01))

    embed_init = cfg.get("embed_initializer", "normal")
    embed_init_weight = float(cfg.get("embed_init_weight", 0.01))
    embed_gain = float(cfg.get("embed_init_gain", 1.0))  # xavier용

    bias_init = cfg.get("bias_initializer", "zeros")
    bias_init_weight = float(cfg.get("bias_init_weight", 0.01))

    # pylint: disable=unnecessary-lambda, no-else-return
    def _parse_init(s, scale, _gain):
        scale = float(scale)
        assert scale > 0.0, "incorrect init_weight"
        if s.lower() == "xavier":
            return lambda p: nn.init.xavier_uniform_(p, gain=_gain)
        elif s.lower() == "uniform":
            return lambda p: nn.init.uniform_(p, a=-scale, b=scale)
        elif s.lower() == "normal":
            return lambda p: nn.init.normal_(p, mean=0.0, std=scale)
        elif s.lower() == "zeros":
            return lambda p: nn.init.zeros_(p)
        else:
            raise ValueError("unknown initializer")

    init_fn_ = _parse_init(init, init_weight, gain)
    embed_init_fn_ = _parse_init(embed_init, embed_init_weight, embed_gain)
    bias_init_fn_ = _parse_init(bias_init, bias_init_weight, gain)

    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'encoder' in name or 'decoder' in name or 'txt_embed' in name:
                continue

            if "txt_embed" in name:
                if "lut" in name:
                    embed_init_fn_(p)

            elif "bias" in name:
                bias_init_fn_(p)

            elif len(p.size()) > 1:

                # RNN은 여러 행렬을 하나로 결합하여 xavier 초기화를 복잡하게 만듦
                if init == "xavier" and "rnn" in name:
                    n = 1
                    if "encoder" in name:
                        n = 4 if isinstance(model.encoder.rnn, nn.LSTM) else 3
                    elif "decoder" in name:
                        n = 4 if isinstance(model.decoder.rnn, nn.LSTM) else 3
                    xavier_uniform_n_(p.data, gain=gain, n=n)
                else:
                    init_fn_(p)

        # 패딩을 0으로 초기화
        if model.txt_embed is not None:
            model.txt_embed.lut.weight.data[txt_padding_idx].zero_()

        orthogonal = cfg.get("init_rnn_orthogonal", False)
        lstm_forget_gate = cfg.get("lstm_forget_gate", 1.0)

        # 인코더 RNN 직교 초기화 & LSTM 망각 게이트
        if hasattr(model.encoder, "rnn"):

            if orthogonal:
                orthogonal_rnn_init_(model.encoder.rnn)

            if isinstance(model.encoder.rnn, nn.LSTM):
                lstm_forget_gate_init_(model.encoder.rnn, lstm_forget_gate)

        # 디코더 RNN 직교 초기화 & LSTM 망각 게이트
        if hasattr(model.decoder, "rnn"):

            if orthogonal:
                orthogonal_rnn_init_(model.decoder.rnn)

            if isinstance(model.decoder.rnn, nn.LSTM):
                lstm_forget_gate_init_(model.decoder.rnn, lstm_forget_gate)
