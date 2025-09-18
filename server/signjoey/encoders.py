# coding: utf-8

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple

from signjoey.helpers import freeze_params


# pylint: disable=abstract-method
class Encoder(nn.Module):
    """
    기본 인코더 클래스
    """

    @property
    def output_size(self):
        """
        출력 크기를 반환합니다.

        :return:
        """
        return self._output_size


class RecurrentEncoder(Encoder):
    """단어 임베딩 시퀀스를 인코딩합니다"""

    # pylint: disable=unused-argument
    def __init__(
        self,
        rnn_type: str = "gru",
        hidden_size: int = 1,
        emb_size: int = 1,
        num_layers: int = 1,
        dropout: float = 0.0,
        emb_dropout: float = 0.0,
        bidirectional: bool = True,
        freeze: bool = False,
        **kwargs
    ) -> None:
        """
        새로운 순환 인코더를 생성합니다.

        :param rnn_type: RNN 유형: `gru` 또는 `lstm`
        :param hidden_size: 각 RNN의 크기
        :param emb_size: 단어 임베딩의 크기
        :param num_layers: 인코더 RNN 레이어 수
        :param dropout: RNN 레이어 사이에 적용되는 드롭아웃
        :param emb_dropout: RNN 입력(단어 임베딩)에 적용되는 드롭아웃
        :param bidirectional: 양방향 RNN 사용 여부
        :param freeze: 학습 중 인코더의 파라미터 고정 여부
        :param kwargs:
        """

        super(RecurrentEncoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.emb_size = emb_size

        # RNN 타입 선택 (RNN, LSTM, GRU 지원)
        if rnn_type.lower() == "rnn":
            rnn = nn.RNN
        elif rnn_type.lower() == "lstm":
            rnn = nn.LSTM
        elif rnn_type.lower() == "gru":
            rnn = nn.GRU
        else:
            raise ValueError(f"지원되지 않는 RNN 타입: {rnn_type}. 지원되는 타입: rnn, lstm, gru")

        self.rnn = rnn(
            emb_size,
            hidden_size,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self._output_size = 2 * hidden_size if bidirectional else hidden_size

        if freeze:
            freeze_params(self)

    # pylint: disable=invalid-name, unused-argument
    def _check_shapes_input_forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> None:
        """
        `self.forward`에 입력되는 텐서들의 형태가 올바른지 확인합니다.
        `self.forward`와 동일한 입력 의미를 가집니다.

        :param embed_src: 임베딩된 소스 토큰
        :param src_length: 소스 길이
        :param mask: 소스 마스크
        """
        assert embed_src.shape[0] == src_length.shape[0]
        assert embed_src.shape[2] == self.emb_size
        # assert mask.shape == embed_src.shape
        assert len(src_length.shape) == 1

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        임베딩 시퀀스 x에 양방향 RNN을 적용합니다.
        입력 미니배치 x는 소스 길이에 따라 정렬되어야 합니다.
        x와 mask는 동일한 차원 [batch, time, dim]을 가져야 합니다.

        :param embed_src: 임베딩된 소스 입력,
            형태 (batch_size, src_len, embed_size)
        :param src_length: 소스 입력의 길이
            (패딩 전 토큰 수), 형태 (batch_size)
        :param mask: 패딩 영역을 나타냄 (패딩이 있는 곳은 0),
            형태 (batch_size, src_len, embed_size)
        :return:
            - output: 은닉 상태,
                형태 (batch_size, max_length, directions*hidden)
            - hidden_concat: 마지막 은닉 상태,
                형태 (batch_size, directions*hidden)
        """
        self._check_shapes_input_forward(
            embed_src=embed_src, src_length=src_length, mask=mask
        )

        # RNN 입력에 드롭아웃 적용
        embed_src = self.emb_dropout(embed_src)

        # src_length를 CPU로 이동
        packed = pack_padded_sequence(embed_src, src_length.cpu(), batch_first=True)
        output, hidden = self.rnn(packed)

        # pylint: disable=unused-variable
        if isinstance(hidden, tuple):
            hidden, memory_cell = hidden

        output, _ = pad_packed_sequence(output, batch_first=True)
        # hidden: dir*layers x batch x hidden
        # output: batch x max_length x directions*hidden
        # batch_size = hidden.size()[1] # 이 줄은 새 로직에서 필요 없어짐

        # 수정된 로직 시작
        num_layers = self.rnn.num_layers
        # hidden 텐서는 (num_layers * num_directions, batch_size, hidden_size) 형태입니다.
        
        if self.rnn.bidirectional:
            # 양방향일 경우, 마지막 계층의 정방향 은닉 상태는 인덱스 (2*num_layers - 2)에,
            # 역방향 은닉 상태는 인덱스 (2*num_layers - 1)에 위치합니다.
            # 예: num_layers=1, bi=True -> hidden[0] (fwd), hidden[1] (bwd)
            # 예: num_layers=3, bi=True -> hidden[4] (fwd_layer2), hidden[5] (bwd_layer2)
            last_fwd_h = hidden[2*num_layers - 2, :, :] # 형태: (batch_size, hidden_size)
            last_bwd_h = hidden[2*num_layers - 1, :, :] # 형태: (batch_size, hidden_size)
            # 이들을 연결하여 (batch_size, 2 * hidden_size) 형태의 텐서 생성
            hidden_concat = torch.cat((last_fwd_h, last_bwd_h), dim=1)
        else:
            # 단방향일 경우, 마지막 계층의 은닉 상태는 인덱스 (num_layers - 1)에 위치합니다.
            # 예: num_layers=1, bi=False -> hidden[0]
            # 예: num_layers=3, bi=False -> hidden[2]
            last_fwd_h = hidden[num_layers - 1, :, :] # 형태: (batch_size, hidden_size)
            hidden_concat = last_fwd_h
        # 수정된 로직 끝

        # hidden_concat: batch x directions*hidden (주석은 이전과 동일한 의미)
        return output, hidden_concat

    def __repr__(self):
        return "%s(%r)" % (self.__class__.__name__, self.rnn)