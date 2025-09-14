# coding: utf-8
"""
모델 정의
"""
from typing import Dict, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from signjoey.embeddings import Embeddings, SpatialEmbeddings
from signjoey.encoders import Encoder, RecurrentEncoder
from signjoey.helpers import freeze_params
from signjoey.vocabulary import Vocabulary


class SignModel(nn.Module):
    """
    단어 분류를 위한 RNN 기반 모델
    """

    def __init__(
        self,
        encoder: Encoder,
        src_embed: nn.Module,
        trg_vocab: Vocabulary, # 타겟 단어 레이블 어휘
        num_classes: int, # 분류할 총 단어 수 (len(trg_vocab)과 동일)
        cfg: dict,
    ):
        """
        모델을 초기화합니다.

        :param encoder: 인코더 (RecurrentEncoder)
        :param src_embed: 소스 임베딩 (키포인트 임베딩 또는 처리 레이어)
        :param trg_vocab: 타겟 단어 레이블 어휘
        :param num_classes: 분류할 총 단어 수
        :param cfg: 설정
        """
        super(SignModel, self).__init__()

        self.src_embed = src_embed
        self.encoder = encoder
        self.trg_vocab = trg_vocab # 타겟 단어 레이블을 위한 어휘
        self.cfg = cfg
        self.num_classes = num_classes

        # RNN 인코더의 출력을 받아 단어 클래스로 분류하는 선형 레이어
        self.output_layer = nn.Linear(self.encoder.output_size, num_classes)

        self.pad_index = self.trg_vocab.stoi.get("<pad>", -100) # pad_index가 없거나 어휘에 <pad>가 없으면 -100 (무시)
        self._loss_function = nn.CrossEntropyLoss(ignore_index=self.pad_index)

    @property
    def loss_function(self):
        """
        손실 함수를 반환합니다.
        """
        return self._loss_function

    @loss_function.setter
    def loss_function(self, loss_function):
        """
        손실 함수를 설정합니다.
        """
        self._loss_function = loss_function

    def forward(
        self,
        src: Tensor, # 키포인트 시퀀스 (batch_size, seq_len, feature_dim)
        src_mask: Tensor, # 소스 마스크 (batch_size, 1, seq_len)
        src_length: Tensor, # 소스 길이 (batch_size)
        trg_labels: Tensor = None, # 타겟 단어 레이블 (batch_size) - 학습 시에만 사용
    ) -> Tuple[Tensor, Tensor]:
        """
        순전파를 수행합니다.

        :param src: 소스 입력 (키포인트 시퀀스)
        :param src_mask: 소스 마스크
        :param src_length: 소스 길이
        :param trg_labels: 타겟 단어 레이블 (학습 시)
        :return: 손실 (학습 시) 및 로짓 (항상)
        """
        # 입력 임베딩/처리
        # SpatialEmbeddings의 경우 mask를 사용, 일반 Linear 레이어 등을 src_embed로 사용 시 mask 인자 불필요할 수 있음
        if isinstance(self.src_embed, SpatialEmbeddings):
            embedded_src = self.src_embed(src, mask=src_mask)
        else: # nn.Linear, nn.Identity 등
            embedded_src = self.src_embed(src)

        # 인코더 순전파
        encoder_output, hidden_for_classification = self.encoder(
            embed_src=embedded_src,
            src_length=src_length,
            mask=src_mask
        )

        # hidden_for_classification은 RecurrentEncoder의 hidden_concat이며,
        # 이미 (batch_size, directions*hidden_size) 형태를 가집니다.
        # 따라서 if isinstance(encoder_hidden, tuple): ... else: ... 부분은 필요 없습니다.
        final_hidden_state = hidden_for_classification 

        logits = self.output_layer(final_hidden_state)

        loss = None
        if trg_labels is not None:
            print(f"[DEBUG] logits.shape: {logits.shape}, trg_labels.shape: {trg_labels.shape}, trg_labels: {trg_labels}") # 디버깅 출력 추가
            loss = self.loss_function(logits, trg_labels)

        return loss, logits

    def encode(
        self, src: Tensor, src_length: Tensor, src_mask: Tensor
    ) -> Tuple[Tensor, Tensor]: # 이 메소드는 사실상 forward에서 처리되므로 직접 호출될 일이 줄어듬
        """
        인코더를 통해 소스 시퀀스(키포인트)를 인코딩합니다.
        실제 인코딩 로직은 forward 메소드 내에 통합되었습니다.
        이 메소드는 API 일관성을 위해 남겨두거나, 내부 호출용으로만 사용할 수 있습니다.
        """
        if isinstance(self.src_embed, SpatialEmbeddings):
            embedded_src = self.src_embed(src, mask=src_mask)
        else:
            embedded_src = self.src_embed(src)

        return self.encoder(
            embed_src=embedded_src,
            src_length=src_length,
            mask=src_mask
        )

    def get_loss_for_batch(
        self,
        batch: Dict[str, Tensor],
    ) -> Tuple[Tensor, int]:
        """
        배치에 대한 손실을 계산합니다.

        :param batch: 배치 데이터 (src, src_mask, src_length, trg_label 포함해야 함)
        :return: 손실, 유효 샘플 수
        """
        loss, logits = self.forward(
            src=batch.src,
            src_mask=batch.src_mask,
            src_length=batch.src_length,
            trg_labels=batch.trg_label,
        )
        
        # pad_index가 -100이 아닌 경우, 실제 레이블이 pad_index가 아닌 샘플만 카운트
        if self.pad_index != -100:
             valid_samples = (batch.trg_label != self.pad_index).sum().item()
        else: # 모든 샘플이 유효하다고 간주
             valid_samples = batch.trg_label.size(0)
        
        # loss가 None이 아닐 경우 (즉, trg_labels가 제공된 경우)에만 valid_samples로 나눠 평균 손실 계산
        # 하지만 CrossEntropyLoss가 reduction='mean' (기본값)으로 설정되어 있다면 이미 평균이 계산됨.
        # 따라서, 여기서는 총 손실 (sum)과 유효 샘플 수를 반환하거나,
        # 이미 평균화된 손실과 유효 샘플 수를 반환.
        # 현재 CrossEntropyLoss 기본값 사용하므로 loss는 이미 평균화됨.
        # n_tokens 역할로 valid_samples 사용.
        return loss, valid_samples

    def run_batch(
        self,
        batch: Dict[str, Tensor],
    ) -> Tensor:
        """
        주어진 배치에 대해 모델을 실행하여 예측합니다 (추론).

        :param batch: 배치 데이터 (src, src_mask, src_length 포함)
        :return: 예측된 레이블 인덱스 (Tensor, shape: batch_size)
        """
        _, logits = self.forward(
            src=batch.src,
            src_mask=batch.src_mask,
            src_length=batch.src_length,
        )
        predicted_labels = torch.argmax(logits, dim=-1)
        return predicted_labels

    def __repr__(self) -> str:
        """
        모델의 문자열 표현을 반환합니다.
        """
        return (
            f"{self.__class__.__name__}(\n"
            f"\tsrc_embed={self.src_embed},\n"
            f"\tencoder={self.encoder},\n"
            f"\toutput_layer={self.output_layer},\n"
            f"\ttrg_vocab_size={len(self.trg_vocab) if self.trg_vocab else 'N/A'},"
            f"\tnum_classes={self.num_classes})"
        )


def build_model(
    cfg: dict,
    trg_vocab: Vocabulary, # 타겟 단어 레이블 어휘
) -> SignModel:
    """
    단어 분류 모델을 구축합니다.

    :param cfg: 설정
    :param trg_vocab: 타겟 단어 레이블 어휘
    :return: 모델
    """
    model_cfg = cfg["model"] # "model" 설정이 있다고 가정
    encoder_cfg = model_cfg["encoder"]
    embeddings_cfg = model_cfg.get("embeddings", encoder_cfg.get("embeddings")) # model.embeddings 우선, 없으면 encoder.embeddings
    
    data_cfg = cfg["data"] # 데이터 설정을 가져옵니다.
    # num_classes = len(trg_vocab) # 타겟 어휘 크기로 클래스 수 결정
    num_classes = model_cfg.get("num_classes", len(trg_vocab)) # 설정 파일의 num_classes 우선 사용
    # print(f"[DEBUG] In build_model: model_cfg_num_classes = {model_cfg.get('num_classes')}, len_trg_vocab = {len(trg_vocab)}, final num_classes = {num_classes}") # 디버깅 로그 제거

    # 소스 임베딩 (키포인트 처리)
    # 입력이 (batch, seq_len, feature_dim) 형태의 float 텐서라고 가정합니다.
    # SpatialEmbeddings를 사용하거나, 간단한 nn.Linear를 사용할 수 있습니다.
    # 또는, 특징 추출이 이미 완료되었다면 nn.Identity()를 사용할 수도 있습니다.
    emb_type = embeddings_cfg.get("type", "spatial") # 기본값을 spatial로 설정
    
    if emb_type == "spatial":
        # SpatialEmbeddings 클래스의 파라미터에 맞춰서 설정값을 제공해야 합니다.
        # 예를 들어, data_cfg도 전달해야 할 수 있습니다.
        src_embed = SpatialEmbeddings(
            data_cfg=data_cfg, # SpatialEmbeddings가 data_cfg를 필요로 한다면 전달
            embedding_dim=embeddings_cfg["embedding_dim"],
            input_size=data_cfg["feature_size"], # 키포인트의 특징 차원
            num_heads=encoder_cfg.get("num_heads", 8), # 예시 값, SpatialEmbeddings에 필요하다면 사용
            freeze=embeddings_cfg.get("freeze", False),
            norm_type=embeddings_cfg.get("norm_type", None),
            activation_type=embeddings_cfg.get("activation_type", None),
            scale=embeddings_cfg.get("scale", False),
            scale_factor=embeddings_cfg.get("scale_factor", None),
        )
        # RecurrentEncoder의 emb_size는 SpatialEmbeddings의 출력 차원(embedding_dim)과 일치해야 합니다.
        recurrent_encoder_emb_size = embeddings_cfg["embedding_dim"]
    elif emb_type == "linear":
        src_embed = nn.Linear(data_cfg["feature_size"], embeddings_cfg["embedding_dim"])
        recurrent_encoder_emb_size = embeddings_cfg["embedding_dim"]
    elif emb_type == "none" or emb_type == "identity": # 별도 임베딩 레이어 없이 직접 인코더에 전달
        src_embed = nn.Identity()
        recurrent_encoder_emb_size = data_cfg["feature_size"] # 이 경우 인코더의 emb_size는 특징 차원
    else:
        raise ValueError(f"알 수 없는 임베딩 유형: {emb_type}")

    # 인코더
    enc_type = encoder_cfg.get("type", "lstm").lower()  # 기본값을 lstm으로 변경
    
    if enc_type in ["rnn", "lstm", "gru", "recurrent"]:
        # RNN 기반 인코더들을 모두 RecurrentEncoder로 처리
        if enc_type == "recurrent":
            # 기존 recurrent 타입은 encoder_cfg에서 rnn_type을 가져옴
            rnn_type = encoder_cfg.get("rnn_type", "LSTM")
        else:
            # 직접 지정된 경우 (rnn, lstm, gru)
            rnn_type = enc_type.upper()
        
        # rnn_type을 encoder_cfg에 추가하여 RecurrentEncoder에 전달
        encoder_cfg_copy = encoder_cfg.copy()
        encoder_cfg_copy["rnn_type"] = rnn_type
        
        encoder = RecurrentEncoder(
            **{k: v for k, v in encoder_cfg_copy.items() if k not in ['embeddings', 'type']}, # embeddings, type 설정은 제외
            emb_size=recurrent_encoder_emb_size, 
        )
    else:
        raise ValueError(f"알 수 없는 인코더 유형 {enc_type}. 지원되는 타입: rnn, lstm, gru, recurrent")

    model = SignModel(
        encoder=encoder,
        src_embed=src_embed,
        trg_vocab=trg_vocab,
        num_classes=num_classes,
        cfg=cfg,
    )

    # 모델 파라미터 초기화
    for name, param in model.named_parameters():
        if param.requires_grad: # 학습 가능한 파라미터만 초기화
            if "bias" in name:
                nn.init.zeros_(param)
            elif "weight" in name:
                if "embed.lut.weight" in name: # 어휘 기반 Embeddings (현재는 사용 안함)
                     nn.init.normal_(param, mean=0.0, std=0.1)
                elif len(param.squeeze().shape) > 1 : # 2D 이상의 가중치
                    nn.init.xavier_uniform_(param)
                elif len(param.squeeze().shape) == 1 and "norm" not in name.lower() : # 1D 가중치 (LayerNorm 등 제외)
                    nn.init.normal_(param, mean=0.0, std=0.02) # 예시: 작은 값으로 초기화

    if encoder_cfg.get("freeze", False):
        freeze_params(model.encoder)
    if embeddings_cfg.get("type", "spatial") != "none" and embeddings_cfg.get("freeze", False):
        freeze_params(model.src_embed)

    return model
