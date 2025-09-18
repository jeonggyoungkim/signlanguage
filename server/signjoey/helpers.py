# coding: utf-8
"""
다양한 헬퍼 함수들을 모아놓은 모듈입니다.
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from sys import platform
from logging import Logger
from typing import Callable, Optional
import numpy as np

import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import yaml
from signjoey.vocabulary import GlossVocabulary


def make_model_dir(model_dir: str, overwrite: bool = False) -> str:
    """
    모델을 위한 새 디렉토리를 생성합니다.

    :param model_dir: 모델 디렉토리 경로
    :param overwrite: 기존 디렉토리를 덮어쓸지 여부
    :return: 모델 디렉토리 경로
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError("모델 디렉토리가 이미 존재하며 덮어쓰기가 비활성화되어 있습니다.")
        # 이전 디렉토리를 삭제하고 빈 디렉토리로 다시 시작합니다.
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    학습 과정을 로깅하기 위한 로거를 생성합니다.

    :param model_dir: 로깅 디렉토리 경로
    :param log_file: 로깅 파일 경로
    :return: 로거 객체
    """
    logger = logging.getLogger(__name__)
    if not logger.handlers: # 핸들러가 이미 설정되지 않은 경우에만 설정
        logger.setLevel(level=logging.DEBUG) # 로거 레벨 설정
        fh = logging.FileHandler("{}/{}".format(model_dir, log_file)) # 파일 핸들러 생성
        fh.setLevel(level=logging.DEBUG) # 파일 핸들러 레벨 설정
        logger.addHandler(fh) # 로거에 파일 핸들러 추가
        formatter = logging.Formatter("%(asctime)s %(message)s") # 로그 포맷 설정
        fh.setFormatter(formatter)
        if platform == "linux": # Linux 환경일 경우 콘솔 출력 핸들러 추가 (선택적)
            sh = logging.StreamHandler()
            sh.setLevel(logging.INFO)
            sh.setFormatter(formatter)
            logging.getLogger("").addHandler(sh)
        logger.info("안녕하세요! Joey-NMT 입니다.") # JoeyNMT는 원래 프로젝트 이름, SignJoey로 변경 고려 가능
        return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg"):
    """
    설정(configuration)을 로그에 기록합니다.

    :param cfg: 로깅할 설정 정보 (딕셔너리 형태)
    :param logger: 로그를 기록할 로거 객체
    :param prefix: 로깅 시 사용할 접두사
    """
    for k, v in cfg.items():
        if isinstance(v, dict): # 값이 딕셔너리인 경우 재귀적으로 호출
            p = ".".join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = ".".join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    N개의 동일한 레이어를 생성합니다.

    :param module: 복제할 모듈
    :param n: 복제할 횟수
    :return: 복제된 모듈 리스트 (nn.ModuleList)
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    후속 위치를 마스킹합니다 (미래 위치에 어텐션하는 것을 방지하기 위함).
    (원래 Transformer 헬퍼 함수)

    :param size: 마스크의 크기 (2번째 및 3번째 차원)
    :return: 0과 1로 이루어진 텐서 (shape: 1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


def set_seed(seed: int):
    """
    torch, numpy, random 모듈의 랜덤 시드를 설정합니다.

    :param seed: 랜덤 시드 값
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(
    train_data: Dataset,
    valid_data: Dataset,
    test_data: Dataset,
    gls_vocab: GlossVocabulary,
    logging_function: Callable[[str], None],
):
    """
    데이터 및 어휘 사전의 통계를 기록합니다.

    :param train_data: 학습 데이터셋
    :param valid_data: 검증 데이터셋
    :param test_data: 테스트 데이터셋
    :param gls_vocab: Gloss 어휘 사전
    :param logging_function: 로깅 함수
    """
    logging_function(
        "데이터셋 크기: \n\t학습 데이터: {:d}개,\n\t검증 데이터: {:d}개,\n\t테스트 데이터: {:d}개".format(
            len(train_data),
            len(valid_data),
            len(test_data) if test_data is not None else 0,
        )
    )

    """ logging_function(
        "First training example:\n\t[GLS] {}\n\t[TXT] {}".format(
            " ".join(vars(train_data[0])["gls"]), " ".join(vars(train_data[0])["txt"])
        )
    ) """

    logging_function(
        "첫 10개 Gloss 단어 (색인, 단어): {}".format(
            " ".join("(%d) %s" % (i, t) for i, t in enumerate(gls_vocab.itos[:10]))
        )
    )

    logging_function("고유 Gloss 단어 수 (타입 수): {}".format(len(gls_vocab)))


def load_config(path="configs/default.yaml") -> dict:
    """
    YAML 설정 파일을 로드하고 파싱합니다.

    :param path: YAML 설정 파일 경로
    :return: 설정 정보를 담은 딕셔너리
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    주어진 디렉토리에서 가장 최근의 체크포인트 파일(시간 기준)을 반환합니다.
    체크포인트가 없으면 None을 반환합니다.

    :param ckpt_dir: 체크포인트 파일이 있는 디렉토리 경로
    :return: 가장 최근의 체크포인트 파일 경로 또는 None
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    저장된 체크포인트로부터 모델을 로드합니다.

    :param path: 체크포인트 파일 경로
    :param use_cuda: CUDA 사용 여부
    :return: 체크포인트 정보를 담은 딕셔너리
    """
    assert os.path.isfile(path), "체크포인트 파일 %s를 찾을 수 없습니다." % path
    checkpoint = torch.load(path, map_location="cuda" if use_cuda else "cpu")
    return checkpoint


# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    주어진 차원 `dim`을 따라 텐서 `x`를 `count` 횟수만큼 타일링(반복)합니다.
    (OpenNMT에서 가져옴. 빔 검색에 사용됨)

    :param x: 타일링할 텐서
    :param count: 타일링 횟수
    :param dim: 타일링을 수행할 차원
    :return: 타일링된 텐서
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = (
        x.view(batch, -1)
        .transpose(0, 1)
        .repeat(count, 1)
        .transpose(0, 1)
        .contiguous()
        .view(*out_size)
    )
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module):
    """
    주어진 모듈의 파라미터를 고정(freeze)합니다.
    즉, 학습 중에 업데이트되지 않도록 합니다.

    :param module: 파라미터를 고정할 모듈
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    """ 심볼릭 링크를 생성하거나 업데이트합니다. 파일이 이미 존재하면 삭제 후 다시 생성합니다. """
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e
