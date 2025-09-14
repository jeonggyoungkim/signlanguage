# coding: utf-8
"""
빌더 함수 모음
"""
from typing import Callable, Optional, Generator, Tuple

import torch
from torch import nn

# 학습률 스케줄러
from torch.optim import lr_scheduler

# 최적화 알고리즘
from torch.optim import Optimizer


def build_gradient_clipper(config: dict) -> Optional[Callable]:
    """
    설정에 지정된 대로 그래디언트 클리핑 함수를 정의합니다.
    지정되지 않은 경우 None을 반환합니다.

    현재 옵션:
        - "clip_grad_val": 그래디언트가 이 값을 초과하면 클리핑,
            `torch.nn.utils.clip_grad_value_` 참조
        - "clip_grad_norm": 그래디언트의 노름이 이 값을 초과하면 클리핑,
            `torch.nn.utils.clip_grad_norm_` 참조

    :param config: 학습 설정이 포함된 딕셔너리
    :return: 클리핑 함수(제자리 수정) 또는 그래디언트 클리핑이 없는 경우 None
    """
    clip_grad_fun = None
    if "clip_grad_val" in config.keys():
        clip_value = config["clip_grad_val"]
        clip_grad_fun = lambda params: nn.utils.clip_grad_value_(
            parameters=params, clip_value=clip_value
        )
    elif "clip_grad_norm" in config.keys():
        max_norm = config["clip_grad_norm"]
        clip_grad_fun = lambda params: nn.utils.clip_grad_norm_(
            parameters=params, max_norm=max_norm
        )

    if "clip_grad_val" in config.keys() and "clip_grad_norm" in config.keys():
        raise ValueError("clip_grad_val와 clip_grad_norm 중 하나만 지정할 수 있습니다.")

    return clip_grad_fun


def build_optimizer(config: dict, parameters) -> Optimizer:
    """
    설정에 지정된 대로 주어진 파라미터에 대한 최적화기를 생성합니다.

    가중치 감쇠(weight decay)와 초기 학습률을 제외하고는
    기본 최적화기 설정이 사용됩니다.

    현재 "optimizer"에 대해 지원되는 설정:
        - "sgd" (기본값): `torch.optim.SGD` 참조
        - "adam": `torch.optim.adam` 참조
        - "adagrad": `torch.optim.adagrad` 참조
        - "adadelta": `torch.optim.adadelta` 참조
        - "rmsprop": `torch.optim.RMSprop` 참조

    초기 학습률은 설정의 "learning_rate"에 따라 설정됩니다.
    가중치 감쇠는 설정의 "weight_decay"에 따라 설정됩니다.
    지정되지 않은 경우 초기 학습률은 3.0e-4로, 가중치 감쇠는 0으로 설정됩니다.

    스케줄러 상태는 체크포인트에 저장되므로, 추가 학습을 위해 모델을 로드할 때는
    동일한 유형의 스케줄러를 사용해야 합니다.

    :param config: 설정 딕셔너리
    :param parameters: 최적화할 파라미터
    :return: 최적화기
    """
    optimizer_name = config.get("optimizer", "radam").lower()
    learning_rate = config.get("learning_rate", 3.0e-4)
    weight_decay = config.get("weight_decay", 0)
    eps = config.get("eps", 1.0e-8)

    # Adam 기반 최적화기
    betas = config.get("betas", (0.9, 0.999))
    amsgrad = config.get("amsgrad", False)

    if optimizer_name == "adam":
        return torch.optim.Adam(
            params=parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_name == "adamw":
        return torch.optim.Adam(
            params=parameters,
            lr=learning_rate,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
        )
    elif optimizer_name == "adagrad":
        return torch.optim.Adagrad(
            params=parameters,
            lr=learning_rate,
            lr_decay=config.get("lr_decay", 0),
            weight_decay=weight_decay,
            eps=eps,
        )
    elif optimizer_name == "adadelta":
        return torch.optim.Adadelta(
            params=parameters,
            rho=config.get("rho", 0.9),
            eps=eps,
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "rmsprop":
        return torch.optim.RMSprop(
            params=parameters,
            lr=learning_rate,
            momentum=config.get("momentum", 0),
            alpha=config.get("alpha", 0.99),
            eps=eps,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "sgd":
        return torch.optim.SGD(
            params=parameters,
            lr=learning_rate,
            momentum=config.get("momentum", 0),
            weight_decay=weight_decay,
        )
    else:
        raise ValueError("알 수 없는 최적화기 {}.".format(optimizer_name))


def build_scheduler(
    config: dict, optimizer: Optimizer, scheduler_mode: str, hidden_size: int = 0
) -> Tuple[Optional[lr_scheduler._LRScheduler], Optional[str]]:
    """
    설정에 지정된 경우 학습률 스케줄러를 생성하고
    스케줄러 단계가 실행되어야 할 시점을 결정합니다.

    현재 옵션:
        - "plateau": `torch.optim.lr_scheduler.ReduceLROnPlateau` 참조
        - "decaying": `torch.optim.lr_scheduler.StepLR` 참조
        - "exponential": `torch.optim.lr_scheduler.ExponentialLR` 참조

    스케줄러가 지정되지 않은 경우 (None, None)을 반환하며,
    이는 일정한 학습률을 사용하게 됩니다.

    :param config: 학습 설정
    :param optimizer: 스케줄러의 최적화기, 스케줄러가 학습률을 설정할
        파라미터 집합을 결정합니다
    :param scheduler_mode: "min" 또는 "max", 검증 점수를 최소화할지
        최대화할지에 따라 결정됩니다.
        "plateau"에만 관련됩니다.
    :param hidden_size: 인코더 히든 크기
    :return:
        - scheduler: 스케줄러 객체
        - scheduler_step_at: "validation" 또는 "epoch"
    """
    scheduler_name = config["scheduling"].lower()

    if scheduler_name == "plateau":
        # 학습률 스케줄러
        return (
            lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer,
                mode=scheduler_mode,
                threshold_mode="abs",
                factor=config.get("decrease_factor", 0.1),
                patience=config.get("patience", 10),
            ),
            "validation",
        )
    elif scheduler_name == "cosineannealing":
        return (
            lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer,
                eta_min=config.get("eta_min", 0),
                T_max=config.get("t_max", 20),
            ),
            "epoch",
        )
    elif scheduler_name == "cosineannealingwarmrestarts":
        return (
            lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer=optimizer,
                T_0=config.get("t_init", 10),
                T_mult=config.get("t_mult", 2),
            ),
            "step",
        )
    elif scheduler_name == "decaying":
        return (
            lr_scheduler.StepLR(
                optimizer=optimizer,
                step_size=config.get("decay_steps", 1),
                gamma=config.get("decay_rate", 0.1),
            ),
            "epoch",
        )
    elif scheduler_name == "exponential":
        return (
            lr_scheduler.ExponentialLR(
                optimizer=optimizer,
                gamma=config.get("decay_rate", 0.99),
            ),
            "epoch",
        )
    else:
        raise ValueError("알 수 없는 스케줄러 {}.".format(scheduler_name))
