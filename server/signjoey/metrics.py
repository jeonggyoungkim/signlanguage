# coding: utf-8
"""
이 모듈은 단어 분류 모델의 성능 평가를 위한 메트릭을 포함합니다.
"""

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import numpy as np

def get_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average_method: str = "weighted", # 'micro', 'macro', 'weighted', 'samples' 또는 None(클래스별 반환)
    labels: list = None, # 고려할 레이블 목록 (None이면 모든 레이블 고려)
    zero_division: int = 0 # precision, recall, F-score에서 0으로 나누는 경우 반환할 값 (0 또는 1)
) -> dict:
    """
    분류 성능 지표 (정확도, 정밀도, 재현율, F1 점수)를 계산합니다.

    :param y_true: 실제 레이블 (1D 배열)
    :param y_pred: 모델 예측 레이블 (1D 배열)
    :param average_method: 다중 클래스에 대한 평균화 방법.
                           'micro': 전체 TP, FP, FN을 집계하여 전역적으로 계산.
                           'macro': 각 클래스에 대한 메트릭을 계산하고 가중치 없이 평균.
                           'weighted': 각 클래스에 대한 메트릭을 계산하고 클래스별 샘플 수로 가중 평균.
                           'samples': 다중 레이블 문제에만 적용.
                           None: 각 클래스에 대한 점수를 반환.
    :param labels: 메트릭에 포함할 레이블 목록. None이면 y_true와 y_pred에 나타나는 모든 레이블.
    :param zero_division: 정밀도, 재현율, F-점수가 0으로 나누어지는 경우 반환할 값.
                           0이면 해당 메트릭이 0으로 설정되고, 1이면 1로 설정됨.
                           "warn"으로 설정하면 경고 발생 후 0 반환.
    :return: 정확도, 정밀도, 재현율, F1 점수를 포함하는 딕셔셔리.
             average_method가 None이 아니면 각 메트릭은 스칼라 값입니다.
             average_method가 None이면 각 메트릭은 클래스별 값의 배열입니다.
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)

    accuracy = accuracy_score(y_true, y_pred)
    
    # precision_recall_fscore_support는 (precision, recall, fbeta_score, support) 튜플을 반환합니다.
    # support는 average=None일 때만 의미가 있습니다 (클래스별 실제 샘플 수).
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, 
        y_pred, 
        average=average_method, 
        labels=labels, 
        zero_division=zero_division,
        warn_for=() # 경고 억제 (zero_division으로 처리하므로)
    )
    
    metrics = {
        f"accuracy": accuracy,
        f"{average_method if average_method else 'classwise'}_precision": precision,
        f"{average_method if average_method else 'classwise'}_recall": recall,
        f"{average_method if average_method else 'classwise'}_f1": f1,
    }

    # average_method가 None이 아니면 support는 None입니다.
    # 필요하다면, average=None으로 호출하여 클래스별 support를 얻을 수 있습니다.
    if average_method is None and support is not None:
        metrics["classwise_support"] = support
        
    return metrics

def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list = None # 혼동 행렬에 포함할 레이블 목록. None이면 모든 레이블.
) -> np.ndarray:
    """
    혼동 행렬(Confusion Matrix)을 계산합니다.

    :param y_true: 실제 레이블 (1D 배열)
    :param y_pred: 모델 예측 레이블 (1D 배열)
    :param labels: 혼동 행렬에 포함하고 정렬할 레이블 목록. None이면 y_true와 y_pred에 나타나는 모든 레이블 사용.
    :return: 혼동 행렬 (C[i, j]는 실제 클래스가 i이고 예측 클래스가 j인 샘플 수)
    """
    if not isinstance(y_true, np.ndarray):
        y_true = np.array(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.array(y_pred)
        
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return cm

# 예시 사용법 (테스트 또는 디버깅용)
if __name__ == '__main__':
    # 예시 데이터 (3개 클래스: 0, 1, 2)
    y_true_example = np.array([0, 1, 2, 0, 1, 2, 0, 0, 1, 2])
    y_pred_example = np.array([0, 2, 1, 0, 0, 1, 0, 1, 1, 2])
    
    print("--- Weighted Average Metrics ---")
    weighted_metrics = get_classification_metrics(y_true_example, y_pred_example, average_method="weighted")
    for metric_name, value in weighted_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\n--- Macro Average Metrics ---")
    macro_metrics = get_classification_metrics(y_true_example, y_pred_example, average_method="macro")
    for metric_name, value in macro_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\n--- Micro Average Metrics ---")
    micro_metrics = get_classification_metrics(y_true_example, y_pred_example, average_method="micro")
    for metric_name, value in micro_metrics.items():
        print(f"{metric_name}: {value:.4f}")

    print("\n--- Class-wise Metrics (average_method=None) ---")
    classwise_metrics = get_classification_metrics(y_true_example, y_pred_example, average_method=None, labels=[0, 1, 2])
    for metric_name, values in classwise_metrics.items():
        if isinstance(values, np.ndarray):
            for i, value in enumerate(values):
                print(f"Class {i} {metric_name}: {value:.4f}")
        else:
            print(f"{metric_name}: {values:.4f}") # 정확도는 스칼라

    print("\n--- Confusion Matrix ---")
    cm_example = compute_confusion_matrix(y_true_example, y_pred_example, labels=[0, 1, 2])
    print("Labels: [0, 1, 2]")
    print(cm_example)
