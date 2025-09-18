#!/usr/bin/env python

import os

#os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import torch

torch.backends.cudnn.deterministic = True

import argparse
import numpy as np
import shutil
import time
import queue
import warnings

# sklearn과 numpy 경고 억제
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=FutureWarning, module='numpy')
warnings.filterwarnings('ignore', message='A single label was found*')
warnings.filterwarnings('ignore', message='elementwise comparison failed*')
warnings.filterwarnings('ignore', message='The default value of.*will change*')

from signjoey.model import build_model
from signjoey.batch import Batch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from signjoey.helpers import (
    log_data_info,
    load_config,
    log_cfg,
    load_checkpoint,
    make_model_dir,
    make_logger,
    set_seed,
    symlink_update,
)
from signjoey.model import SignModel
from signjoey.data import load_data, make_data_iter
from signjoey.builders import build_optimizer, build_scheduler, build_gradient_clipper
from signjoey.metrics import get_classification_metrics, compute_confusion_matrix
from signjoey.vocabulary import Vocabulary, UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SIL_TOKEN
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset
from typing import List, Dict, Tuple

import matplotlib
matplotlib.use('Agg')  # GUI 없이 사용
import matplotlib.pyplot as plt


def validate_on_data(model, data_iter, device, vocab_info):
    """
    다중 클래스 분류 모델의 검증을 수행합니다.
    
    :param model: 검증할 모델
    :param data_iter: 검증 데이터 이터레이터  
    :param device: 디바이스 (cuda/cpu)
    :param vocab_info: 어휘 정보
    :return: 검증 결과 딕셔너리
    """
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_labels = []
    total_samples = 0
    
    # GPU 사용 시 설정
    use_cuda = device.type == "cuda"
    
    with torch.no_grad():
        for batch_data in data_iter:
            # 배치 데이터 추출
            sgn = batch_data['sgn']
            txt = batch_data['txt']
            sgn_lengths = batch_data['sgn_lengths']
            
            # GPU 사용 시 강제로 모든 텐서를 GPU로 이동
            if use_cuda:
                sgn = sgn.cuda(non_blocking=True)
                txt = txt.cuda(non_blocking=True)
                sgn_lengths = sgn_lengths.cuda(non_blocking=True)
                
                # 이동 확인
                if not sgn.is_cuda:
                    sgn = sgn.to(device)
                if not txt.is_cuda:
                    txt = txt.to(device)
                if not sgn_lengths.is_cuda:
                    sgn_lengths = sgn_lengths.to(device)
            else:
                # CPU 사용 시
                sgn = sgn.cpu()
                txt = txt.cpu()
                sgn_lengths = sgn_lengths.cpu()
            
            # txt 차원 처리
            if txt.dim() > 1:
                txt = txt.squeeze(-1)
            
            # 마스크 생성 - 같은 디바이스에서
            batch_size, max_len = sgn.size(0), sgn.size(1)
            src_mask = torch.zeros(batch_size, 1, max_len, device=sgn.device)
            for i, length in enumerate(sgn_lengths):
                src_mask[i, 0, :length] = 1
            
            # 모델 순전파 - GPU에서 실행
            if use_cuda:
                with torch.cuda.device(device):
                    loss, logits = model(
                        src=sgn,
                        src_mask=src_mask,
                        src_length=sgn_lengths,
                        trg_labels=txt
                    )
            else:
                loss, logits = model(
                    src=sgn,
                    src_mask=src_mask,
                    src_length=sgn_lengths,
                    trg_labels=txt
                )
            
            if loss is not None:
                total_loss += loss.item() * sgn.size(0)
            
            # 예측값 계산 - GPU에서 수행 후 CPU로 이동
            predictions = torch.argmax(logits, dim=-1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(txt.cpu().numpy())
            total_samples += sgn.size(0)
    
    # 평균 손실 계산
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
    
    # 분류 메트릭 계산
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        accuracy = accuracy_score(all_labels, all_predictions)
        f1_weighted = f1_score(all_labels, all_predictions, average='weighted')
        f1_macro = f1_score(all_labels, all_predictions, average='macro')
        precision_weighted = precision_score(all_labels, all_predictions, average='weighted')
        recall_weighted = recall_score(all_labels, all_predictions, average='weighted')
    
    # 안전한 confusion matrix 생성
    try:
        # 실제 등장한 레이블만 사용
        unique_labels = sorted(list(set(all_labels + all_predictions)))
        # 최소 2개의 클래스가 있도록 보장 (confusion matrix 경고 방지)
        if len(unique_labels) < 2:
            # 가능한 모든 클래스 레이블을 포함
            max_label = max(max(all_labels), max(all_predictions)) if all_labels and all_predictions else 1
            unique_labels = list(range(max_label + 1))
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cm = confusion_matrix(all_labels, all_predictions, labels=unique_labels)
    except Exception as e:
        print(f"Warning: Could not generate confusion matrix: {e}")
        cm = None
    
    # 클래스별 예측 결과 생성
    # vocab_info에서 클래스 수 정보 가져오기
    num_classes = vocab_info.get('num_classes', len(vocab_info.get('class_names', [])))
    if num_classes == 0:
        num_classes = max(max(all_labels), max(all_predictions)) + 1 if all_labels and all_predictions else 2
    
    class_names = vocab_info.get('class_names', [f'class_{i}' for i in range(num_classes)])
    gls_ref = [class_names[label] if label < len(class_names) else f'class_{label}' for label in all_labels]
    gls_hyp = [class_names[pred] if pred < len(class_names) else f'class_{pred}' for pred in all_predictions]
    
    return {
        "valid_recognition_loss": avg_loss,
        "valid_scores": {
            "accuracy": accuracy,
            "f1_weighted": f1_weighted,
            "f1_macro": f1_macro,
            "precision_weighted": precision_weighted,
            "recall_weighted": recall_weighted,
            "wer": 1.0 - accuracy,  # WER 호환성을 위해 (1 - accuracy)로 설정
            "wer_scores": {
                "del_rate": 0.0,
                "ins_rate": 0.0,
                "sub_rate": 1.0 - accuracy
            }
        },
        "gls_ref": gls_ref,
        "gls_hyp": gls_hyp,
        "confusion_matrix": cm
    }


def wer_single(r, h):
    """
    Dummy WER calculation function - replace with actual implementation
    """
    # This is a placeholder function to prevent undefined variable errors
    return {
        "alignment_out": {
            "align_ref": r,
            "align_hyp": h,
            "alignment": ""
        }
    }


def save_loss(plot_file_path, loss_values):
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Recognition loss")
    plt.plot(loss_values)
    plt.savefig(plot_file_path)

# pylint: disable=too-many-instance-attributes
class TrainManager:
    """ Manages training loop, validations, learning rate scheduling
    and early stopping."""

    def __init__(self, model: SignModel, config: dict, trg_vocab: Vocabulary) -> None:
        """
        Creates a new TrainManager for a model, specified as in configuration.

        :param model: torch module defining the model
        :param config: dictionary containing the training configurations
        :param trg_vocab: target word label vocabulary
        """
        self.train_cfg = config["training"]
        self.model_cfg = config["model"]
        self.data_cfg = config["data"]
        self.trg_vocab = trg_vocab

        # files for logging and storing
        self.model_dir = make_model_dir(
            self.train_cfg["model_dir"], overwrite=self.train_cfg.get("overwrite", False)
        )
        self.logger = make_logger(model_dir=self.model_dir)
        self.logging_freq = self.train_cfg.get("logging_freq", 100)
        self.valid_report_file = "{}/validations.txt".format(self.model_dir)
        self.tb_writer = SummaryWriter(log_dir=self.model_dir + "/tensorboard/")

        # input
        self.feature_size = self.data_cfg["feature_size"]

        # model
        self.model = model
        # self._log_parameters_list() # 모델이 None일 수 있으므로 여기서 호출하지 않음
        
        # Add missing attributes
        self.do_recognition = self.train_cfg.get("do_recognition", True)
        self.recognition_loss_function = None
        self.recognition_loss_weight = self.train_cfg.get("recognition_loss_weight", 1.0)
        self.eval_recognition_beam_size = self.train_cfg.get("eval_recognition_beam_size", 1)
        self.tokeniser = None
        self.dataset_version = self.train_cfg.get("dataset_version", "phoenix_2014_trans")
        self.plot_file_path = self.train_cfg.get("plot_file_path", "loss_plot.png")
        
        # optimization
        self.learning_rate_min = self.train_cfg.get("learning_rate_min", 1.0e-8)
        self.clip_grad_fun = build_gradient_clipper(config=self.train_cfg)
        # Optimizer and Scheduler will be initialized later, after model is built
        self.optimizer = None
        self.scheduler = None
        self.scheduler_step_at = None
        self.batch_multiplier = self.train_cfg.get("batch_multiplier", 1)

        # Determine CUDA availability first
        self.use_cuda = self.train_cfg.get("use_cuda", False) and torch.cuda.is_available()

        # AMP (Automatic Mixed Precision) GradScaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_cuda)

        # validation & early stopping
        self.validation_freq = self.train_cfg.get("validation_freq", 100)
        self.num_valid_log = self.train_cfg.get("num_valid_log_examples", 5)
        self.ckpt_queue = queue.Queue(maxsize=self.train_cfg.get("keep_last_ckpts", 5))
        
        # Eval metric should now be accuracy or f1_weighted
        self.eval_metric = self.train_cfg.get("eval_metric", "accuracy").lower()
        available_metrics = ["accuracy", "f1_weighted", "f1_macro", "f1_micro", "precision_weighted", "recall_weighted", "loss"]
        if self.eval_metric not in available_metrics:
            raise ValueError(
                f"Invalid setting for 'eval_metric': {self.eval_metric} (must be one of {available_metrics})"
            )
        
        self.early_stopping_metric = self.train_cfg.get(
            "early_stopping_metric", self.eval_metric
        ).lower()
        if self.early_stopping_metric not in available_metrics:
             raise ValueError(
                f"Invalid setting for 'early_stopping_metric': {self.early_stopping_metric} (must be one of {available_metrics})"
            )

        # Metric direction for early stopping
        if self.early_stopping_metric == "loss":
            self.minimize_metric = True
        elif self.early_stopping_metric in ["accuracy", "f1_weighted", "f1_macro", "f1_micro", "precision_weighted", "recall_weighted"]:
            self.minimize_metric = False
        else:
            raise ValueError(
                "Invalid setting for 'early_stopping_metric': {}".format(
                    self.early_stopping_metric
                )
            )

        # data_augmentation parameters
        self.frame_subsampling_ratio = self.data_cfg.get(
            "frame_subsampling_ratio", None
        )
        self.random_frame_subsampling = self.data_cfg.get(
            "random_frame_subsampling", None
        )
        self.random_frame_masking_ratio = self.data_cfg.get(
            "random_frame_masking_ratio", None
        )

        # learning rate scheduling
        # Scheduler initialization moved to _initialize_optimizer_and_scheduler
        # scheduler_hidden_size = self.model_cfg["encoder"]["hidden_size"] if "encoder" in self.model_cfg and "hidden_size" in self.model_cfg["encoder"] else None
        # self.scheduler, self.scheduler_step_at = build_scheduler(
        #     config=self.train_cfg,
        #     scheduler_mode="min" if self.minimize_metric else "max",
        #     optimizer=self.optimizer, # This would be None here
        #     hidden_size=scheduler_hidden_size,
        # )

        # data & batch handling
        self.level = self.data_cfg.get("level", "word")

        self.shuffle = self.train_cfg.get("shuffle", True)
        self.epochs = self.train_cfg["epochs"]
        self.batch_size = self.train_cfg["batch_size"]
        self.batch_type = self.train_cfg.get("batch_type", "sentence")
        self.eval_batch_size = self.train_cfg.get("eval_batch_size", self.batch_size)
        self.eval_batch_type = self.train_cfg.get("eval_batch_type", self.batch_type)

        if self.use_cuda:
            self.device = torch.device("cuda")
            # GPU 메모리 최적화 설정
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True
            
            # self.model.cuda() # 모델이 None일 수 있으므로 여기서 호출하지 않음
            # # 모델의 모든 파라미터가 GPU에 있는지 확인 # 모델이 None일 수 있으므로 여기서 호출하지 않음
            # for param in self.model.parameters(): # 모델이 None일 수 있으므로 여기서 호출하지 않음
            #     if not param.is_cuda:
            #         param.data = param.data.cuda()
            
            self.logger.info(f"CUDA enabled - Using device: {self.device}")
            self.logger.info(f"GPU device name: {torch.cuda.get_device_name(0)}")
            self.logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
            self.logger.info(f"GPU memory cached: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
        else:
            self.device = torch.device("cpu")
            self.logger.info(f"CUDA disabled - Using device: {self.device}")
            if self.train_cfg["use_cuda"]:
                self.logger.warning("use_cuda is True but CUDA is not available!")
            
        # Verify model is on correct device - Moved to train function, after model is built
        # model_device = next(self.model.parameters()).device
        # self.logger.info(f"Model is on device: {model_device}")
        # if str(model_device) != str(self.device):
        #     self.logger.warning(f"Device mismatch! Expected: {self.device}, Actual: {model_device}")
        #     # 강제로 모델을 올바른 디바이스로 이동
        #     if self.use_cuda:
        #         self.model.cuda()
        #         self.logger.info("Forced model to GPU")
        #     else:
        #         self.model.cpu()
        #         self.logger.info("Forced model to CPU")

        # initialize training statistics
        self.steps = 0
        self.total_steps = 0
        self.stop = False
        self.total_samples_processed = 0
        self.best_ckpt_iteration = 0
        self.best_ckpt_score = float('inf') if self.minimize_metric else float('-inf')
        self.best_valid_metrics = {}
        self.is_best = (
            lambda score: score < self.best_ckpt_score
            if self.minimize_metric
            else score > self.best_ckpt_score
        )

        # model parameters
        # Checkpoint loading will be handled in _initialize_optimizer_and_scheduler
        # if "load_model" in self.train_cfg.keys():
        #     model_load_path = self.train_cfg["load_model"]
        #     self.logger.info("Loading model from %s", model_load_path)
        #     reset_best_ckpt = self.train_cfg.get("reset_best_ckpt", False)
        #     reset_scheduler = self.train_cfg.get("reset_scheduler", False)
        #     reset_optimizer = self.train_cfg.get("reset_optimizer", False)
        #     self.init_from_checkpoint(
        #         model_load_path,
        #         reset_best_ckpt=reset_best_ckpt,
        #         reset_scheduler=reset_scheduler,
        #         reset_optimizer=reset_optimizer,
        #     )

    def _initialize_optimizer_and_scheduler(self):
        """Initializes optimizer, scheduler, and loads checkpoint if specified."""
        if self.model is None:
            self.logger.error("Model is not initialized before optimizer/scheduler setup.")
            raise ValueError("Model must be set before initializing optimizer and scheduler.")

        self.optimizer = build_optimizer(
            config=self.train_cfg,
            parameters=self.model.parameters()
        )

        # Load checkpoint after optimizer and scheduler are initialized
        if "load_model" in self.train_cfg.keys():
            model_load_path = self.train_cfg["load_model"]
            self.logger.info("Loading model, optimizer, and scheduler states from %s", model_load_path)
            reset_best_ckpt = self.train_cfg.get("reset_best_ckpt", False)
            reset_scheduler = self.train_cfg.get("reset_scheduler", False)
            reset_optimizer = self.train_cfg.get("reset_optimizer", False)
            self.init_from_checkpoint(
                model_load_path,
                reset_best_ckpt=reset_best_ckpt,
                reset_scheduler=reset_scheduler,
                reset_optimizer=reset_optimizer,
            )

    def _save_checkpoint(self, epoch_no, is_best: bool = False) -> None:
        """
        Save the model's current parameters and the training state to a
        checkpoint.
        """
        model_path = "{}/{}.ckpt".format(self.model_dir, self.steps)
        state = {
            "total_steps": self.steps,
            "total_samples_processed": self.total_samples_processed,
            "best_ckpt_score": self.best_ckpt_score,
            "best_valid_metrics": self.best_valid_metrics,
            "best_ckpt_iteration": self.best_ckpt_iteration,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
            "trg_vocab": self.trg_vocab
        }
        torch.save(state, model_path)
        if self.ckpt_queue.full():
            to_delete = self.ckpt_queue.get()
            try:
                if os.path.exists(to_delete):
                    os.remove(to_delete)
            except FileNotFoundError:
                self.logger.warning(
                    "Wanted to delete old checkpoint %s but " "file does not exist.",
                    to_delete,
                )

        self.ckpt_queue.put(model_path)

        if is_best:
            symlink_update(
                os.path.basename(model_path), "{}/best.ckpt".format(self.model_dir)
            )

    def init_from_checkpoint(
        self,
        path: str,
        reset_best_ckpt: bool = False,
        reset_scheduler: bool = False,
        reset_optimizer: bool = False,
    ) -> None:
        """
        Initialize the trainer from a given checkpoint file.
        """
        model_checkpoint = load_checkpoint(path=path, use_cuda=self.use_cuda)

        self.model.load_state_dict(model_checkpoint["model_state"])

        if not reset_optimizer:
            if "optimizer_state" in model_checkpoint and model_checkpoint["optimizer_state"] is not None:
                self.optimizer.load_state_dict(model_checkpoint["optimizer_state"])
            else:
                self.logger.warning("Checkpoint does not contain optimizer state.")

        if not reset_scheduler:
            if "scheduler_state" in model_checkpoint and model_checkpoint["scheduler_state"] is not None and self.scheduler is not None:
                self.scheduler.load_state_dict(model_checkpoint["scheduler_state"])
            elif self.scheduler is not None:
                self.logger.warning("Checkpoint does not contain scheduler state or current scheduler is None.")
        
        self.steps = model_checkpoint.get("total_steps", self.steps)
        
        if not reset_best_ckpt:
            self.best_ckpt_score = model_checkpoint.get("best_ckpt_score", self.best_ckpt_score)
            self.best_valid_metrics = model_checkpoint.get("best_valid_metrics", self.best_valid_metrics)
            self.best_ckpt_iteration = model_checkpoint.get("best_ckpt_iteration", self.best_ckpt_iteration)
        
        self.logger.info("Checkpoint loaded: %s", path)

        # Checkpoint loading
        self.trg_vocab = model_checkpoint["trg_vocab"]
        if "trg_vocab" in model_checkpoint:
            self.trg_vocab = model_checkpoint["trg_vocab"]
        elif "gls_vocab" in model_checkpoint:
            self.trg_vocab = model_checkpoint["gls_vocab"]
            self.logger.info("Loaded 'gls_vocab' as 'trg_vocab' from checkpoint.")
        else:
            self.logger.warning(
                "Checkpoint does not contain 'trg_vocab' or 'gls_vocab'. "
                "Using a new vocabulary may lead to unexpected behavior if model architecture depends on it."
            )

        # Log the loaded vocabulary details
        if self.trg_vocab:
            self.logger.info(
                f"Loaded target vocabulary with {len(self.trg_vocab)} items. "
                f"First 5 items: {self.trg_vocab.itos[:5]}"
            )

        # Check if the loaded vocabulary size matches the model's output layer if model is already built
        # This check might be more relevant after the model is built or if its structure depends on vocab size at init
        if hasattr(self.model, 'output_layer') and self.model.output_layer is not None:
            expected_num_classes = len(self.trg_vocab)
            if self.model.output_layer.out_features != expected_num_classes:
                self.logger.warning(
                    f"Mismatch between loaded vocabulary size ({expected_num_classes}) "
                    f"and model output layer size ({self.model.output_layer.out_features}). "
                    f"This might cause issues. Consider re-building the model or ensuring vocab consistency."
                )

    def train_and_validate(self, train_iter, valid_iter, vocab_info) -> None:
        """
        Full training logic including validation, early stopping and lr scheduling.

        :param train_data: Training data
        :param valid_data: Validation data
        :param vocab_info: Vocabulary and other info from data loading
        """
        # Training starts
        self.logger.info(
            "Train stats: training examples: %d, validation examples: %d",
            len(train_iter.dataset),
            len(valid_iter.dataset),
        )
        self.logger.info(
            "Model info: %.2fM params", sum(p.numel() for p in self.model.parameters()) / 1000000.0
        )

        for epoch_no in range(self.epochs):
            self.logger.info("EPOCH %d", epoch_no)

            # Reset epoch statistics
            start_time = time.time()
            epoch_recognition_loss = 0.0
            epoch_correct_predictions = 0
            epoch_total_samples = 0
            
            self.model.train()
            # Iterate over training batches
            for i, batch_data in enumerate(train_iter):
                # 배치 데이터 추출
                sgn = batch_data['sgn']
                txt = batch_data['txt']
                sgn_lengths = batch_data['sgn_lengths']
                
                # GPU 사용 시 강제로 모든 텐서를 GPU로 이동
                if self.use_cuda:
                    sgn = sgn.cuda(non_blocking=True)
                    txt = txt.cuda(non_blocking=True)
                    sgn_lengths = sgn_lengths.cuda(non_blocking=True)
                    
                    # 이동 확인 및 강제 이동
                    if not sgn.is_cuda:
                        sgn = sgn.to(self.device)
                    if not txt.is_cuda:
                        txt = txt.to(self.device)
                    if not sgn_lengths.is_cuda:
                        sgn_lengths = sgn_lengths.to(self.device)
                else:
                    # CPU 사용 시
                    sgn = sgn.cpu()
                    txt = txt.cpu()
                    sgn_lengths = sgn_lengths.cpu()
                
                # txt 차원 처리
                if txt.dim() > 1:
                    txt = txt.squeeze(-1)

                # 첫 번째 배치에서 디바이스 확인
                if i == 0:
                    self.logger.info(f"Batch tensors - sgn device: {sgn.device}, txt device: {txt.device}, lengths device: {sgn_lengths.device}")
                    if self.use_cuda:
                        self.logger.info(f"Current GPU memory usage: {torch.cuda.memory_allocated(0) / 1024**3:.3f} GB")

                # 마스크 생성 - 같은 디바이스에서
                batch_size, max_len = sgn.size(0), sgn.size(1)
                src_mask = torch.zeros(batch_size, 1, max_len, device=sgn.device)
                for idx, length in enumerate(sgn_lengths):
                    src_mask[idx, 0, :length] = 1
                
                # Train batch
                batch_loss, batch_acc = self._train_batch_multi_class(
                    sgn, txt, sgn_lengths, src_mask, update=True # 항상 업데이트 시도, 내부에서 batch_multiplier로 조절
                )

                # Accumulate epoch statistics
                epoch_recognition_loss += batch_loss * txt.size(0) # 배치 손실은 이미 평균이므로 샘플 수 곱함
                epoch_correct_predictions += batch_acc * txt.size(0) # 배치 정확도에 샘플 수 곱하여 정확한 예측 수 누적
                epoch_total_samples += txt.size(0)

                # Log batch loss periodically
                if self.steps % self.logging_freq == 0:
                    current_lr = self.optimizer.param_groups[0]["lr"]
                    elapsed_time = time.time() - start_time
                    self.tb_writer.add_scalar(
                        "train/train_batch_loss", batch_loss, self.steps
                    )
                    self.logger.info(
                        "Epoch %d, Batch %d/%d, Batch Loss: %.4f, LR: %.6f, Time: %.2fs",
                        epoch_no, i + 1, len(train_iter), batch_loss, current_lr, elapsed_time
                    )
            
            # Calculate average epoch training loss and accuracy
            avg_epoch_train_loss = epoch_recognition_loss / epoch_total_samples if epoch_total_samples > 0 else 0
            avg_epoch_train_accuracy = epoch_correct_predictions / epoch_total_samples if epoch_total_samples > 0 else 0

            self.logger.info(
                "Epoch %d: Train Loss: %.4f, Train Accuracy: %.4f, Time: %.2fs",
                epoch_no, avg_epoch_train_loss, avg_epoch_train_accuracy, time.time() - start_time
            )
            self.tb_writer.add_scalar("train/epoch_train_loss", avg_epoch_train_loss, epoch_no)
            self.tb_writer.add_scalar("train/epoch_train_accuracy", avg_epoch_train_accuracy, epoch_no)

            # Validate and log validation results
            if (epoch_no + 1) % self.validation_freq == 0:
                self.logger.info("Validating at epoch %d...", epoch_no)
                val_start_time = time.time()
                
                validation_results = validate_on_data(
                    model=self.model,
                    data_iter=valid_iter,
                    device=self.device,
                    vocab_info=vocab_info
                )
                
                val_loss = validation_results["valid_recognition_loss"]
                val_accuracy = validation_results["valid_scores"]["accuracy"]
                val_f1_weighted = validation_results["valid_scores"]["f1_weighted"]
                cm = validation_results.get("confusion_matrix") # 혼동 행렬 가져오기
                
                self.logger.info(
                    "Epoch %d: Validation Loss: %.4f, Validation Accuracy: %.4f, Validation F1 (Weighted): %.4f, Time: %.2fs",
                    epoch_no, val_loss, val_accuracy, val_f1_weighted, time.time() - val_start_time
                )
                
                self.tb_writer.add_scalar("validation/val_loss", val_loss, epoch_no)
                self.tb_writer.add_scalar("validation/val_accuracy", val_accuracy, epoch_no)
                self.tb_writer.add_scalar("validation/val_f1_weighted", val_f1_weighted, epoch_no)

                # classification_report 로깅 (상세)
                if "gls_ref" in validation_results and "gls_hyp" in validation_results:
                    # target_names을 self.data_cfg에서 가져오도록 수정
                    target_names_for_report = self.data_cfg.get('class_names')
                    # 만약 class_names가 설정에 없다면, trg_vocab에서 특수토큰 제외하고 가져오기
                    if not target_names_for_report:
                        target_names_for_report = [
                            name for name in self.trg_vocab.itos 
                            if name not in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SIL_TOKEN]
                        ]
                        # 모델의 num_classes만큼만 사용
                        target_names_for_report = target_names_for_report[:self.model.num_classes] 

                    # 실제 등장하는 모든 레이블을 수집
                    all_labels_set = set(validation_results["gls_ref"] + validation_results["gls_hyp"])
                    
                    # 실제 사용된 레이블만 포함 (숫자 인덱스로 변환)
                    if target_names_for_report:
                        # 클래스 이름을 인덱스로 매핑
                        name_to_idx = {name: idx for idx, name in enumerate(target_names_for_report)}
                        used_indices = []
                        used_names = []
                        
                        for label in all_labels_set:
                            if isinstance(label, str) and label in name_to_idx:
                                idx = name_to_idx[label]
                                if idx not in used_indices:
                                    used_indices.append(idx)
                                    used_names.append(label)
                            elif isinstance(label, int) and 0 <= label < len(target_names_for_report):
                                if label not in used_indices:
                                    used_indices.append(label)
                                    used_names.append(target_names_for_report[label])
                        
                        used_indices.sort()
                        used_names = [target_names_for_report[i] for i in used_indices]
                    else:
                        # target_names가 없으면 숫자 레이블 사용
                        used_indices = sorted([i for i in all_labels_set if isinstance(i, int)])
                        used_names = [f"class_{i}" for i in used_indices]
                    
                    try:
                        # 실제 사용된 레이블과 이름만으로 리포트 생성
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            report_str = classification_report(
                                validation_results["gls_ref"],
                                validation_results["gls_hyp"],
                                labels=used_indices,  # 실제 사용된 레이블만
                                target_names=used_names,  # 실제 사용된 이름만
                                zero_division=0
                            )
                        self.logger.info("Validation Classification Report:\n%s", report_str)
                        self.tb_writer.add_text("validation/classification_report", report_str, epoch_no)
                    except Exception as e:
                        self.logger.warning(f"Could not generate classification report: {e}")
                        # 간단한 정확도 리포트만 출력
                        unique_refs = len(set(validation_results["gls_ref"]))
                        unique_hyps = len(set(validation_results["gls_hyp"]))
                        self.logger.info(f"Validation: Unique reference classes: {unique_refs}, Unique predicted classes: {unique_hyps}")

                # 혼동 행렬 로깅
                if cm is not None:
                    try:
                        self.logger.info("Validation Confusion Matrix:\n%s", cm)
                        # TensorBoard에 혼동 행렬을 텍스트로 로깅
                        cm_str = np.array2string(cm, separator=', ')
                        self.tb_writer.add_text("validation/confusion_matrix", cm_str, epoch_no)
                    except Exception as e:
                        self.logger.warning(f"Could not log confusion matrix: {e}")

                # Update learning rate scheduler
                if self.scheduler is not None:
                    metric_for_scheduler = val_accuracy if self.eval_metric == "accuracy" else val_loss
                    self.scheduler.step(metric_for_scheduler)
                
                # Save checkpoint if new best model
                current_valid_metric = val_accuracy if self.early_stopping_metric == "accuracy" else val_loss
                
                is_best = False
                if self.minimize_metric:
                    if current_valid_metric < self.best_ckpt_score:
                        self.best_ckpt_score = current_valid_metric
                        is_best = True
                else: # Maximize metric (e.g., accuracy)
                    if current_valid_metric > self.best_ckpt_score:
                        self.best_ckpt_score = current_valid_metric
                        is_best = True
                
                if is_best:
                    self.best_ckpt_iteration = self.steps
                    self.best_valid_metrics = validation_results["valid_scores"]
                    self.logger.info(
                        "New best validation score: %.4f at step %d",
                        self.best_ckpt_score,
                        self.steps,
                    )

                self._save_checkpoint(epoch_no, is_best=is_best)

                # if self.early_stopping.step(current_valid_metric): # EarlyStopping 로직 주석 처리
                #     self.logger.info("Early stopping triggered.")
                #     break  # Stop training
            
            # 에폭 종료 후 항상 최신 체크포인트 저장 (선택적, 필요에 따라 주석 처리)
            # self._save_checkpoint(epoch_no, is_best=False)

        self.logger.info("Finished training.")
        self.tb_writer.close()

    def _train_batch(self, batch: Batch, update: bool = True) -> Tensor:
        """
        Train the model on one batch: Compute the loss, make a gradient step.

        :param batch: training batch
        :param update: if False, only store gradient. if True also make update
        :return normalized_recognition_loss: Normalized recognition loss
        """

        recognition_loss = self.model.get_loss_for_batch(
            batch=batch,
            recognition_loss_function=self.recognition_loss_function
            if self.do_recognition else None,
            recognition_loss_weight=self.recognition_loss_weight
            if self.do_recognition else None,
        )

        if self.do_recognition:
            normalized_recognition_loss = recognition_loss / self.batch_multiplier if self.batch_multiplier > 0 else recognition_loss
        else:
            normalized_recognition_loss = 0

        total_loss = normalized_recognition_loss
        
        if isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            total_loss.backward()
        elif not isinstance(total_loss, torch.Tensor):
            pass
        

        if self.clip_grad_fun is not None and isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            self.clip_grad_fun(params=self.model.parameters())

        if update and isinstance(total_loss, torch.Tensor) and total_loss.requires_grad:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.steps += 1

        if self.do_recognition:
            self.total_samples_processed += batch.num_gls_tokens

        return normalized_recognition_loss

    def _train_batch_multi_class(self, sgn, txt, sgn_lengths, src_mask, update: bool = True) -> Tuple[float, float]:
        """
        다중 클래스 분류를 위한 단일 배치 학습을 수행합니다.
        손실과 정확도를 반환합니다.
        """
        # 모델을 학습 모드로 설정
        self.model.train() # 확실하게 train 모드로 설정
        self.logger.debug(f"Model training mode in _train_batch_multi_class: {self.model.training}") # 학습 모드 확인 로그

        # GPU 사용 시 모든 텐서를 GPU로 강제 이동
        if self.use_cuda:
            sgn = sgn.cuda(non_blocking=True)
            txt = txt.cuda(non_blocking=True)
            sgn_lengths = sgn_lengths.cuda(non_blocking=True)
            src_mask = src_mask.cuda(non_blocking=True)
            
            # 디바이스 확인
            if not sgn.is_cuda or not txt.is_cuda:
                self.logger.warning("Tensors not on GPU after cuda() call!")
                sgn = sgn.to(self.device)
                txt = txt.to(self.device)
                sgn_lengths = sgn_lengths.to(self.device)
                src_mask = src_mask.to(self.device)

        # 옵티마이저 그래디언트 초기화 (batch_multiplier 고려)
        if self.steps % self.batch_multiplier == 0:
            self.optimizer.zero_grad(set_to_none=True) # set_to_none=True for potential performance improvement

        # 모델 순전파 - GPU에서 실행되도록 보장
        with torch.amp.autocast(device_type='cuda', enabled=self.use_cuda, dtype=torch.float16 if self.use_cuda else torch.float32):
            loss, logits = self.model(
                src=sgn,
                src_mask=src_mask,
                src_length=sgn_lengths,
                trg_labels=txt
            )

        # 첫 번째 스텝에서 모델 출력 디바이스 확인
        if self.steps == 0:
            self.logger.info(f"Model outputs - loss device: {loss.device if hasattr(loss, 'device') else 'scalar'}, logits device: {logits.device}")

        # 손실 스케일링 (batch_multiplier 사용 시)
        if self.batch_multiplier > 1:
            loss = loss / self.batch_multiplier
        
        # 역전파 - GPU에서 실행 (GradScaler 사용)
        self.scaler.scale(loss).backward()

        # 그래디언트 클리핑 (옵티마이저 스텝 전, unscale 후)
        if self.clip_grad_fun is not None:
            self.scaler.unscale_(self.optimizer) # Unscale a.k.a zero inf/NaNs for clipping
            self.clip_grad_fun(self.model.parameters())

        # 옵티마이저 스텝 (batch_multiplier 고려, GradScaler 사용)
        if update and self.steps % self.batch_multiplier == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.steps += 1 # 옵티마이저 스텝이 실제로 일어났을 때만 증가

        # 정확도 계산 - GPU에서 수행
        with torch.no_grad():
            predictions = torch.argmax(logits, dim=-1)
            correct_predictions = (predictions == txt).sum().item()
            batch_accuracy = correct_predictions / txt.size(0)

        for name, param in self.model.named_parameters():
            self.logger.info(f"Parameter: {name}, requires_grad: {param.requires_grad}")
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]

        return loss.item(), batch_accuracy

    def _add_report(
        self,
        valid_scores: Dict,
        valid_recognition_loss: float,
        eval_metric: str,
        new_best: bool = False,
    ) -> None:
        """
        Append a one-line report to validation logging file.
        """
        current_lr = -1
        for param_group in self.optimizer.param_groups:
            current_lr = param_group["lr"]

        if new_best:
            self.learning_rate_min = current_lr

        if current_lr < self.learning_rate_min:
            self.stop = True

        with open(self.valid_report_file, "a", encoding="utf-8") as opened_file:
            opened_file.write(
                "Steps: {}\t"
                "Recognition Loss: {:.5f}\t"
                "Eval Metric: {}\t"
                "WER {:.2f}\t(DEL: {:.2f},\tINS: {:.2f},\tSUB: {:.2f})\t"
                "LR: {:.8f}\t{}\n".format(
                    self.steps,
                    valid_recognition_loss if self.do_recognition else -1,
                    eval_metric,
                    valid_scores["wer"] if self.do_recognition else -1,
                    valid_scores.get("wer_scores", {}).get("del_rate", -1) if self.do_recognition else -1,
                    valid_scores.get("wer_scores", {}).get("ins_rate", -1) if self.do_recognition else -1,
                    valid_scores.get("wer_scores", {}).get("sub_rate", -1) if self.do_recognition else -1,
                    current_lr,
                    "*" if new_best else "",
                )
            )

    def _log_parameters_list(self) -> None:
        """
        Write all model parameters (name, shape) to the log.
        """
        model_parameters = filter(lambda p: p.requires_grad, self.model.parameters())
        n_params = sum([np.prod(p.size()) for p in model_parameters])
        self.logger.info("Total params: %d", n_params)
        trainable_params = [
            n for (n, p) in self.model.named_parameters() if p.requires_grad
        ]
        self.logger.info("Trainable parameters: %s", sorted(trainable_params))
        assert trainable_params

    def _log_examples(
        self,
        sequences: List[str],
        gls_references: List[str],
        gls_hypotheses: List[str],
    ) -> None:
        """
        Log `self.num_valid_log` number of samples from valid.
        """

        if self.do_recognition:
            assert len(gls_references) == len(gls_hypotheses)
            num_sequences = len(gls_hypotheses)
        else:
            return
            
        if num_sequences == 0 or self.num_valid_log == 0:
            return
        actual_num_to_log = min(self.num_valid_log, num_sequences)
        rand_idx = np.sort(np.random.permutation(num_sequences)[:actual_num_to_log])
        
        self.logger.info("Logging Recognition Outputs")
        self.logger.info("=" * 120)
        for ri in rand_idx:
            self.logger.info("Logging Sequence: %s", sequences[ri])
            if self.do_recognition:
                gls_res = wer_single(r=gls_references[ri], h=gls_hypotheses[ri])
                self.logger.info(
                    "\tGloss Reference :\t%s", gls_res["alignment_out"]["align_ref"]
                )
                self.logger.info(
                    "\tGloss Hypothesis:\t%s", gls_res["alignment_out"]["align_hyp"]
                )
                self.logger.info(
                    "\tGloss Alignment :\t%s", gls_res["alignment_out"]["alignment"]
                )
            self.logger.info("=" * 120)

    def _store_outputs(
        self, tag: str, sequence_ids: List[str], hypotheses: List[str], sub_folder=None
    ) -> None:
        """
        Write current validation outputs to file in `self.model_dir.`
        """
        if sub_folder:
            out_folder = os.path.join(self.model_dir, sub_folder)
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            current_valid_output_file = "{}/{}.{}".format(out_folder, self.steps, tag)
        else:
            out_folder = self.model_dir
            current_valid_output_file = "{}/{}".format(out_folder, tag)

        with open(current_valid_output_file, "w", encoding="utf-8") as opened_file:
            for seq, hyp in zip(sequence_ids, hypotheses):
                opened_file.write("{}|{}\n".format(seq, hyp))

def train(cfg_file: str) -> None:
    """
    Main training function. After training, also test on test data if given.
    """
    cfg = load_config(cfg_file)
    print("--- Loaded cfg from config file: ---")
    print(cfg)
    print("-------------------------------------")

    set_seed(seed=cfg["training"].get("random_seed", 42))

    data_cfg = cfg["data"] # data 설정 부분을 변수로 미리 할당
    # Build model (trg_vocab을 먼저 얻기 위해 trainer 생성 전에 위치 변경 가능성 있음)
    # 또는, load_data에서 trg_vocab만 먼저 로드하고, trainer 생성 후 나머지를 로드하는 방식
    
    # Train manager를 먼저 생성하여 device 정보를 확립
    # 임시 trg_vocab (None 또는 기본값)으로 모델을 빌드하고, 실제 vocab 로드 후 모델 재설정 필요 가능성
    # 여기서는 일단 trainer 생성 후 device 정보를 가져와서 load_data에 넘기는 방식으로 진행
    
    # 임시 trg_vocab 생성 (모델 빌드 시 필요할 수 있음)
    # auto_discover 모드에 따라 trg_vocab 생성 방식이 달라지므로, 
    # 이 부분은 load_data 함수 내부에서 처리되도록 하고, 
    # build_model은 load_data 이후로 미루거나, trg_vocab을 None으로 전달 후 나중에 설정
    
    # Build model을 load_data 이후로 이동
    # model = build_model(cfg=cfg, trg_vocab=trg_vocab) # 이 줄을 아래로 이동

    # Train manager (trg_vocab 없이 초기화 후, load_data에서 얻은 vocab으로 설정)
    # 주의: TrainManager가 trg_vocab을 초기화 시점에 필수로 요구한다면, 이 구조 변경 필요
    # 현재 TrainManager는 trg_vocab을 필수로 받음.
    # 따라서, trg_vocab을 먼저 얻는 과정이 필요함.
    # 1. 설정에서 vocab 파일 경로 읽기
    # 2. auto_discover면 디렉토리 스캔해서 임시 vocab 만들기 (load_data 로직 일부 선실행)

    temp_trg_vocab_for_trainer = Vocabulary() # 임시 Vocabulary
    if data_cfg.get("auto_discover", False) and data_cfg.get("auto_generate_vocab", False):
        # auto_discover 모드면 data_path에서 클래스명 읽어와서 임시 vocab 구성
        class_dirs = []
        if os.path.exists(data_cfg["data_path"]):
            for item in os.listdir(data_cfg["data_path"]):
                item_path = os.path.join(data_cfg["data_path"], item)
                if os.path.isdir(item_path):
                    class_dirs.append(item)
        class_dirs.sort()
        if class_dirs:
            temp_trg_vocab_for_trainer._from_list(class_dirs)
        else:
            # 기본 어휘 추가 또는 오류 처리
            temp_trg_vocab_for_trainer._from_list(['<unk>']) # 예시
            print("Warning: No class directories found for temporary vocab generation. Using default.")
    elif data_cfg.get("trg_vocab_file"):
        # 파일 기반이면 해당 파일에서 로드 시도
        try:
            temp_trg_vocab_for_trainer._from_file(data_cfg["trg_vocab_file"])
        except FileNotFoundError:
            print(f"Warning: Vocab file {data_cfg['trg_vocab_file']} not found for temporary vocab. Using default.")
            temp_trg_vocab_for_trainer._from_list(['<unk>']) # 예시
    else: # Fallback
        temp_trg_vocab_for_trainer._from_list(['<unk>'])
        print("Warning: Could not determine temporary vocab source. Using default.")

    trainer = TrainManager(model=None, config=cfg, trg_vocab=temp_trg_vocab_for_trainer) # 모델은 나중에 빌드
    
    # 이제 trainer.device 정보를 사용할 수 있음
    current_device = str(trainer.device)
    
    print("--- data_cfg content: ---")
    print(data_cfg)
    print(f"'annotation_file' in data_cfg: {'annotation_file' in data_cfg}")
    print(f"'auto_discover' in data_cfg: {data_cfg.get('auto_discover', False)}")
    print(f"'auto_generate_vocab' in data_cfg: {data_cfg.get('auto_generate_vocab', False)}")
    print("-------------------------")

    # 데이터 로딩 분기 조건 수정: auto_discover나 annotation_file 존재 여부로 판단
    if data_cfg.get("auto_discover", False) or "annotation_file" in data_cfg:
        # 다중 클래스 분류 모드 (auto_discover 또는 annotation_file 기반)
        # load_data 호출 시 current_device 전달
        train_iter, test_iter, vocab_info = load_data(cfg=cfg, device=current_device)
        
        # load_data가 반환하는 vocab_info에서 실제 trg_vocab을 사용
        trg_vocab = vocab_info["txt_vocab"]
        
        # trainer의 vocab 업데이트 (만약 load_data에서 생성된 vocab과 다를 수 있다면)
        trainer.trg_vocab = trg_vocab
        
        print(f"Multi-class classification mode (auto_discover: {data_cfg.get('auto_discover', False)})")
        class_names_to_print = vocab_info.get('class_names', []) # vocab_info에서 class_names 가져오기 시도
        if not class_names_to_print and trg_vocab: # class_names가 없으면 trg_vocab에서 가져오기
            class_names_to_print = trg_vocab.itos
        
        # 특수 토큰 제외하고 출력
        class_names_to_print = [name for name in class_names_to_print if name not in [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, SIL_TOKEN]]
        print(f"Classes: {class_names_to_print}")
        print(f"Feature size: {vocab_info['feature_size']}")
        
        train_samples = len(train_iter.dataset) if hasattr(train_iter, 'dataset') and train_iter.dataset is not None else 'Unknown'
        test_samples = len(test_iter.dataset) if hasattr(test_iter, 'dataset') and test_iter.dataset is not None else 'Unknown'
        print(f"Training samples: {train_samples}")
        print(f"Test samples: {test_samples}")
        
    else:
        # 기존 단일 클래스 모드 (이 부분은 변경하지 않음)
        trg_vocab_file = data_cfg.get("trg_vocab_file", None)
        trg_max_size = data_cfg.get("trg_max_size", None)

        if not trg_vocab_file or not os.path.exists(trg_vocab_file):
            raise ValueError(
                f"Target vocabulary file 'data.trg_vocab_file' ({trg_vocab_file}) is not specified or does not exist. "
                f"This file is crucial for word classification."
            )
        
        specials_list = [UNK_TOKEN, PAD_TOKEN]
        trg_vocab = Vocabulary()
        trg_vocab.specials = specials_list # specials 속성을 먼저 설정
        trg_vocab._from_file(file=trg_vocab_file) # 인스턴스 메소드 _from_file 사용
        # TODO: max_size 로직은 Vocabulary 클래스에 직접 구현되어 있지 않으므로, 
        # build_vocab 함수와 동일하게 동작하려면 추가 구현 필요.
        # 현재는 AttributeError 해결에 집중.

        print(f"Target Vocabulary (from file: {trg_vocab_file}): {len(trg_vocab)} items.")
        if len(trg_vocab) < 20:
            print(f"Target Vocab items (first 20): {trg_vocab.itos[:20]}")

        # 기존 데이터 로딩 방식 (이 호출은 현재 signjoey/data.py의 load_data(cfg)와 호환되지 않을 수 있음)
        # 이 경로로 실행되지 않도록 위쪽 if 조건에서 현재 설정을 처리함.
        train_data, dev_data, test_data, _, _ = load_data(
            data_cfg=data_cfg, trg_vocab=trg_vocab
        )
        
        train_iter = make_data_iter(
            train_data,
            batch_size=cfg["training"]["batch_size"],
            batch_type=cfg["training"].get("batch_type", "sentence"),
            train=True,
            shuffle=cfg["training"].get("shuffle", True),
        )
        test_iter = make_data_iter(
            test_data,
            batch_size=cfg["training"].get("eval_batch_size", cfg["training"]["batch_size"]),
            batch_type=cfg["training"].get("eval_batch_type", cfg["training"].get("batch_type", "sentence")),
            train=False,
            shuffle=False,
        )
        
        vocab_info = {
            "feature_size": data_cfg.get("feature_size", 274),
            "class_names": [trg_vocab.itos[i] for i in range(len(trg_vocab))],
            "num_classes": len(trg_vocab),
            "txt_vocab": trg_vocab  # 추가: txt_vocab 정보도 포함
        }

    # Build model (실제 trg_vocab 사용)
    model = build_model(cfg=cfg, trg_vocab=trg_vocab)
    
    # Attempt to compile the model (PyTorch 2.0+ feature)
    if trainer.use_cuda and hasattr(torch, "compile"):
        print("Attempting to compile the model with torch.compile()...")
        try:
            # backend="inductor"는 기본값이며, Pytorch 2.0의 주요 백엔드입니다.
            # mode는 "default", "reduce-overhead", "max-autotune" 등을 시도해볼 수 있습니다.
            # "reduce-overhead"는 작은 모델이나 배치에서 오버헤드를 줄이는 데 유용할 수 있습니다.
            model = torch.compile(model, mode="reduce-overhead") 
            print("Model compiled successfully.")
        except Exception as e:
            print(f"Failed to compile model: {e}. Proceeding without compilation.")
            
    trainer.model = model # trainer에 모델 설정
    trainer._log_parameters_list() # 파라미터 로깅 (모델 설정 후)

    # 모델을 올바른 디바이스로 이동 (trainer 생성 시 model=None이었으므로, 여기서 직접 처리)
    if trainer.use_cuda:
        trainer.model.cuda()
        model_device_check = next(trainer.model.parameters()).device
        trainer.logger.info(f"Model explicitly moved to CUDA. Final model device: {model_device_check}")
    else:
        trainer.model.cpu()
        model_device_check = next(trainer.model.parameters()).device
        trainer.logger.info(f"Model explicitly moved to CPU. Final model device: {model_device_check}")

    # 옵티마이저, 스케줄러 및 체크포인트 로딩 초기화
    trainer._initialize_optimizer_and_scheduler()

    # GPU 사용 시 데이터셋을 GPU 디바이스로 다시 생성하는 로직 제거 또는 수정
    # -> 이미 load_data에서 device를 사용하므로 이 부분은 필요 없음

    shutil.copy2(cfg_file, trainer.model_dir + "/config.yaml")
    log_cfg(cfg, trainer.logger)

    trainer.logger.info(str(model))
    trainer.logger.info(f"Vocabulary info: {vocab_info}")

    trainer.train_and_validate(train_iter=train_iter, valid_iter=test_iter, vocab_info=vocab_info)

    ckpt = "{}/{}.ckpt".format(trainer.model_dir, trainer.best_ckpt_iteration)
    output_name = "best.IT_{:08d}_recognition_results".format(trainer.best_ckpt_iteration)
    output_path = os.path.join(trainer.model_dir, output_name)
    logger = trainer.logger
    del trainer

    logger.info(f"Training finished. Best model saved at {ckpt}. Recognition results would be in {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Joey-NMT")
    parser.add_argument(
        "config",
        default="configs/default.yaml",
        type=str,
        help="Training configuration file (yaml).",
    )
    parser.add_argument(
        "--gpu_id", type=str, default="0", help="gpu to run your job on"
    )
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    train(cfg_file=args.config)
