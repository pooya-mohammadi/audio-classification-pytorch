import os
from argparse import Namespace
from typing import Union
from os.path import join
from dataclasses import dataclass
from deep_utils import PyUtils, mkdir_incremental


@dataclass(init=True)
class BasicConfig:
    file_name = "best"

    random_state = 1486
    target_sampling_rate = 16_000  # audio sampling rate

    num_proc = 10  # number of processors used in dataset mapping...

    train_path = "../sentiment_data/train.csv"
    test_path = "../sentiment_data/val.csv"

    save_path = "results"

    best_path, label2id_path, model_path = None, None, None

    def basic_config_update(self):
        self.model_path = mkdir_incremental(self.save_path)
        self.best_path = join(self.model_path, self.file_name)
        os.makedirs(self.best_path, exist_ok=True)
        self.label2id_path = join(self.model_path, self.file_name, "label2id.pkl")


@dataclass(init=True)
class ModelConfig:
    feature_extractor = "facebook/wav2vec2-base"
    attention_dropout = 0.1
    hidden_dropout = 0.1
    feat_proj_dropout = 0.0
    mask_time_prob = 0.05
    layerdrop = 0.1
    ctc_loss_reduction = "mean"
    ctc_zero_infinity = True
    gradient_checkpointing = False


@dataclass(init=True)
class TrainConfig:
    num_train_epochs = 50
    warmup_steps = 100
    dataloader_num_workers = 8
    logging_steps = 1
    fp16 = True
    evaluation_strategy = "epoch"
    group_by_length = True
    per_device_train_batch_size = 64
    per_device_eval_batch_size = 64
    gradient_accumulation_steps = 1  # After every two steps, gradient flow will occur
    learning_rate = 5e-5
    min_learning_rate = 1e-6
    save_total_limit = 1
    report_to = "tensorboard"
    dataloader_pin_memory = True
    load_best_model_at_end = True
    metric_for_best_model = 'loss'
    early_stopping_patience = 5


class Config(BasicConfig, ModelConfig, TrainConfig):
    def __init__(self, args: Union[dict, Namespace, None] = None):
        args = dict() if args is None else args
        PyUtils.update_obj_params(self, args)
        self.basic_config_update()

    def __repr__(self):
        return PyUtils.variable_repr(self)
