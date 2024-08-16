from os.path import split

import torch
from deep_utils import DirUtils


class BasicConfig:
    train_dataset_dir = "../sentiment_data/train"
    val_dataset_dir = "../sentiment_data/val"
    output_dir = "output"
    file_name = "best"


class Config(BasicConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 50
    batch_size = 32
    sample_rate = 8_000
    num_classes = None
    # classifier_output = 2
    feature_size = 40
    hidden_size = 128
    num_layers = 1
    dropout = 0.1
    bidirectional = False
    lr = 1e-4
    lr_reduce_factor = 0.5
    lr_patience = 10
    n_workers = 8
    pin_memory = True
    label2name = None

Config.pin_memory = False if Config.device == "cpu" else True
Config.num_classes = len(DirUtils.list_dir_full_path(BasicConfig.train_dataset_dir, only_directories=True))
Config.label2name = {index: split(name)[-1] for index, name in
                     enumerate(DirUtils.list_dir_full_path(Config.train_dataset_dir, only_directories=True))}

if __name__ == '__main__':
    print(vars(Config))
