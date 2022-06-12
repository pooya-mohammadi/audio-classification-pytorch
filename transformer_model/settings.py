import torch


class BasicConfig:
    train_json_path = "/home/ai/projects/speech/audio_classification/data/train_gender.json"
    test_json_path = "/home/ai/projects/speech/audio_classification/data/test_gender.json"
    output_dir = "output"
    file_name = "best"
    label2name = {0: "female", 1: "male"}


class Config(BasicConfig):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs = 50
    batch_size = 32
    sample_rate = 8_000
    num_classes = 2
    classifier_output = 2
    feature_size = 40
    hidden_size = 128
    num_layers = 1
    dropout = 0.1
    bidirectional = False
    lr = 1e-4
    lr_reduce_factor = 0.5
    lr_patience = 10
    n_workers = 0
    pin_memory = True


Config.pin_memory = False if Config.device == "cpu" else True
