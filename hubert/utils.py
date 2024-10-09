from transformers import HubertForSequenceClassification, Wav2Vec2FeatureExtractor
from jinja2 import Template
import torch
import yaml
import os


def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def load_config(config_path, context_path):
    context = load_yaml_file(context_path)
    with open(config_path, "r") as file:
        template = Template(file.read())

    rendered_yaml = template.render(context)
    config = yaml.safe_load(rendered_yaml)

    config["training_args"]["learning_rate"] = float(
        config["training_args"]["learning_rate"]
    )

    merged_config = {**config, **context}

    return merged_config


def load_model(config):
    model = HubertForSequenceClassification.from_pretrained(
        config["hubert_model_name"], num_labels=config["num_of_classes"]
    )
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        config["hubert_model_name"]
    )
    return model, feature_extractor


def preprocess_function(examples, feature_extractor):
    audio = [example["array"] for example in examples["file_path"]]
    labels = [example for example in examples["label"]]
    audio_features = feature_extractor(
        audio, sampling_rate=16000, padding=True, return_tensors="pt"
    )
    return {**audio_features, "labels": torch.tensor(labels, dtype=torch.long)}


def remove_empty_exp_folders(base_dir="results"):
    if not os.path.exists(base_dir):
        return

    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path) and not os.listdir(folder_path):
            print(f"Removing empty folder: {folder_path}")
            os.rmdir(folder_path)


def create_incremented_exp_folder(base_dir="results", experiment_name="exp"):
    remove_empty_exp_folders(base_dir)

    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    if os.path.exists(os.path.join(base_dir, experiment_name)):
        # Get all directories in the base folder
        existing_folders = [
            f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))
        ]

        # Find the highest experiment number
        exp_numbers = [
            int(f.replace(experiment_name, ""))
            for f in existing_folders
            if f.startswith(experiment_name) and f != experiment_name
        ]

        # Determine the next experiment number
        next_exp_num = max(exp_numbers) + 1 if exp_numbers else 1
        new_exp_dir = os.path.join(base_dir, f"{experiment_name}{next_exp_num}")
    else:
        new_exp_dir = os.path.join(base_dir, experiment_name)

    # Create the new experiment folder
    os.makedirs(new_exp_dir)
    return new_exp_dir
