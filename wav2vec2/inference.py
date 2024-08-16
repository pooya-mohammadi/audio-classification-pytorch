import numpy as np
import torch
from deep_utils import PickleUtils
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from settings import Config
from pathlib import Path
import librosa
from datasets import load_dataset, Audio

inference_dir = Path("./results/exp_12/best")
sample_path = "../sentiment_data/val/S/F02S05.wav"

config = Config()
label2id = PickleUtils.load_pickle(inference_dir / "label2id.pkl")
id2label = {int(v): k for k, v in label2id.items()}
feature_extractor = AutoFeatureExtractor.from_pretrained(config.feature_extractor)


def get_audio(path: str):
    audio, _ = librosa.load(path)
    inputs = feature_extractor(
        audio,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True
    )
    return inputs['input_values'][0]


# early_stopping = EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)

device = "cuda:1"
model = AutoModelForAudioClassification.from_pretrained(
    inference_dir,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# model.load_state_dict(torch.load(inference_dir / "model.safetensors"))
model = model.to(device)

if __name__ == '__main__':
    audio_array = get_audio(sample_path)
    with torch.no_grad():
        audio_array = torch.tensor(audio_array).to(device=device)
        audio_array = audio_array[None, ...]
        output = model(audio_array)
        logits = output["logits"][0]
        cls_index = torch.argmax(logits).item()
        print(f"class: {cls_index}, cls_name: {id2label[cls_index]}")
