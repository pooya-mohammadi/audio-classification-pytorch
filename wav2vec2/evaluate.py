import numpy as np
import torch
from deep_utils import PickleUtils, DirUtils, JsonUtils
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from settings import Config
from pathlib import Path
import librosa
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

sample_path = "../test_data"

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

device = "cuda"
model = AutoModelForAudioClassification.from_pretrained(
    inference_dir,
    num_labels=len(label2id),
    label2id=label2id,
    id2label=id2label
)

# model.load_state_dict(torch.load(inference_dir / "model.safetensors"))
model = model.to(device)

if __name__ == '__main__':
    files, true_labels, mapping = DirUtils.crawl_directory_dataset(sample_path, ext_filter=".wav", map_labels=True)
    predictions = []
    with torch.no_grad():
        for sample_audio_path in tqdm(files):
            audio_array = get_audio(sample_audio_path)
            audio_array = torch.tensor(audio_array).to(device=device)
            audio_array = audio_array[None, ...]
            output = model(audio_array)
            logits = output["logits"][0]
            cls_index = torch.argmax(logits).item()
            predictions.append(cls_index)
            # print(f"class: {cls_index}, cls_name: {cls_name}")
    print("f1_score: ", f1_score(true_labels, predictions, labels=list(mapping.values()), average="macro"))
    print("precision_score: ", precision_score(true_labels, predictions, labels=list(mapping.values()), average="macro"))
    print("accuracy_score: ", accuracy_score(true_labels, predictions))
    print("recall_score: ", recall_score(true_labels, predictions, labels=list(mapping.values()), average="macro"))
