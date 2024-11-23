import numpy as np
import torch
from deep_utils import PickleUtils, DirUtils, JsonUtils
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from settings import Config
from pathlib import Path
import librosa
from tqdm import tqdm

sample_path = "/home/ai/projects/audio-data-movies"

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
    audio_files = DirUtils.list_dir_full_path(sample_path, interest_extensions=".wav")
    with torch.no_grad():
        for sample_audio_path in tqdm(audio_files):
            audio_array = get_audio(sample_audio_path)
            audio_array = torch.tensor(audio_array).to(device=device)
            audio_array = audio_array[None, ...]
            output = model(audio_array)
            logits = output["logits"][0]
            cls_index = torch.argmax(logits).item()
            cls_name = id2label[cls_index]
            json_path = DirUtils.split_extension(sample_audio_path, extension=".json")
            data = JsonUtils.load(json_path)
            data['label_name'] = cls_name
            data['label_value'] = cls_index
            JsonUtils.dump(json_path, data)
            # print(f"class: {cls_index}, cls_name: {cls_name}")
