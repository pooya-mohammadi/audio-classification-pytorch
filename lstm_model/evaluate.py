import numpy as np
import torch
from deep_utils import PickleUtils, DirUtils, JsonUtils
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
from settings import Config
from pathlib import Path
import librosa
from tqdm import tqdm
from inference import Inference
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score

model_path = Path("./output/exp_8/best.ckpt")
sample_path = "../test_data"


if __name__ == '__main__':
    files, true_labels, mapping = DirUtils.crawl_directory_dataset(sample_path, ext_filter=".wav", map_labels=True)
    predictions = []
    with torch.no_grad():
        for sample_audio_path in tqdm(files):
            model = Inference(model_path)
            output = model.recognize(sample_audio_path)
            print(output)
            predictions.append(output)
            # print(f"class: {cls_index}, cls_name: {cls_name}")
    print("f1_score: ", f1_score(true_labels, predictions, labels=list(mapping.values()), average="macro"))
    print("precision_score: ", precision_score(true_labels, predictions, labels=list(mapping.values()), average="macro"))
    print("accuracy_score: ", accuracy_score(true_labels, predictions))
    print("recall_score: ", recall_score(true_labels, predictions, labels=list(mapping.values()), average="macro"))
