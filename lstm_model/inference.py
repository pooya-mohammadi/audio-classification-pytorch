from argparse import ArgumentParser
from pathlib import Path
from typing import Union
import numpy as np
import torch
import torchaudio
from model import LSTMModel
from dataset import get_featurizer


class Inference:
    def __init__(self, model_path, device='cpu'):
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']
        self.model = LSTMModel(classifier_output=config.classifier_output, feature_size=config.feature_size,
                               hidden_size=config.hidden_size, num_layers=config.num_layers,
                               dropout=config.dropout, bidirectional=config.bidirectional, device=device).to(device)
        self.model.eval()
        self.model.load_state_dict({k.replace("model.", ""): v for k, v in checkpoint['state_dict'].items()})
        self.device = device
        self.audio_transform = get_featurizer(config.sample_rate)
        self.sr = config.sample_rate
        self.label2name = config.label2name
        del checkpoint

    def recognize(self, audio: Union[str, Path, np.ndarray]):
        if isinstance(audio, str) or isinstance(audio, Path):
            waveform, sr = torchaudio.load(audio)
        else:
            raise ValueError()
        if sr > self.sr:
            waveform = torchaudio.transforms.Resample(sr, self.sr)(waveform)
        mfcc = self.audio_transform(waveform)
        mfcc = mfcc.squeeze(0).transpose(0, 1).unsqueeze(1).to(self.device)
        with torch.no_grad():
            preds = self.model(mfcc).squeeze(0)
            output_label = torch.argmax(preds).item()
        cls_name = self.label2name[output_label]
        return cls_name


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--model_path", default="output/exp_1/best.ckpt")
    parser.add_argument("--audio_path", default="audio_samples/man_01.mp3")
    args = parser.parse_args()
    model = Inference(args.model_path)
    output = model.recognize(args.audio_path)
    print(output)
