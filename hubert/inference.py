import torch
from utils import load_config, load_model
import soundfile as sf
import argparse


def load_audio(audio_file):
    """ Load audio file for inference. """
    audio, sample_rate = sf.read(audio_file)
    return audio, sample_rate


def predict(audio_file, model, feature_extractor):
    """ Run inference on a single audio file. """
    # Load and preprocess the audio file
    audio, _ = load_audio(audio_file)

    # Preprocess the audio file using the feature extractor
    inputs = feature_extractor(
        audio, sampling_rate=16000, return_tensors="pt", padding=True
    )

    # Ensure the model is in evaluation mode
    model.eval()

    # Run the model to get predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the predicted class (logits are unnormalized scores, argmax gives the highest score)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=-1).item()

    return predicted_class_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, choices=["small", "medium"], default="medium"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/home/ariyadis-pc03/github/brariyadis/asr/hubert/results/gender_classification2/checkpoint-636/",
    )
    parser.add_argument(
        "--audio_filepath",
        type=str,
        default="/home/ariyadis-pc03/github/brariyadis/asr/dataset_whole/selected/10106030201-0000.wav",
    )
    args = parser.parse_args()

    # Load the config
    config = load_config("config.yaml", "target_variable.yaml")
    target_variable = config["target_variable"]

    # Load the model and feature extractor
    model, feature_extractor = load_model(config["model_args"])

    # Reverse the id2label dictionary to create label2id
    label2id = config["target_variables"][target_variable]["label2id"]
    id2label = {v: k for k, v in label2id.items()}

    # Ensure the model is in evaluation mode
    model.eval()

    # Audio file for prediction
    audio_file = args.audio_filepath

    # Perform prediction
    predicted_class_id = predict(audio_file, model, feature_extractor)

    # Convert predicted class id to label
    predicted_label = id2label[predicted_class_id]

    print(f"Predicted label: {predicted_label}")
