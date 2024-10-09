from utils import load_config, load_model
from prepare_dataset import load_dataset
import soundfile as sf
import argparse
from sklearn.metrics import classification_report
from tqdm import tqdm
import torch


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


def evaluate(model, feature_extractor, test_dataset, label2id, id2label):
    """ Evaluate the model on a test dataset. """
    y_true = []
    y_pred = []

    for audio_file, label in tqdm(test_dataset):
        # Predict class for each audio file
        predicted_class_id = predict(audio_file, model, feature_extractor)

        # Append the true label and predicted label
        y_true.append(label)
        y_pred.append(predicted_class_id)

    # Generate classification report
    print(classification_report(y_true, y_pred, target_names=id2label.values()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_type", type=str, choices=["small", "medium"], default="medium"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/results/gender_classification2/checkpoint-636/",
    )
    # parser.add_argument("--test_data_path", type=str, required=True, help="Path to the folder containing test audio files and labels.")
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

    # Load the dataset
    dataset = {}
    dataset["train"], dataset["test"] = load_dataset(
        json_folder=config["folders"]["json_folder"],
        audio_folder=config["folders"]["audio_folder"],
        target_variable=target_variable,
        label2id=config["target_variables"][target_variable]["label2id"],
    )
    test_dataset = []
    for data in dataset["test"]:
        label = data["label"]
        test_dataset.append((data["file_path"]["path"], label))

    # Perform evaluation
    evaluate(model, feature_extractor, test_dataset, label2id, id2label)
