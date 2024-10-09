import os
import glob
import json
from datasets import Dataset, Audio
from sklearn.model_selection import train_test_split

label2id = {"male": 0, "female": 1, "unknown": 2}


def get_audio_filepath(json_filepath, audio_folder):
    return os.path.join(
        audio_folder, os.path.basename(json_filepath.replace(".json", ".wav"))
    )


def load_dataset(json_folder, audio_folder, label2id, target_variable, test_size=0.2):
    data = []
    json_filepaths = glob.glob(os.path.join(json_folder, "*"), recursive=True)
    for json_filepath in json_filepaths:
        with open(json_filepath, "r") as json_file:
            file = json.load(json_file)
            audio_filepath = get_audio_filepath(json_filepath, audio_folder)
            data.append(
                {"file_path": audio_filepath, "label": label2id[file[target_variable]]}
            )

    # Split the dataset into train and test sets
    train_data, test_data = train_test_split(data, test_size=test_size, shuffle=True)

    train_dataset = Dataset.from_list(train_data)
    train_dataset = train_dataset.cast_column("file_path", Audio(sampling_rate=16000))

    test_dataset = Dataset.from_list(test_data)
    test_dataset = test_dataset.cast_column("file_path", Audio(sampling_rate=16000))

    return Dataset.from_list(train_dataset), Dataset.from_list(test_dataset)
