import os
import shutil

import pandas as pd
from deep_utils import DirUtils
from os.path import split, join

data_folders = ["female", "male"]
output_dir = "../sentiment_data/data"
val_dir = "../sentiment_data/val"
train_dir = "../sentiment_data/train"

DirUtils.remove_create(output_dir)
test_size = 0.1

for folder in data_folders:
    for sample in DirUtils.list_dir_full_path(folder, interest_extensions=".wav"):
        name = split(sample)[-1]
        gender = name[0]
        sample_id = name[1:3]
        audio_class = name[3]
        audio_id = name[4:]

        os.makedirs(join(output_dir, audio_class), exist_ok=True)
        shutil.copy(sample, join(output_dir, audio_class, name))


def save_csv(data_path: str, csv_path: str):
    file_paths, labels, name2label = DirUtils.crawl_directory_dataset(data_path, map_labels=True)
    label2name = {v: k for k, v in name2label.items()}
    columns = ["audio_path", "label"]
    csv_df = [[path, label2name[lbl]] for path, lbl in zip(file_paths, labels)]
    pd.DataFrame(csv_df, columns=columns).to_csv(csv_path, index_label=False)


if __name__ == '__main__':
    DirUtils.split_dir_of_dir(output_dir, train_dir=train_dir, val_dir=val_dir, test_size=test_size)
    save_csv(train_dir, train_dir + ".csv")
    save_csv(val_dir, val_dir + ".csv")
