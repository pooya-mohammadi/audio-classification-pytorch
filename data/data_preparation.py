import os
import shutil

import pandas as pd
from deep_utils import DirUtils
from os.path import split, join
from collections import defaultdict

from sklearn.model_selection import train_test_split

data_folders = ["female", "male"]
output_dir = "../sentiment_data/"
val_dir = "../sentiment_data/val"
test_dir = "../sentiment_data/test"
train_dir = "../sentiment_data/train"

DirUtils.remove_create(output_dir)
test_size = 0.25

data_sample_id = dict()


def move_me(sample_ids, out_dir):
    data = []
    for lst in sample_ids:
        data.extend(lst)

    for audio_cls, audio_name, sample_path in data:
        os.makedirs(join(out_dir, audio_cls), exist_ok=True)
        shutil.copy(sample_path, join(out_dir, audio_cls, audio_name))


for folder in data_folders:
    for sample in DirUtils.list_dir_full_path(folder, interest_extensions=".wav"):
        name = split(sample)[-1]
        gender = name[0]
        sample_id = name[1:3]
        audio_class = name[3]
        audio_id = name[4:]
        if sample_id and sample_id not in data_sample_id:
            data_sample_id[sample_id] = [(audio_class, name, sample)]
        elif sample_id and sample_id in data_sample_id:
            data_sample_id[sample_id].append((audio_class, name, sample))
        else:
            raise ValueError(sample_id)
train_sample_id, val_sample_id = train_test_split(list(data_sample_id.keys()), test_size=test_size)
val_sample_id, test_sample_id = train_test_split(val_sample_id, test_size=0.5)

move_me([data_sample_id[key] for key in train_sample_id if data_sample_id[key]], train_dir)
move_me([data_sample_id[key] for key in val_sample_id if data_sample_id[key]], val_dir)
move_me([data_sample_id[key] for key in test_sample_id if data_sample_id[key]], test_dir)

    # os.makedirs(join(output_dir, audio_class), exist_ok=True)
    # shutil.copy(sample, join(output_dir, audio_class, name))


def save_csv(data_path: str, csv_path: str):
    file_paths, labels, name2label = DirUtils.crawl_directory_dataset(data_path, map_labels=True)
    label2name = {v: k for k, v in name2label.items()}
    columns = ["audio_path", "label"]
    csv_df = [[path, label2name[lbl]] for path, lbl in zip(file_paths, labels)]
    pd.DataFrame(csv_df, columns=columns).to_csv(csv_path, index_label=False)


if __name__ == '__main__':
    # DirUtils.split_dir_of_dir(output_dir, train_dir=train_dir, val_dir=val_dir + "_val", test_size=test_size)
    # DirUtils.split_dir_of_dir(val_dir + "_val", train_dir=val_dir, val_dir=test_dir, mode="mv", test_size=0.5)
    save_csv(train_dir, train_dir + ".csv")
    save_csv(val_dir, val_dir + ".csv")
    save_csv(test_dir, test_dir + ".csv")
