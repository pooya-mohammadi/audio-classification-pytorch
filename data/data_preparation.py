import os
import shutil

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

DirUtils.split_dir_of_dir(output_dir, train_dir=train_dir, val_dir=val_dir, test_size=test_size)
