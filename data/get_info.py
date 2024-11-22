import os
import shutil
from collections import defaultdict

import pandas as pd
from deep_utils import DirUtils
from os.path import split, join

data_folders = ["female", "male"]
output_dir = "../sentiment_data/data"
val_dir = "../sentiment_data/val"
train_dir = "../sentiment_data/train"

DirUtils.remove_create(output_dir)
test_size = 0.1
data = defaultdict(int)
for folder in data_folders:
    for sample in DirUtils.list_dir_full_path(folder, interest_extensions=".wav"):
        name = split(sample)[-1]
        gender = name[0]
        sample_id = name[1:3]
        audio_class = name[3]
        audio_id = name[4:]
        data[audio_class] += 1
print(data)
