""""
This is not working!
"""

import time

from peft import LoraConfig, get_peft_model, LoftQConfig, prepare_model_for_kbit_training
import warnings

warnings.filterwarnings("ignore")

from os.path import join
from deep_utils import warmup_cosine, PickleUtils
from datasets import load_dataset, Audio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification, TrainingArguments, Trainer, \
    BitsAndBytesConfig
from settings import Config
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score
import numpy as np
import torch
from transformers import EarlyStoppingCallback
import os


# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_and_save_label2id(label2id_path, labels):
    label2id, id2label = dict(), dict()
    for i, label in enumerate(labels):
        label2id[label] = str(i)
        id2label[str(i)] = label
    PickleUtils.dump_pickle(label2id_path, label2id)
    print(f"Successfully saved label2id to {label2id_path}")
    return label2id, id2label


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    recall = recall_score(labels, predictions, average="weighted")
    precision = precision_score(labels, predictions, average="weighted")

    return {"accuracy": acc, "f1-score": f1, "recall-score": recall, "precision-score": precision}


def preprocess_function(examples):
    audio_arrays = [x["array"] for x in examples["audio_path"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        max_length=16000,
        truncation=True
    )
    label = [int(label2id[x]) for x in examples["label"]]
    inputs["label"] = label
    return inputs


if __name__ == '__main__':
    config = Config()
    feature_extractor = AutoFeatureExtractor.from_pretrained(config.feature_extractor)
    dataset = load_dataset('csv', data_files={'train': config.train_path, 'test': config.test_path})
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=config.target_sampling_rate))
    labels = set(dataset["train"]['label'])
    label2id, id2label = get_and_save_label2id(config.label2id_path, labels)
    encoded_dataset = dataset.map(preprocess_function, remove_columns="audio_path", batched=True)

    # early_stopping = EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)

    total_steps = int(
        (np.ceil(encoded_dataset["train"].num_rows / config.per_device_train_batch_size) * config.num_train_epochs))

    num_labels = len(id2label)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        load_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForAudioClassification.from_pretrained(
        "facebook/wav2vec2-base", num_labels=num_labels, label2id=label2id, id2label=id2label,
        # quantization_config=bnb_config
    )
    # model.is_gradient_checkpointing = False
    # model.frease_feature_extractor()
    # model.gradient_checkpointing_disable()

    model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        # task_type="SEQ_CLS",
        # inference_mode=False,
        r=64,
        lora_alpha=64,
        # lora_dropout=0.1,
        target_modules=["projector", "k_proj", "v_proj", "q_proj", "output_dense", "intermediate_dense", "out_proj",
                        # "conv",
                        # *[f"wav2vec2.encoder.layers.{i}.feed_forward.intermediate_dense" for i in range(12)],

                        ],
        modules_to_save=['classifier']
    )
    # config = peft.LoraConfig(
    #     r=8,
    #     target_modules=["seq.0", "seq.2"],  # Trained with LoRA. Only linear, Conv1D, and Conv2D are supported!
    #     modules_to_save=["seq.4"],  # trained but not with LoRA, for it's the last output
    # )

    model = get_peft_model(model, peft_config)

    model.to("cuda:0")
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=config.model_path,
        evaluation_strategy=config.evaluation_strategy,
        save_strategy=config.evaluation_strategy,
        num_train_epochs=config.num_train_epochs,
        report_to=config.report_to,
        load_best_model_at_end=config.load_best_model_at_end,
        save_total_limit=config.save_total_limit,
        metric_for_best_model=config.metric_for_best_model,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        logging_steps=config.logging_steps,
        label_names=['labels']
        # gradient_checkpointing=False
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_cosine(config.warmup_steps,
                                                                           max_lr=config.learning_rate,
                                                                           total_steps=total_steps,
                                                                           optimizer_lr=config.learning_rate,
                                                                           min_lr=config.min_learning_rate))
    # reduce lr with a cosine annealing if total_steps is set to total_steps
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=feature_extractor,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, scheduler)
    )
    tic = time.time()
    trainer.train()
    print(f"[INFO] Time: {time.time() - tic}")
    trainer.save_model(join(config.model_path, config.file_name))
    model.print_trainable_parameters()
