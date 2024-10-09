from transformers import TrainingArguments, Trainer
from prepare_dataset import load_dataset
from utils import (
    load_config,
    create_incremented_exp_folder,
    load_model,
    preprocess_function,
)
from functools import partial

if __name__ == "__main__":
    # Load configuration files
    config = load_config("config.yaml", "target_variable.yaml")
    training_args_config = config["training_args"]
    target_variable = config["target_variable"]

    # Create a new auto-incremented experiment folder
    new_exp_folder = create_incremented_exp_folder(
        config["folders"]["output_base"], config["folders"]["experiment_name"]
    )

    # Update the output_dir in the training arguments with the new experiment folder
    training_args_config["output_dir"] = new_exp_folder

    # Load the model and feature extractor
    model, feature_extractor = load_model(config["model_args"])

    # Load the dataset
    dataset = {}
    dataset["train"], dataset["test"] = load_dataset(
        json_folder=config["folders"]["json_folder"],
        audio_folder=config["folders"]["audio_folder"],
        target_variable=target_variable,
        label2id=config["target_variables"][target_variable]["label2id"],
    )

    # Use partial to bind feature_extractor to preprocess_function
    preprocess_fn = partial(preprocess_function, feature_extractor=feature_extractor)
    dataset["train"] = dataset["train"].map(preprocess_fn, batched=True)
    dataset["test"] = dataset["test"].map(preprocess_fn, batched=True)

    # Define training arguments with updated output directory
    training_args = TrainingArguments(**training_args_config)

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=None,
    )

    # Fine-tune the model
    trainer.train()

    # Save the fine-tuned model in .bin format
    trainer.save_model(
        new_exp_folder
    )  # Saves model as 'pytorch_model.bin' inside the new_exp_folder
