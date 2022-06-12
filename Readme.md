# Audio Classification

In this project, several approaches for training/finetuning an audio gender recognition is provided. The code can simply
be used for any other classification by changing the number of classes and the input dataset.

# Models

1. LSTM_Model: uses mfccs to train a lstm model for audio classification. Trained using pytorchlightning.
    1. the idea of this structure is taken from [LearnedVector](https://github.com/LearnedVector) repository which
       contains a wakeup model.
2. transformer_scratch: Uses a transformer block for training an audio classification model with mfccs taken as inputs.
   Trained using pytorchlightning.
    1. main implementation is taken from [AnubhavGupta3377](https://github.com/AnubhavGupta3377)'s repo
       called [Text-Classification-Models-Pytorch](https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch)
    2. It's modified to train audio samples.
3. wav2vec2: Finetuning wav2vec2-base as an audio classification model using huggingface trainer.

# Result on Gender Recognition

Trained and evaluated on a custom dataset. You can simply download common-voice dataset and use the samples.

Model | Train ACC | Val Acc | Train F1-score | Val-F1-score |
--- |-----------|-------------|----------------|-------------|
LSTM | 89        | 90 | 90.83          | 91 |
Wav2vec2 | -         | 96.4| -              | 96.4
transfomer | 85.1 | 81.7| 87.1| 84.6

# references:

1. https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant
2. https://github.com/huggingface/transformers
3. https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
4. https://pytorch.org/tutorials/beginner/transformer_tutorial.html
5. https://github.com/pooya-mohammadi/deep_utils