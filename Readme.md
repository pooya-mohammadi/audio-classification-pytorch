# Audio Classification

In this project, several approaches for training/finetuning an audio gender recognition is provided. The code can simply
be used for any other classification by changing the number of classes and the input dataset. 

# Models
1. LSTM_Model: uses mfccs to train a lstm model for audio classification. 
   1. the idea of this structure is taken from [LearnedVector](https://github.com/LearnedVector) repository which contains a wakeup model.
2. transformer_scratch: Uses a transformer block for training an audio classification model with mfccs taken as inputs
3. wav2vec2: Finetuning wav2vec2-base as an audio classification model



# references:
1. https://github.com/LearnedVector/A-Hackers-AI-Voice-Assistant
2. https://github.com/huggingface/transformers
3. https://github.com/AnubhavGupta3377/Text-Classification-Models-Pytorch
4. https://pytorch.org/tutorials/beginner/transformer_tutorial.html
5. https://github.com/pooya-mohammadi/deep_utils