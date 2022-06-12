import torch.nn as nn
from copy import deepcopy
from model.train_utils import Embeddings, PositionalEncoding
from model.attention import MultiHeadedAttention
from model.encoder import EncoderLayer, Encoder
from model.feed_forward import PositionwiseFeedForward


class Transformer(nn.Module):
    def __init__(self, input_size, d_model=256, d_ff=512, h=8, dropout=0.1, n_encoders=1, n_classes=2):
        super(Transformer, self).__init__()

        attn = MultiHeadedAttention(h, d_model)
        fead_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        position = PositionalEncoding(d_model, dropout)
        encoder_layer = EncoderLayer(d_model, deepcopy(attn), deepcopy(fead_forward), dropout)
        self.encoder = Encoder(encoder_layer, n_encoders)

        self.input_embedding = nn.Sequential(nn.Linear(input_size, d_model),
                                             deepcopy(position)
                                             )  # Embeddings followed by PE

        # Fully-Connected Layer
        self.fc = nn.Linear(d_model, n_classes)

        # Softmax non-linearity
        # self.softmax = nn.Softmax()

    def forward(self, x):
        # x -> seq_len, batch_size, feature_size
        x = x.permute(1, 0, 2)  # shape = (batch_size, seq_len, feature_size)
        embedded_inputs = self.input_embedding(x)  # shape = (batch_size, seq_len, d_model)
        encoded_features = self.encoder(embedded_inputs)

        # Convert input to (batch_size, d_model) for linear layer
        # like hidden layers in lstm we take the last one as classifier
        final_feature_map = encoded_features[:, -1, :]
        final_out = self.fc(final_feature_map)
        return final_out