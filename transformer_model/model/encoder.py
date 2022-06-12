# encoder.py

from torch import nn
from model.train_utils import clones
from model.sublayer import LayerNorm, ResidualAddNormDrop


class Encoder(nn.Module):
    '''
    Transformer Encoder
    
    It is a stack of N layers.
    '''

    def __init__(self, layer, n_encoders):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_encoders)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class EncoderLayer(nn.Module):
    '''
    An encoder layer
    
    Made up of self-attention and a feed forward layer.
    Each of these sublayers have residual and layer norm, implemented by SublayerOutput.
    '''

    def __init__(self, d_model, self_attn, feed_forward, dropout):
        """

        :param d_model:
        :param self_attn: multi-head attention
        :param feed_forward: feed forward final layer
        :param dropout: dropout
        """
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.multi_head_residual = ResidualAddNormDrop(d_model, dropout)
        self.feed_forward_residual = ResidualAddNormDrop(d_model, dropout)
        self.size = d_model

    def forward(self, x, mask=None):
        "Transformer Encoder"
        x = self.multi_head_residual(x, lambda x: self.self_attn(x, x, x, mask))  # Encoder self-attention
        return self.feed_forward_residual(x, self.feed_forward)
