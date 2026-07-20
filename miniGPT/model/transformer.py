import torch
import torch.nn as nn

from model.attention import CausalSelfAttention
from model.feedforward import FeedForward


class TransformerBlock(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.layer_norm_1 = nn.LayerNorm(
            config.d_model
        )

        self.attention = CausalSelfAttention(
            config
        )

        self.layer_norm_2 = nn.LayerNorm(
            config.d_model
        )

        self.feed_forward = FeedForward(
            config
        )

    def forward(self, x):

        # Attention + Residual Connection
        x = x + self.attention(
            self.layer_norm_1(x)
        )

        # Feed Forward + Residual Connection
        x = x + self.feed_forward(
            self.layer_norm_2(x)
        )

        return x