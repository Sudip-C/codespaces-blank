import math

import torch
import torch.nn as nn


class CausalSelfAttention(nn.Module):

    def __init__(self, config):

        super().__init__()

        assert config.d_model % config.n_heads == 0

        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        self.qkv_projection = nn.Linear(
            config.d_model,
            3 * config.d_model
        )

        self.output_projection = nn.Linear(
            config.d_model,
            config.d_model
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        mask = torch.tril(
            torch.ones(
                config.max_seq_len,
                config.max_seq_len
            )
        )

        self.register_buffer(
            "mask",
            mask.view(
                1,
                1,
                config.max_seq_len,
                config.max_seq_len
            )
        )

    def forward(self, x):

        batch_size, seq_len, d_model = x.shape

        # Create Query, Key, Value
        qkv = self.qkv_projection(x)

        # Shape:
        # (batch, sequence, 3 * d_model)

        q, k, v = qkv.chunk(
            3,
            dim=-1
        )

        # Reshape into multiple heads
        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # Attention scores
        attention_scores = (
            q @ k.transpose(-2, -1)
        ) / math.sqrt(self.head_dim)

        # Causal mask
        attention_scores = attention_scores.masked_fill(
            self.mask[:, :, :seq_len, :seq_len] == 0,
            float("-inf")
        )

        # Convert scores to probabilities
        attention_weights = torch.softmax(
            attention_scores,
            dim=-1
        )

        attention_weights = self.dropout(
            attention_weights
        )

        # Weighted values
        output = attention_weights @ v

        # Combine heads
        output = output.transpose(
            1,
            2
        ).contiguous()

        output = output.view(
            batch_size,
            seq_len,
            d_model
        )

        # Final projection
        output = self.output_projection(
            output
        )

        return output

    def split_heads(self, x):

        batch_size, seq_len, _ = x.shape

        x = x.view(
            batch_size,
            seq_len,
            self.n_heads,
            self.head_dim
        )

        return x.transpose(
            1,
            2
        )