import torch
import torch.nn as nn

from model.embedding import GPTEmbedding
from model.transformer import TransformerBlock


class GPTModel(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.embedding = GPTEmbedding(
            config
        )

        self.transformer_blocks = nn.ModuleList(

            [
                TransformerBlock(config)

                for _ in range(
                    config.n_layers
                )
            ]
        )

        self.final_layer_norm = nn.LayerNorm(
            config.d_model
        )

        self.output_head = nn.Linear(

            config.d_model,

            config.vocab_size,

            bias=False
        )

    def forward(self, input_ids):

        x = self.embedding(
            input_ids
        )

        for block in self.transformer_blocks:

            x = block(x)

        x = self.final_layer_norm(
            x
        )

        logits = self.output_head(
            x
        )

        return logits