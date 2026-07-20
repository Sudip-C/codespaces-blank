import torch
import torch.nn as nn


class FeedForward(nn.Module):

    def __init__(self, config):

        super().__init__()

        self.network = nn.Sequential(

            nn.Linear(
                config.d_model,
                config.d_ff
            ),

            nn.GELU(),

            nn.Linear(
                config.d_ff,
                config.d_model
            ),

            nn.Dropout(
                config.dropout
            )
        )

    def forward(self, x):

        return self.network(x)