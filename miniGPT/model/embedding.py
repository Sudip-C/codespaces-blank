import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class PositionalEmbedding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=max_seq_len,
            embedding_dim=d_model
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        )

        positions = positions.unsqueeze(0)

        positions = positions.expand(
            batch_size,
            seq_len
        )

        return self.embedding(positions)


class GPTEmbedding(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.token_embedding = TokenEmbedding(
            config.vocab_size,
            config.d_model
        )

        self.position_embedding = PositionalEmbedding(
            config.max_seq_len,
            config.d_model
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        token_embeddings = self.token_embedding(x)

        position_embeddings = self.position_embedding(x)

        embeddings = (
            token_embeddings
            + position_embeddings
        )

        return self.dropout(embeddings)