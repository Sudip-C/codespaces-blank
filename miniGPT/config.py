from dataclasses import dataclass
import torch


@dataclass
class GPTConfig:

    vocab_size: int = 50257

    max_seq_len: int = 32

    d_model: int = 256

    n_heads: int = 8

    n_layers: int = 6

    d_ff: int = 1024

    dropout: float = 0.1

    batch_size: int = 16

    learning_rate: float = 3e-4

    epochs: int = 5

    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )