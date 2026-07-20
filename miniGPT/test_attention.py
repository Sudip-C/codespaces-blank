import torch

from config import GPTConfig
from model.attention import CausalSelfAttention


def main():

    config = GPTConfig()

    model = CausalSelfAttention(config)

    x = torch.randn(
        2,
        10,
        config.d_model
    )

    output = model(x)

    print("Input shape:", x.shape)

    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()