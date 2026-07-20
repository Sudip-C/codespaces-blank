import torch

from config import GPTConfig
from model.embedding import GPTEmbedding


def main():

    config = GPTConfig()

    model = GPTEmbedding(config)

    x = torch.randint(
        low=0,
        high=config.vocab_size,
        size=(2, 10)
    )

    output = model(x)

    print("Input shape:", x.shape)

    print("Output shape:", output.shape)


if __name__ == "__main__":
    main()