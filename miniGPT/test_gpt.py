import torch

from config import GPTConfig
from model.gpt import GPTModel


def main():

    config = GPTConfig()

    model = GPTModel(
        config
    )

    input_ids = torch.randint(

        low=0,

        high=config.vocab_size,

        size=(2, 10)
    )

    logits = model(
        input_ids
    )

    print(
        "Input shape:",
        input_ids.shape
    )

    print(
        "Output shape:",
        logits.shape
    )


if __name__ == "__main__":

    main()