import torch

from torch.utils.data import DataLoader

from config import GPTConfig

from model.gpt import GPTModel

from model.tokenizer import GPTTokenizer

from data.dataset import TextDataset


def main():

    config = GPTConfig()

    print(
        f"Using device: {config.device}"
    )

    # Load text

    with open(
        "data/train.txt",
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # Tokenizer

    tokenizer = GPTTokenizer()

    token_ids = tokenizer.encode(
        text
    )

    print(
        f"Number of tokens: {len(token_ids)}"
    )

    # Dataset

    dataset = TextDataset(

        token_ids,

        config.max_seq_len
    )

    dataloader = DataLoader(

        dataset,

        batch_size=config.batch_size,

        shuffle=True
    )

    # Model

    model = GPTModel(
        config
    )

    model = model.to(
        config.device
    )

    # Optimizer

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.learning_rate
    )

    # Loss function

    loss_function = torch.nn.CrossEntropyLoss()

    # Training

    model.train()

    for epoch in range(
        config.epochs
    ):

        total_loss = 0

        for batch_idx, (
            input_ids,
            target_ids
        ) in enumerate(dataloader):

            input_ids = input_ids.to(
                config.device
            )

            target_ids = target_ids.to(
                config.device
            )

            # Forward pass

            logits = model(
                input_ids
            )

            # Reshape logits

            logits = logits.view(

                -1,

                config.vocab_size
            )

            targets = target_ids.view(
                -1
            )

            # Calculate loss

            loss = loss_function(

                logits,

                targets
            )

            # Clear gradients

            optimizer.zero_grad()

            # Backpropagation

            loss.backward()

            # Update weights

            optimizer.step()

            total_loss += loss.item()

            if batch_idx % 10 == 0:

                print(

                    f"Epoch: {epoch + 1} | "

                    f"Batch: {batch_idx} | "

                    f"Loss: {loss.item():.4f}"

                )

        average_loss = (

            total_loss
            / len(dataloader)
        )

        print(

            f"\nEpoch {epoch + 1} "

            f"Average Loss: "
            f"{average_loss:.4f}\n"
        )

    # Save model

    torch.save(

        model.state_dict(),

        "model_checkpoint.pt"
    )

    print(
        "Model saved successfully."
    )


if __name__ == "__main__":

    main()