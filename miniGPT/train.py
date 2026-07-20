import torch

from torch.utils.data import DataLoader

from config import GPTConfig

from model.gpt import GPTModel

from model.tokenizer import GPTTokenizer

from data.dataset import TextDataset


def evaluate(
    model,
    dataloader,
    loss_function,
    device,
    vocab_size
):

    model.eval()

    total_loss = 0

    total_batches = 0

    with torch.no_grad():

        for input_ids, target_ids in dataloader:

            input_ids = input_ids.to(
                device
            )

            target_ids = target_ids.to(
                device
            )

            logits = model(
                input_ids
            )

            logits = logits.view(
                -1,
                vocab_size
            )

            targets = target_ids.view(
                -1
            )

            loss = loss_function(
                logits,
                targets
            )

            total_loss += loss.item()

            total_batches += 1

    model.train()

    return total_loss / total_batches


def main():

    config = GPTConfig()

    device = config.device

    print(
        f"Using device: {device}"
    )

    # --------------------------------
    # Load text
    # --------------------------------

    with open(
        "data/train.txt",
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    # --------------------------------
    # Tokenizer
    # --------------------------------

    tokenizer = GPTTokenizer()

    token_ids = tokenizer.encode(
        text
    )

    print(
        f"Total tokens: {len(token_ids)}"
    )

    # --------------------------------
    # Train / validation split
    # --------------------------------

    split_index = int(
        len(token_ids) * 0.9
    )

    train_tokens = token_ids[
        :split_index
    ]

    validation_tokens = token_ids[
        split_index:
    ]

    # --------------------------------
    # Datasets
    # --------------------------------

    train_dataset = TextDataset(

        train_tokens,

        config.max_seq_len
    )

    validation_dataset = TextDataset(

        validation_tokens,

        config.max_seq_len
    )

    # --------------------------------
    # DataLoaders
    # --------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.batch_size,

        shuffle=True
    )

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=config.batch_size,

        shuffle=False
    )

    # --------------------------------
    # Model
    # --------------------------------

    model = GPTModel(
        config
    )

    model = model.to(
        device
    )

    # --------------------------------
    # Optimizer
    # --------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.learning_rate,

        weight_decay=0.01
    )

    # --------------------------------
    # Scheduler
    # --------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=config.epochs
    )

    # --------------------------------
    # Loss
    # --------------------------------

    loss_function = torch.nn.CrossEntropyLoss()

    best_validation_loss = float(
        "inf"
    )

    # --------------------------------
    # Training
    # --------------------------------

    for epoch in range(
        config.epochs
    ):

        model.train()

        total_train_loss = 0

        for batch_idx, (

            input_ids,

            target_ids

        ) in enumerate(train_loader):

            input_ids = input_ids.to(
                device
            )

            target_ids = target_ids.to(
                device
            )

            # Forward pass

            logits = model(
                input_ids
            )

            logits = logits.view(

                -1,

                config.vocab_size
            )

            targets = target_ids.view(
                -1
            )

            # Loss

            loss = loss_function(

                logits,

                targets
            )

            # Clear gradients

            optimizer.zero_grad()

            # Backpropagation

            loss.backward()

            # Gradient clipping

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0
            )

            # Update weights

            optimizer.step()

            total_train_loss += loss.item()

            if batch_idx % 10 == 0:

                print(

                    f"Epoch {epoch + 1} | "

                    f"Batch {batch_idx} | "

                    f"Loss {loss.item():.4f}"
                )

        # Average training loss

        average_train_loss = (

            total_train_loss
            / len(train_loader)
        )

        # Validation

        validation_loss = evaluate(

            model,

            validation_loader,

            loss_function,

            device,

            config.vocab_size
        )

        # Update learning rate

        scheduler.step()

        print(

            f"\nEpoch {epoch + 1}"

        )

        print(

            f"Train Loss: "
            f"{average_train_loss:.4f}"
        )

        print(

            f"Validation Loss: "
            f"{validation_loss:.4f}"
        )

        # Save best model

        if validation_loss < best_validation_loss:

            best_validation_loss = validation_loss

            torch.save(

                model.state_dict(),

                "best_model.pt"
            )

            print(
                "Best model saved."
            )

        print(
            "-" * 50
        )


if __name__ == "__main__":

    main()