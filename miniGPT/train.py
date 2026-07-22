import os

import torch

from torch.utils.data import DataLoader

from config import GPTConfig

from model.gpt import GPTModel

from model.tokenizer import GPTTokenizer

from data.dataset import TextDataset


# ============================================
# Evaluation Function
# ============================================

def evaluate(
    model,
    dataloader,
    loss_function,
    device,
    vocab_size
):

    model.eval()

    total_loss = 0.0

    total_batches = 0

    with torch.no_grad():

        for input_ids, target_ids in dataloader:

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

            # Reshape logits

            logits = logits.view(

                -1,

                vocab_size
            )

            # Reshape targets

            targets = target_ids.view(
                -1
            )

            # Calculate loss

            loss = loss_function(

                logits,

                targets
            )

            total_loss += loss.item()

            total_batches += 1

    model.train()

    return total_loss / total_batches


# ============================================
# Main Function
# ============================================

def main():

    # ----------------------------------------
    # Configuration
    # ----------------------------------------

    config = GPTConfig()

    device = config.device

    print(
        f"Using device: {device}"
    )


    # ----------------------------------------
    # Load Training Text
    # ----------------------------------------

    with open(

        "data/train.txt",

        "r",

        encoding="utf-8"

    ) as file:

        train_text = file.read()


    # ----------------------------------------
    # Load Validation Text
    # ----------------------------------------

    with open(

        "data/validation.txt",

        "r",

        encoding="utf-8"

    ) as file:

        validation_text = file.read()


    # ----------------------------------------
    # Tokenizer
    # ----------------------------------------

    tokenizer = GPTTokenizer()


    train_tokens = tokenizer.encode(

        train_text
    )


    validation_tokens = tokenizer.encode(

        validation_text
    )


    print(

        f"Training tokens: "
        f"{len(train_tokens):,}"
    )


    print(

        f"Validation tokens: "
        f"{len(validation_tokens):,}"
    )


    print(

        f"Total tokens: "

        f"{len(train_tokens) + len(validation_tokens):,}"
    )


    # ----------------------------------------
    # Training Dataset
    # ----------------------------------------

    train_dataset = TextDataset(

        train_tokens,

        config.max_seq_len
    )


    # ----------------------------------------
    # Validation Dataset
    # ----------------------------------------

    validation_dataset = TextDataset(

        validation_tokens,

        config.max_seq_len
    )


    # ----------------------------------------
    # Training DataLoader
    # ----------------------------------------

    train_loader = DataLoader(

        train_dataset,

        batch_size=config.batch_size,

        shuffle=True
    )
    print(

        f"Total batches per epoch: "

        f"{len(train_loader):,}"
    )

    # ----------------------------------------
    # Validation DataLoader
    # ----------------------------------------

    validation_loader = DataLoader(

        validation_dataset,

        batch_size=config.batch_size,

        shuffle=False
    )


    # ----------------------------------------
    # Model
    # ----------------------------------------

    model = GPTModel(

        config
    )


    model = model.to(

        device
    )


    # ----------------------------------------
    # Optimizer
    # ----------------------------------------

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=config.learning_rate,

        weight_decay=0.01
    )


    # ----------------------------------------
    # Learning Rate Scheduler
    # ----------------------------------------

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(

        optimizer,

        T_max=config.epochs
    )


    # ----------------------------------------
    # Loss Function
    # ----------------------------------------

    loss_function = torch.nn.CrossEntropyLoss()


    # ----------------------------------------
    # Checkpoint Variables
    # ----------------------------------------

    checkpoint_path = "checkpoint.pt"

    start_epoch = 0

    resume_batch = -1

    best_validation_loss = float(
        "inf"
    )


    # ----------------------------------------
    # Load Checkpoint
    # ----------------------------------------

    if os.path.exists(

        checkpoint_path
    ):

        print()

        print(
            "Loading checkpoint..."
        )


        checkpoint = torch.load(

            checkpoint_path,

            map_location=device
        )


        # Restore model

        model.load_state_dict(

            checkpoint[
                "model_state_dict"
            ]
        )


        # Restore optimizer

        optimizer.load_state_dict(

            checkpoint[
                "optimizer_state_dict"
            ]
        )


        # Restore scheduler

        scheduler.load_state_dict(

            checkpoint[
                "scheduler_state_dict"
            ]
        )


        # Restore epoch

        start_epoch = checkpoint[

            "epoch"
        ]


        # Restore batch

        resume_batch = checkpoint[

            "batch"
        ]


        # Restore best validation loss

        best_validation_loss = checkpoint[

            "best_validation_loss"
        ]


        print(

            f"Resuming from "

            f"Epoch {start_epoch + 1}, "

            f"Batch {resume_batch + 1}"
        )


    # ----------------------------------------
    # Training
    # ----------------------------------------

    for epoch in range(

        start_epoch,

        config.epochs
    ):


        model.train()


        total_train_loss = 0.0


        print()

        print(

            f"Starting Epoch {epoch + 1}"
        )


        # ------------------------------------
        # Batch Training
        # ------------------------------------

        for batch_idx, (

            input_ids,

            target_ids

        ) in enumerate(train_loader):


            # --------------------------------
            # Resume Logic
            # --------------------------------

            if (

                epoch == start_epoch

                and batch_idx <= resume_batch
            ):

                continue


            # Move data to device

            input_ids = input_ids.to(

                device
            )


            target_ids = target_ids.to(

                device
            )


            # --------------------------------
            # Forward Pass
            # --------------------------------

            logits = model(

                input_ids
            )


            # --------------------------------
            # Reshape Logits
            # --------------------------------

            logits = logits.view(

                -1,

                config.vocab_size
            )


            # --------------------------------
            # Reshape Targets
            # --------------------------------

            targets = target_ids.view(

                -1
            )


            # --------------------------------
            # Calculate Loss
            # --------------------------------

            loss = loss_function(

                logits,

                targets
            )


            # --------------------------------
            # Clear Gradients
            # --------------------------------

            optimizer.zero_grad()


            # --------------------------------
            # Backpropagation
            # --------------------------------

            loss.backward()


            # --------------------------------
            # Gradient Clipping
            # --------------------------------

            torch.nn.utils.clip_grad_norm_(

                model.parameters(),

                max_norm=1.0
            )


            # --------------------------------
            # Update Weights
            # --------------------------------

            optimizer.step()


            # --------------------------------
            # Add Loss
            # --------------------------------

            total_train_loss += loss.item()


            # --------------------------------
            # Print Progress
            # --------------------------------

            if batch_idx % 10 == 0:

                print(

                    f"Epoch {epoch + 1} | "

                    f"Batch {batch_idx} | "

                    f"Loss {loss.item():.4f}"
                )


            # --------------------------------
            # Save Checkpoint
            # --------------------------------

            if (

                batch_idx > 0

                and batch_idx % 200 == 0
            ):


                torch.save(

                    {

                        "model_state_dict":

                            model.state_dict(),


                        "optimizer_state_dict":

                            optimizer.state_dict(),


                        "scheduler_state_dict":

                            scheduler.state_dict(),


                        "epoch":

                            epoch,


                        "batch":

                            batch_idx,


                        "best_validation_loss":

                            best_validation_loss

                    },

                    checkpoint_path
                )


                print()


                print(

                    f"Checkpoint saved at "

                    f"Epoch {epoch + 1}, "

                    f"Batch {batch_idx}"
                )


        # ------------------------------------
        # Average Training Loss
        # ------------------------------------

        average_train_loss = (

            total_train_loss

            / len(train_loader)
        )


        # ------------------------------------
        # Validation
        # ------------------------------------

        validation_loss = evaluate(

            model,

            validation_loader,

            loss_function,

            device,

            config.vocab_size
        )


        # ------------------------------------
        # Update Learning Rate
        # ------------------------------------

        scheduler.step()


        # ------------------------------------
        # Print Epoch Results
        # ------------------------------------

        print()


        print(

            f"Epoch {epoch + 1}"
        )


        print(

            f"Train Loss: "

            f"{average_train_loss:.4f}"
        )


        print(

            f"Validation Loss: "

            f"{validation_loss:.4f}"
        )


        # ------------------------------------
        # Save Best Model
        # ------------------------------------

        if (

            validation_loss

            < best_validation_loss
        ):


            best_validation_loss = validation_loss


            torch.save(

                model.state_dict(),

                "best_model.pt"
            )


            print(

                "Best model saved."
            )


        # ------------------------------------
        # Save End-of-Epoch Checkpoint
        # ------------------------------------

        torch.save(

            {

                "model_state_dict":

                    model.state_dict(),


                "optimizer_state_dict":

                    optimizer.state_dict(),


                "scheduler_state_dict":

                    scheduler.state_dict(),


                # Next epoch

                "epoch":

                    epoch + 1,


                # -1 means no batch to skip

                "batch":

                    -1,


                "best_validation_loss":

                    best_validation_loss

            },

            checkpoint_path
        )


        print(

            "End-of-epoch checkpoint saved."
        )


        print(

            "-" * 50
        )


# ============================================
# Program Entry Point
# ============================================

if __name__ == "__main__":

    main()