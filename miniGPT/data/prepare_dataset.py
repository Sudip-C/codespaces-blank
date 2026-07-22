from pathlib import Path

import tiktoken


RAW_PATH = Path(
    "data/raw/dataset.txt"
)

TRAIN_PATH = Path(
    "data/train.txt"
)

VALIDATION_PATH = Path(
    "data/validation.txt"
)


def clean_text(text):

    start_marker = "*** START OF THE PROJECT GUTENBERG EBOOK"

    end_marker = "*** END OF THE PROJECT GUTENBERG EBOOK"

    if start_marker in text:

        text = text.split(
            start_marker,
            1
        )[1]

    if end_marker in text:

        text = text.split(
            end_marker,
            1
        )[0]

    return text.strip()


def main():

    print("Reading raw dataset...")

    text = RAW_PATH.read_text(
        encoding="utf-8"
    )

    text = clean_text(
        text
    )

    print(
        f"Characters: {len(text):,}"
    )

    tokenizer = tiktoken.get_encoding(
        "gpt2"
    )

    token_ids = tokenizer.encode(
        text
    )

    print(
        f"Total tokens: {len(token_ids):,}"
    )

    split_index = int(
        len(token_ids) * 0.9
    )

    train_tokens = token_ids[
        :split_index
    ]

    validation_tokens = token_ids[
        split_index:
    ]

    train_text = tokenizer.decode(
        train_tokens
    )

    validation_text = tokenizer.decode(
        validation_tokens
    )

    TRAIN_PATH.write_text(
        train_text,
        encoding="utf-8"
    )

    VALIDATION_PATH.write_text(
        validation_text,
        encoding="utf-8"
    )

    print(
        f"Training tokens: {len(train_tokens):,}"
    )

    print(
        f"Validation tokens: {len(validation_tokens):,}"
    )


if __name__ == "__main__":

    main()