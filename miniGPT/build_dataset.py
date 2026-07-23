from pathlib import Path
import random


# --------------------------------
# Configuration
# --------------------------------

RAW_DATA_DIR = Path("data/raw")

TRAIN_FILE = Path("data/train.txt")

VALIDATION_FILE = Path(
    "data/validation.txt"
)

VALIDATION_SPLIT = 0.10

RANDOM_SEED = 42


# --------------------------------
# Find text files
# --------------------------------

text_files = list(

    RAW_DATA_DIR.rglob("*.txt")

)


if not text_files:

    raise ValueError(

        "No .txt files found inside "
        "data/raw/"

    )


print(

    f"Found {len(text_files)} text files."

)


# --------------------------------
# Read documents
# --------------------------------

documents = []


for file_path in text_files:

    print(

        f"Reading: {file_path}"

    )


    with open(

        file_path,

        "r",

        encoding="utf-8"

    ) as file:

        text = file.read()


    text = text.strip()


    if text:

        documents.append(text)


print(

    f"Loaded {len(documents)} documents."

)


# --------------------------------
# Shuffle documents
# --------------------------------

random.seed(

    RANDOM_SEED

)


random.shuffle(

    documents

)


# --------------------------------
# Train / Validation split
# --------------------------------

train_documents = []

validation_documents = []


for document in documents:

    # Split document into paragraphs

    paragraphs = document.split(

        "\n\n"

    )


    paragraphs = [

        paragraph.strip()

        for paragraph in paragraphs

        if paragraph.strip()

    ]


    # Shuffle paragraphs

    random.shuffle(

        paragraphs

    )


    # Calculate validation size

    validation_count = max(

        1,

        int(

            len(paragraphs)
            * VALIDATION_SPLIT

        )

    )


    # Add 10% of this document to validation

    validation_documents.extend(

        paragraphs[

            :validation_count

        ]

    )


    # Add 90% of this document to training

    train_documents.extend(

        paragraphs[

            validation_count:

        ]

    )

# --------------------------------
# Combine text
# --------------------------------

train_text = "\n\n".join(

    train_documents

)


validation_text = "\n\n".join(

    validation_documents

)


# --------------------------------
# Save datasets
# --------------------------------

with open(

    TRAIN_FILE,

    "w",

    encoding="utf-8"

) as file:

    file.write(

        train_text

    )


with open(

    VALIDATION_FILE,

    "w",

    encoding="utf-8"

) as file:

    file.write(

        validation_text

    )


# --------------------------------
# Statistics
# --------------------------------

print()

print(

    "Dataset successfully built."

)


print(

    f"Training characters: "
    f"{len(train_text):,}"

)


print(

    f"Validation characters: "
    f"{len(validation_text):,}"

)


print(

    f"Training documents: "
    f"{len(train_documents)}"

)


print(

    f"Validation documents: "
    f"{len(validation_documents)}"

)