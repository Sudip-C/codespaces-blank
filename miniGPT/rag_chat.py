import torch

from config import GPTConfig
from model.gpt import GPTModel
from model.tokenizer import GPTTokenizer

from rag.rag_pipeline import RAGPipeline


def main():

    # -------------------------------
    # Configuration
    # -------------------------------

    config = GPTConfig()

    device = config.device

    # -------------------------------
    # Tokenizer
    # -------------------------------

    tokenizer = GPTTokenizer()

    # -------------------------------
    # Model
    # -------------------------------

    model = GPTModel(config)

    model.load_state_dict(

        torch.load(

            "best_model_32.pt",

            map_location=device

        )

    )

    model.to(device)

    model.eval()

    # -------------------------------
    # RAG Pipeline
    # -------------------------------

    pipeline = RAGPipeline(

        model,

        tokenizer,

        device

    )

    # -------------------------------
    # Chat Loop
    # -------------------------------

    while True:

        question = input(

            "\nAsk a question (or type 'exit'): "

        )

        if question.lower() == "exit":

            break

        result = pipeline.ask(

            question

        )

        print()

        print("=" * 60)

        print("ANSWER")

        print("=" * 60)

        print()

        print(

            result["answer"]

        )

        print()

        print("=" * 60)

        print("SOURCES")

        print("=" * 60)

        print()

        for source in result["sources"]:

            print(

                source["source"]

            )


if __name__ == "__main__":

    main()