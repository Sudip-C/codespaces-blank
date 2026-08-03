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
    print("Loading tokenizer...")
    tokenizer = GPTTokenizer()

    # -------------------------------
    # Model
    # -------------------------------
    print("Loading model...")
    model = GPTModel(config)
    print("Loading weights...")
    model.load_state_dict(

        torch.load(

            "best_model_32.pt",

            map_location=device

        )

    )
    print("Moving model to device...")
    model.to(device)

    model.eval()

    # -------------------------------
    # RAG Pipeline
    # -------------------------------
    print("Creating RAG pipeline...")
    pipeline = RAGPipeline(

        model,

        tokenizer,

        device

    )
    print("Ready!")
    # -------------------------------
    # Chat Loop
    # -------------------------------

    while True:

        question = input(

            "\nAsk a question (or type 'exit'): "

        )

        if question.lower() == "exit":

            break

##---------------
        print("Retrieving context...")

        retrieved = pipeline.retrieve(question)

        print("Context retrieved.")

        context = pipeline.build_context(retrieved)

        prompt = pipeline.build_prompt(
            question,
            context
        )

        print("Prompt built.")

        print("Generating answer...")

        answer = pipeline.generate_answer(prompt)

        print("Answer generated.")

        result = {
            "answer": answer,
            "sources": retrieved
        }


##---------------
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