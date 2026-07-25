import torch

from config import GPTConfig

from model.gpt import GPTModel

from model.tokenizer import GPTTokenizer

from rag.document_loader import DocumentLoader

from rag.chunker import TextChunker

from rag.rag_pipeline import RAGPipeline


def main():

    # --------------------------------
    # Configuration
    # --------------------------------

    config = GPTConfig()

    device = config.device


    # --------------------------------
    # Load tokenizer
    # --------------------------------

    tokenizer = GPTTokenizer()


    # --------------------------------
    # Load model
    # --------------------------------

    model = GPTModel(

        config

    )


    model.load_state_dict(

        torch.load(

            "best_model.pt",

            map_location=device

        )

    )


    model.to(

        device

    )


    model.eval()


    # --------------------------------
    # Load knowledge
    # --------------------------------

    loader = DocumentLoader(

        "data/raw"

    )


    documents = (

        loader.load_documents()

    )


    # --------------------------------
    # Create chunks
    # --------------------------------

    chunker = TextChunker(

        chunk_size=500,

        overlap=100

    )


    chunks = (

        chunker.chunk_documents(

            documents

        )

    )


    print(

        f"Loaded {len(chunks)} chunks"

    )


    # --------------------------------
    # Create RAG pipeline
    # --------------------------------

    rag = RAGPipeline(

        chunks,

        model,

        tokenizer,

        device

    )


    # --------------------------------
    # Chat loop
    # --------------------------------

    while True:


        question = input(

            "\nAsk a question "

            "(or type 'exit'): "

        )


        if question.lower() == "exit":

            break


        # Retrieve context

        context, results = (

            rag.retrieve_context(

                question,

                top_k=3

            )

        )


        # Build prompt

        prompt = (

            rag.build_prompt(

                question,

                context

            )

        )


        print()

        print(

            "Retrieved from:"

        )


        for result in results:

            print(

                result["source"]

            )


        print()

        print(

            "Context:"

        )


        print(

            context[:1000]

        )


        print()

        print(

            "Prompt sent to model:"

        )


        print(

            prompt

        )


if __name__ == "__main__":

    main()