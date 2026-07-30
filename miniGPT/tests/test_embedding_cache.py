

from rag.embedding_retriever import EmbeddingRetriever

retriever = EmbeddingRetriever()

results = retriever.retrieve(

    "What is artificial intelligence?",

    top_k=3

)

print()

for i, result in enumerate(results, start=1):

    print("-" * 60)

    print(f"Result {i}")

    print(f"Score : {result['score']:.4f}")

    print(f"Source: {result['source']}")

    print()

    print(result["text"][:500])

    print()