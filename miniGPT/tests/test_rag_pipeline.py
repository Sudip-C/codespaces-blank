from rag.rag_pipeline import RAGPipeline


pipeline = RAGPipeline()


result = pipeline.ask(

    "What is artificial intelligence?"

)


print()

print("=" * 60)

print("QUESTION")

print("=" * 60)

print()

print(

    result["question"]

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

print()

print("=" * 60)

print("PROMPT")

print("=" * 60)

print()

print(

    result["prompt"]

)