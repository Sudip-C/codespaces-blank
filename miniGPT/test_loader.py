from rag.document_loader import DocumentLoader

from rag.chunker import TextChunker

from rag.retriever import KeywordRetriever


# -----------------------------
# Load documents
# -----------------------------

loader = DocumentLoader(

    "data/raw"

)


documents = loader.load_documents()


print(

    f"Loaded documents: "

    f"{len(documents)}"

)


# -----------------------------
# Create chunks
# -----------------------------

chunker = TextChunker(

    chunk_size=500,

    overlap=100

)


chunks = chunker.chunk_documents(

    documents

)


print(

    f"Total chunks: "

    f"{len(chunks)}"

)


# -----------------------------
# Create retriever
# -----------------------------

retriever = KeywordRetriever(

    chunks

)


# -----------------------------
# Test query
# -----------------------------

query = (

    "What is artificial intelligence?"

)


results = retriever.retrieve(

    query,

    top_k=3

)


# -----------------------------
# Display results
# -----------------------------

print()

print(

    "Query:"

)


print(

    query

)


print()

print(

    "Retrieved chunks:"

)


for index, result in enumerate(

    results,

    start=1

):


    print()

    print(

        f"--- Result {index} ---"

    )


    print(

        "Source:",

        result["source"]

    )


    print(

        result["text"][:1000]

    )