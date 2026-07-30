from pathlib import Path
import json
import numpy as np

from sentence_transformers import SentenceTransformer

from rag.document_loader import DocumentLoader
from rag.chunker import TextChunker


# ==========================================
# Configuration
# ==========================================

DATA_DIRECTORY = "data/raw"

CACHE_DIRECTORY = Path("rag/cache")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ==========================================
# Main
# ==========================================

def main():

    print()
    print("=" * 50)
    print("Building Embedding Cache")
    print("=" * 50)

    # --------------------------------------
    # Load documents
    # --------------------------------------

    print()
    print("Loading documents...")

    loader = DocumentLoader(
        DATA_DIRECTORY
    )

    documents = loader.load_documents()

    print(
        f"Loaded {len(documents)} documents."
    )

    # --------------------------------------
    # Chunk documents
    # --------------------------------------

    print()
    print("Chunking documents...")

    chunker = TextChunker(
        chunk_size=500,
        overlap=100
    )

    chunks = chunker.chunk_documents(
        documents
    )

    print(
        f"Created {len(chunks)} chunks."
    )

    # --------------------------------------
    # Load embedding model
    # --------------------------------------

    print()
    print("Loading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # --------------------------------------
    # Generate embeddings
    # --------------------------------------

    print()
    print("Generating embeddings...")

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = model.encode(
        texts,
        batch_size=16,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print()

    print(
        f"Embedding shape: {embeddings.shape}"
    )

    # --------------------------------------
    # Create cache directory
    # --------------------------------------

    CACHE_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------
    # Save embeddings
    # --------------------------------------

    np.save(
        CACHE_DIRECTORY / "embeddings.npy",
        embeddings
    )

    # --------------------------------------
    # Save chunks
    # --------------------------------------

    with open(
        CACHE_DIRECTORY / "chunks.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            chunks,
            file,
            indent=2,
            ensure_ascii=False
        )

    # --------------------------------------
    # Save metadata
    # --------------------------------------

    metadata = {

        "embedding_model": EMBEDDING_MODEL,

        "documents": len(documents),

        "chunks": len(chunks),

        "embedding_dimension": int(
            embeddings.shape[1]
        ),

        "chunk_size": 500,

        "overlap": 100

    }

    with open(
        CACHE_DIRECTORY / "metadata.json",
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2
        )

    # --------------------------------------
    # Finished
    # --------------------------------------

    print()
    print("=" * 50)
    print("Embedding Cache Created Successfully")
    print("=" * 50)

    print()
    print(
        f"Embeddings : {CACHE_DIRECTORY/'embeddings.npy'}"
    )

    print(
        f"Chunks      : {CACHE_DIRECTORY/'chunks.json'}"
    )

    print(
        f"Metadata    : {CACHE_DIRECTORY/'metadata.json'}"
    )


if __name__ == "__main__":

    main()