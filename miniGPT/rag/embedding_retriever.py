import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:

    def __init__(
        self,
        cache_directory="rag/cache",
        embedding_model="all-MiniLM-L6-v2"
    ):

        self.cache_directory = Path(
            cache_directory
        )

        print()

        print("Loading embedding cache...")

        self.embeddings = np.load(

            self.cache_directory /
            "embeddings.npy"

        )

        with open(

            self.cache_directory /
            "chunks.json",

            "r",

            encoding="utf-8"

        ) as file:

            self.chunks = json.load(
                file
            )

        print(

            f"Loaded {len(self.chunks)} chunks."
        )

        print()

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            embedding_model
        )

    # -----------------------------------

    def encode_query(

        self,

        query

    ):

        return self.model.encode(

            query,

            convert_to_numpy=True,

            normalize_embeddings=True

        )

    # -----------------------------------

    def similarity_search(

        self,

        query_embedding,

        top_k=3

    ):

        scores = (

            self.embeddings
            @ query_embedding

        )

        top_indices = np.argsort(

            scores

        )[::-1][:top_k]

        results = []

        for index in top_indices:

            results.append(

                {

                    "source":

                        self.chunks[index]["source"],

                    "text":

                        self.chunks[index]["text"],

                    "score":

                        float(scores[index])

                }

            )

        return results

    # -----------------------------------

    def retrieve(

        self,

        query,

        top_k=3

    ):

        query_embedding = self.encode_query(

            query

        )

        return self.similarity_search(

            query_embedding,

            top_k

        )