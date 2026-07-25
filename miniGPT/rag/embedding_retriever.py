import numpy as np

from sentence_transformers import SentenceTransformer


class EmbeddingRetriever:

    def __init__(

        self,

        chunks,

        model_name=(

            "all-MiniLM-L6-v2"

        )

    ):

        self.chunks = chunks


        print(

            "Loading embedding model..."

        )


        self.model = (

            SentenceTransformer(

                model_name

            )

        )


        print(

            "Creating chunk embeddings..."

        )


        texts = [

            chunk["text"]

            for chunk in chunks

        ]


        self.embeddings = (

            self.model.encode(

                texts,

                convert_to_numpy=True,

                show_progress_bar=True

            )

        )


        # Normalize embeddings

        norms = np.linalg.norm(

            self.embeddings,

            axis=1,

            keepdims=True

        )


        self.embeddings = (

            self.embeddings / norms

        )


    def retrieve(

        self,

        query,

        top_k=3

    ):


        query_embedding = (

            self.model.encode(

                [query],

                convert_to_numpy=True

            )[0]

        )


        query_norm = np.linalg.norm(

            query_embedding

        )


        query_embedding = (

            query_embedding / query_norm

        )


        similarities = (

            self.embeddings

            @ query_embedding

        )


        top_indices = np.argsort(

            similarities

        )[::-1][:top_k]


        results = []


        for index in top_indices:

            result = (

                self.chunks[index]

                .copy()

            )


            result["similarity"] = (

                float(

                    similarities[index]

                )

            )


            results.append(

                result

            )


        return results