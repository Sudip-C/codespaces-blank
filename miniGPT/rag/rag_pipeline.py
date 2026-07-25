from rag.embedding_retriever import EmbeddingRetriever


class RAGPipeline:

    def __init__(

        self,

        chunks,

        model,

        tokenizer,

        device,

        max_context_length=2000

    ):

        self.retriever = (

            EmbeddingRetriever(

                chunks

            )

        )


        self.model = model

        self.tokenizer = tokenizer

        self.device = device

        self.max_context_length = (

            max_context_length

        )


    def retrieve_context(

        self,

        question,

        top_k=3

    ):

        results = (

            self.retriever.retrieve(

                question,

                top_k=top_k

            )

        )


        context_parts = []


        current_length = 0


        for result in results:

            text = result["text"]


            remaining = (

                self.max_context_length

                - current_length

            )


            if remaining <= 0:

                break


            text = text[:remaining]


            context_parts.append(text)


            current_length += len(text)


        context = "\n\n".join(

            context_parts

        )


        return context, results


    def build_prompt(

        self,

        question,

        context

    ):

        prompt = f"""

Context:

{context}


Question:

{question}


Answer:

"""


        return prompt