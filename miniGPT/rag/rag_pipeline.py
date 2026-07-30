import torch

from rag.embedding_retriever import EmbeddingRetriever


class RAGPipeline:

    def __init__(

        self,

        model,

        tokenizer,

        device,

        top_k=3

    ):

        self.model = model

        self.tokenizer = tokenizer

        self.device = device

        self.top_k = top_k

        self.retriever = EmbeddingRetriever()

    # ----------------------------------

    def retrieve(

        self,

        question

    ):

        return self.retriever.retrieve(

            question,

            top_k=self.top_k

        )

    # ----------------------------------

    def build_context(

        self,

        retrieved_chunks

    ):

        context = "\n\n".join(

            [

                chunk["text"]

                for chunk in retrieved_chunks

            ]

        )

        return context

    # ----------------------------------

    def build_prompt(

        self,

        question,

        context

    ):

        prompt = f"""Context:

{context}

Question:

{question}

Answer:
"""

        return prompt

    # ----------------------------------

    def generate_answer(

        self,

        prompt,

        max_new_tokens=150,

        temperature=0.8,

        top_k=40

    ):

        input_ids = self.tokenizer.encode(

            prompt

        )

        input_ids = torch.tensor(

            [input_ids],

            dtype=torch.long,

            device=self.device

        )

        output_ids = self.model.generate(

            input_ids,

            max_new_tokens=max_new_tokens,

            temperature=temperature,

            top_k=top_k

        )

        # Only decode newly generated tokens

        input_length = input_ids.shape[1]

        generated_tokens = output_ids[0][

            input_length:

        ]

        answer = self.tokenizer.decode(

            generated_tokens.tolist()

        )

        return answer.strip()

    # ----------------------------------

    def ask(

        self,

        question

    ):

        retrieved_chunks = self.retrieve(

            question

        )

        context = self.build_context(

            retrieved_chunks

        )

        prompt = self.build_prompt(

            question,

            context

        )

        answer = self.generate_answer(

            prompt

        )

        return {

            "question": question,

            "answer": answer,

            "sources": retrieved_chunks,

            "context": context,

            "prompt": prompt

        }