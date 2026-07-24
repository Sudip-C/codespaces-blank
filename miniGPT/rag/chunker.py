class TextChunker:

    def __init__(

        self,

        chunk_size=500,

        overlap=100

    ):

        self.chunk_size = chunk_size

        self.overlap = overlap


    def chunk_document(

        self,

        document

    ):

        text = document["text"]

        source = document["source"]


        words = text.split()


        chunks = []


        start = 0


        while start < len(words):


            end = (

                start

                + self.chunk_size

            )


            chunk_words = words[

                start:end

            ]


            chunk_text = " ".join(

                chunk_words

            )


            chunks.append(

                {

                    "source": source,

                    "text": chunk_text

                }

            )


            start = (

                end

                - self.overlap

            )


        return chunks


    def chunk_documents(

        self,

        documents

    ):

        all_chunks = []


        for document in documents:

            chunks = self.chunk_document(

                document

            )


            all_chunks.extend(

                chunks

            )


        return all_chunks