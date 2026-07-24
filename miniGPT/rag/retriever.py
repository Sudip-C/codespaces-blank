import re
import math
from collections import Counter


class KeywordRetriever:

    def __init__(self, chunks):

        self.chunks = chunks

        self.stop_words = {

            "a", "an", "the",
            "is", "are", "was", "were",
            "what", "who", "where",
            "when", "why", "how",
            "of", "in", "on", "at",
            "to", "for", "and",
            "or", "with", "from",
            "by", "as", "it",
            "this", "that"

        }

        self.document_frequency = {}

        self._build_index()


    def tokenize(self, text):

        text = text.lower()

        words = re.findall(

            r"\b[a-zA-Z]+\b",

            text

        )

        return [

            word

            for word in words

            if word not in self.stop_words

        ]


    def _build_index(self):

        total_chunks = len(

            self.chunks

        )

        word_document_counts = Counter()


        for chunk in self.chunks:

            words = set(

                self.tokenize(

                    chunk["text"]

                )

            )


            for word in words:

                word_document_counts[word] += 1


        for word, count in (

            word_document_counts.items()

        ):

            self.document_frequency[word] = (

                math.log(

                    total_chunks / (1 + count)

                )

            )


    def score(self, query, chunk):

        query_words = self.tokenize(

            query

        )


        chunk_words = self.tokenize(

            chunk["text"]

        )


        chunk_word_counts = Counter(

            chunk_words

        )


        score = 0.0


        for word in query_words:

            if word in chunk_word_counts:

                tf = (

                    chunk_word_counts[word]

                )


                idf = (

                    self.document_frequency.get(

                        word,

                        0

                    )

                )


                score += tf * idf


        return score


    def retrieve(self, query, top_k=3):

        scored_chunks = []


        for index, chunk in enumerate(

            self.chunks

        ):

            score = self.score(

                query,

                chunk

            )


            scored_chunks.append(

                (

                    score,

                    index,

                    chunk

                )

            )


        scored_chunks.sort(

            key=lambda x: x[0],

            reverse=True

        )


        results = []

        seen_sources = set()


        for score, index, chunk in (

            scored_chunks

        ):

            if score <= 0:

                continue


            source = chunk["source"]


            if source in seen_sources:

                continue


            results.append(chunk)

            seen_sources.add(source)


            if len(results) >= top_k:

                break


        return results