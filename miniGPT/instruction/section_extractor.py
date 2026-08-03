import re


class SectionExtractor:

    def __init__(

        self,

        min_words=80,

        max_words=250

    ):

        self.min_words = min_words

        self.max_words = max_words


    # --------------------------------------

    def split(

        self,

        text

    ):

        paragraphs = re.split(

            r"\n\s*\n",

            text

        )

        sections = []

        current = []


        for paragraph in paragraphs:

            paragraph = paragraph.strip()

            if not paragraph:

                continue


            current.extend(

                paragraph.split()

            )


            if len(current) >= self.min_words:

                sections.append(

                    " ".join(

                        current[:self.max_words]

                    )

                )

                current = current[self.max_words:]


        if current:

            sections.append(

                " ".join(current)

            )


        return sections