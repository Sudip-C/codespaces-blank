from pathlib import Path

import re


class TopicExtractor:

    def extract(self, document):

        source = document["source"]

        text = document["text"]


        # -----------------------------
        # Try extracting from title
        # -----------------------------

        lines = text.splitlines()

        for line in lines[:20]:

            line = line.strip()

            if len(line) < 4:
                continue

            if len(line) > 80:
                continue

            if line.isupper():

                return self.clean(line.title())

            if line.istitle():

                return self.clean(line)


        # -----------------------------
        # Fallback to filename
        # -----------------------------

        filename = Path(source).stem

        filename = filename.replace("_", " ")

        return self.clean(filename)


    def clean(self, topic):

        topic = re.sub(

            r"\s+",

            " ",

            topic

        )

        return topic.strip()