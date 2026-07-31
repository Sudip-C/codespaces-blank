from instruction.templates import (
    DEFINITION_TEMPLATES,
    BEGINNER_TEMPLATES,
    PURPOSE_TEMPLATES,
    APPLICATION_TEMPLATES,
    FEATURE_TEMPLATES,
    WORKING_TEMPLATES
)


class QAGenerator:

    def __init__(self):

        self.templates = []

        self.templates.extend(DEFINITION_TEMPLATES)
        self.templates.extend(BEGINNER_TEMPLATES)
        self.templates.extend(PURPOSE_TEMPLATES)
        self.templates.extend(APPLICATION_TEMPLATES)
        self.templates.extend(FEATURE_TEMPLATES)
        self.templates.extend(WORKING_TEMPLATES)

    # ----------------------------------------

    def generate(
        self,
        topic,
        document,
        max_answer_words=250
    ):

        text = document["text"]

        words = text.split()

        answer = " ".join(
            words[:max_answer_words]
        )

        dataset = []

        for template in self.templates:

            dataset.append(

                {
                    "instruction": template.format(topic),

                    "response": answer,

                    "source": document["source"]
                }

            )

        return dataset