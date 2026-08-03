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

        self.definition = (
            DEFINITION_TEMPLATES
            + BEGINNER_TEMPLATES
        )

        self.purpose = PURPOSE_TEMPLATES

        self.application = APPLICATION_TEMPLATES

        self.features = FEATURE_TEMPLATES

        self.working = WORKING_TEMPLATES

    # ----------------------------------

    def detect_template_group(

        self,

        section

    ):

        text = section.lower()

        if any(

            word in text

            for word in [

                "used",

                "application",

                "applications",

                "industry"

            ]

        ):

            return self.application


        if any(

            word in text

            for word in [

                "important",

                "benefit",

                "advantage",

                "purpose"

            ]

        ):

            return self.purpose


        if any(

            word in text

            for word in [

                "works",

                "working",

                "process",

                "steps",

                "algorithm"

            ]

        ):

            return self.working


        if any(

            word in text

            for word in [

                "feature",

                "characteristic",

                "property"

            ]

        ):

            return self.features


        return self.definition

    # ----------------------------------

    def generate(

        self,

        topic,

        section,

        source

    ):

        templates = self.detect_template_group(

            section

        )

        examples = []

        for template in templates:

            examples.append(

                {

                    "instruction": template.format(topic),

                    "response": section,

                    "source": source

                }

            )

        return examples