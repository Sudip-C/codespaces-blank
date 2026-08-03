class InstructionFormatter:

    def __init__(

        self,

        eos_token="<|endoftext|>"

    ):

        self.eos_token = eos_token

    # -----------------------------------------

    def format_example(

        self,

        example

    ):

        instruction = example["instruction"].strip()

        response = example["response"].strip()

        return (
            f"### Instruction:\n"
            f"{instruction}\n\n"
            f"### Response:\n"
            f"{response}\n\n"
            f"{self.eos_token}\n"
        )

    # -----------------------------------------

    def format_dataset(

        self,

        examples

    ):

        formatted = []

        for example in examples:

            formatted.append(

                self.format_example(

                    example

                )

            )

        return "\n".join(formatted)