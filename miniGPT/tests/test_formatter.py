from instruction.formatter import InstructionFormatter

formatter = InstructionFormatter()

example = {

    "instruction": "What is Artificial Intelligence?",

    "response": (
        "Artificial intelligence is a field of "
        "computer science."
    )

}

print(

    formatter.format_example(

        example

    )

)