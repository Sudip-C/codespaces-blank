from pathlib import Path

from rag.document_loader import DocumentLoader

from instruction.topic_extractor import TopicExtractor
from instruction.section_extractor import SectionExtractor
from instruction.qa_generator import QAGenerator
from instruction.formatter import InstructionFormatter


class InstructionDatasetBuilder:

    def __init__(

        self,

        data_directory="data/raw",

        output_file="data/instruction_train.txt"

    ):

        self.data_directory = data_directory

        self.output_file = Path(output_file)

        self.loader = DocumentLoader(data_directory)

        self.topic_extractor = TopicExtractor()

        self.section_extractor = SectionExtractor()

        self.generator = QAGenerator()

        self.formatter = InstructionFormatter()

    # -------------------------------------------------

    def build(self):

        documents = self.loader.load_documents()

        print(f"Loaded {len(documents)} documents.")

        dataset = []

        total_sections = 0

        total_examples = 0

        for document in documents:

            topic = self.topic_extractor.extract(document)

            sections = self.section_extractor.split(

                document["text"]

            )

            total_sections += len(sections)

            for section in sections:

                examples = self.generator.generate(

                    topic,

                    section,

                    document["source"]

                )

                dataset.extend(examples)

                total_examples += len(examples)

        formatted = self.formatter.format_dataset(

            dataset

        )

        self.output_file.parent.mkdir(

            parents=True,

            exist_ok=True

        )

        self.output_file.write_text(

            formatted,

            encoding="utf-8"

        )

        print()

        print("=" * 50)

        print("Instruction Dataset Created")

        print("=" * 50)

        print(f"Sections : {total_sections}")

        print(f"Examples : {total_examples}")

        print(f"Saved to : {self.output_file}")

if __name__ == "__main__":

    builder = InstructionDatasetBuilder()

    builder.build()