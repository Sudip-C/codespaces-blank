from rag.document_loader import DocumentLoader

from instruction.topic_extractor import TopicExtractor

from instruction.section_extractor import SectionExtractor

from instruction.qa_generator import QAGenerator


loader = DocumentLoader("data/raw")

documents = loader.load_documents()

topic_extractor = TopicExtractor()

section_extractor = SectionExtractor()

generator = QAGenerator()


document = documents[0]

topic = topic_extractor.extract(document)

sections = section_extractor.split(document["text"])


print(f"Topic: {topic}")

print(f"Sections: {len(sections)}")

print()


examples = generator.generate(

    topic,

    sections[0],

    document["source"]

)


for example in examples:

    print("=" * 60)

    print(example["instruction"])

    print()

    print(example["response"][:200])

    print()