from rag.document_loader import DocumentLoader

from instruction.topic_extractor import TopicExtractor

from instruction.qa_generator import QAGenerator


loader = DocumentLoader("data/raw")

documents = loader.load_documents()

extractor = TopicExtractor()

generator = QAGenerator()


document = documents[0]

topic = extractor.extract(document)

examples = generator.generate(
    topic,
    document
)

print(f"Generated {len(examples)} examples\n")

for example in examples[:5]:

    print("=" * 60)

    print(example["instruction"])

    print()

    print(example["response"][:250])

    print()