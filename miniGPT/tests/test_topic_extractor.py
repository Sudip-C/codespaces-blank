from instruction.topic_extractor import TopicExtractor
from rag.document_loader import DocumentLoader


loader = DocumentLoader("data/raw")

documents = loader.load_documents()

extractor = TopicExtractor()

for document in documents:

    topic = extractor.extract(document)

    print(document["source"])

    print("Topic:", topic)

    print("-" * 50)