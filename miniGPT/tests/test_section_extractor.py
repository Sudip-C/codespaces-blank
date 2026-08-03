from rag.document_loader import DocumentLoader

from instruction.section_extractor import SectionExtractor


loader = DocumentLoader(

    "data/raw"

)

documents = loader.load_documents()

extractor = SectionExtractor()

document = documents[0]

sections = extractor.split(

    document["text"]

)

print()

print(

    f"Sections: {len(sections)}"

)

print()

for i, section in enumerate(

    sections[:5],

    start=1

):

    print("=" * 70)

    print(

        f"Section {i}"

    )

    print()

    print(section[:400])

    print()