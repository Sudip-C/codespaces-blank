from pathlib import Path


class DocumentLoader:

    def __init__(self, data_directory):

        self.data_directory = Path(
            data_directory
        )


    def load_documents(self):

        documents = []


        text_files = self.data_directory.rglob(
            "*.txt"
        )


        for file_path in text_files:

            text = file_path.read_text(
                encoding="utf-8"
            )


            text = text.strip()


            if text:

                documents.append(

                    {

                        "source": str(
                            file_path
                        ),

                        "text": text

                    }

                )


        return documents