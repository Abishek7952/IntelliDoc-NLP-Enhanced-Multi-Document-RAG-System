# loader.py

import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.schema import Document

def load_pdfs_from_folder(folder_path: str) -> list[Document]:
    """
    Loads all PDF files in the given folder and returns a list of LangChain Document objects.
    Each document will contain metadata like source file and page number.
    """
    all_docs = []

    for filename in os.listdir(folder_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(folder_path, filename)
            try:
                loader = PyMuPDFLoader(file_path)
                docs = loader.load()
                print(f"✅ Loaded {len(docs)} pages from {filename}")
                all_docs.extend(docs)
            except Exception as e:
                print(f"❌ Failed to load {filename}: {e}")

    return all_docs

# Optional test
if __name__ == "__main__":
    docs = load_pdfs_from_folder("data/uploads/meidtations.pdf")
    print(f"Total pages loaded: {len(docs)}")
