# chunker.py

import os
import pickle
import spacy
from langchain.schema import Document
from loader import load_pdfs_from_folder  # from your loader module

nlp = spacy.load("en_core_web_sm")

def process_text(text):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

def chunk_documents(documents, chunk_size=500, overlap=50):
    chunks = []
    for doc in documents:
        sents = process_text(doc.page_content)

        chunk = ""
        for sent in sents:
            if len(chunk) + len(sent) <= chunk_size:
                chunk += " " + sent
            else:
                chunks.append(Document(page_content=chunk.strip(), metadata=doc.metadata))
                chunk = sent  # carry over
        if chunk:
            chunks.append(Document(page_content=chunk.strip(), metadata=doc.metadata))
    return chunks

if __name__ == "__main__":
    docs = load_pdfs_from_folder("data/uploads")
    print(f"🔹 Loaded {len(docs)} raw documents")

    chunks = chunk_documents(docs)
    print(f"✅ Generated {len(chunks)} NLP-based text chunks")

    # ✅ Save as Pickle for use in embedder.py
    os.makedirs("data/preprocessed", exist_ok=True)
    with open("data/preprocessed/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    print("📦 Chunks saved to data/preprocessed/chunks.pkl")
