# vector_store.py

import os
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# Constants
VECTOR_DB_PATH = "vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def load_vector_store():
    """
    Loads the FAISS vector store from disk.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    if not os.path.exists(VECTOR_DB_PATH):
        raise FileNotFoundError(f"Vector store not found at {VECTOR_DB_PATH}. Run embedder.py first.")
    return FAISS.load_local(VECTOR_DB_PATH, embeddings, allow_dangerous_deserialization=True)

if __name__ == "__main__":
    db = load_vector_store()
    print(f"✅ Loaded vector store with {db.index.ntotal} vectors.")
