# embedder.py
import os
os.environ["HF_HOME"] = "D:/hf_cache"

import pickle
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load preprocessed chunks
with open("data/preprocessed/chunks.pkl", "rb") as f:
    chunks = pickle.load(f)

print(f"[DEBUG] Number of chunks to embed: {len(chunks)}")
if not chunks:
    raise ValueError("No chunks to embed. Please run chunker.py first.")

# Embedding model
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

# Create FAISS vector store
db = FAISS.from_documents(chunks, embeddings)

# Save vector store
db.save_local("vector_db")
print("✅ Embedding and vector store creation complete.")
