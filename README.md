🧠 IntelliDoc+: An NLP-Enhanced Multi-Document RAG System using LangChain & Deep Learning
📌 Overview
IntelliDoc+ is an advanced NLP + DL powered Retrieval-Augmented Generation (RAG) system that lets users upload multiple documents (PDFs) and ask natural language questions to retrieve precise, well-grounded answers from their content. It integrates semantic chunking, vector search, intent classification, LLM-based answer generation, and reranking with postprocessing — all wrapped in a modern Streamlit UI.

📐 Architecture Diagram
PDFs → NLP Preprocessing → Chunking → Embeddings → Vector Store
        ↑                     ↓                     ↓
      Classifier       Query Understanding → Retriever + Reranker
                                                 ↓
                                            LangChain + LLM
                                                 ↓
                                        Answer + Sources → UI


🚀 Features
📄 Multi-PDF upload and processing

✂️ Semantic-aware chunking using NLP (spaCy / Sentence-BERT)

🤖 Embedding with OpenAI / HuggingFace models

🔎 Fast vector search using FAISS or ChromaDB

🧠 Query classification and intent routing

🧬 Reranking using transformer-based similarity models (SBERT, BGE)

🧾 Answer generation using LLMs (OpenAI, Claude, Mistral, etc.)

🧹 Postprocessing (NER highlighting, grammar fix)

📊 Retrieval evaluation with Recall@k, latency, precision

🌐 Streamlit front-end for interaction


🧰 Tech Stack

| Component  | Tools / Models                                    |
| ---------- | ------------------------------------------------- |
| LLMs       | OpenAI GPT-3.5 / Claude / Phi-3 / Mistral         |
| Embeddings | `all-MiniLM`, `instructor-xl`, `text-embedding-3` |
| Vector DB  | FAISS / Chroma                                    |
| NLP        | spaCy, NLTK, Transformers                         |
| UI         | Streamlit                                         |
| Evaluation | Custom + `RAGAS` (optional)                       |

📁 Folder Structure
project_root/
├── app.py                  # Streamlit UI
├── main.py                 # Pipeline orchestrator
├── loader.py               # PDF loader logic
├── chunker.py              # NLP chunker (NER, POS, lemmatization)
├── embedder.py             # Embedding generator
├── vector_store.py         # Vector DB logic
├── classifier.py           # DL-based query classifier
├── reranker.py             # SBERT / BGE reranker
├── rag_chain.py            # LangChain-based chain logic
├── postprocessor.py        # NER-based and grammar postprocessing
├── evaluation.py           # Recall@k, F1, latency, etc.
├── nlp_utils.py            # Utility functions using spaCy/NLTK
├── requirements.txt
├── README.md
├── data/
│   ├── uploads/            # Uploaded PDFs
│   └── preprocessed/       # Cleaned, chunked data
├── vector_db/              # Serialized vector store
└── assets/                 # Flowcharts, images, docs


✅ How It Works
Upload PDFs

Text is extracted → preprocessed → semantically chunked

Chunks embedded and stored in FAISS

User types a question → query classified

Retriever fetches relevant chunks

Reranker reorders by relevance

LLM generates the final answer

NER tags, grammar fix, and sources are added

Result shown in Streamlit UI


🧪 Evaluation
Recall@k: % of relevant chunks in top-k results

Latency: Response time breakdown (retriever vs LLM)

Manual QA: Evaluate factual correctness

Precision: Answers matching gold reference sets

📌 Future Extensions
 Add voice-based input via Whisper

 Replace OpenAI with local models (e.g., Mistral, Phi-3)

 Document summarization before embedding

 Metadata filters (e.g., by source, page, section)

