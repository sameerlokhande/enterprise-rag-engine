# ⚡ Enterprise RAG Engine

An enterprise-grade, secured, and evaluated **Retrieval-Augmented Generation (RAG)** pipeline powered by local LLMs via **Ollama**, **Qdrant**, **LangGraph**, and **Ragas**. 

This system features real-time agentic thought-process tracking, input security guardrails, semantic caching, confidence short-circuiting, clean source citations, and automated local quality evaluations.

---

## 🌟 Key Features

* **🛡️ Input Guardrails & PII Sanitization**: Validates queries for prompt safety and automatically sanitizes sensitive PII data before hitting vector stores or LLMs.
* **⚡ Semantic Query Caching**: High-speed, session-aware caching layer to serve repeated or semantically similar queries instantly.
* **📄 Robust Multi-Format Document Ingestion**: LangGraph-driven ingestion pipeline supporting `.pdf`, `.docx`, and `.txt` files with automated text extraction and chunking.
* **🔍 High-Precision Retrieval & Confidence Thresholding**: Qdrant Vector DB search coupled with a `ConfidenceEvaluator` that short-circuits out-of-domain queries when retrieved context relevance falls below threshold.
* **🧠 Real-Time Thought Process UI**: Streamlit `st.status` widget showing step-by-step execution details (Safety Check ➔ Cache ➔ Vector DB ➔ Confidence ➔ Generation ➔ Quality Audit) in real time.
* **📊 Local Ragas Quality Auditing**: Calculates **Faithfulness** (hallucination detection) and **Answer Relevancy** locally using Ollama without sending data to third-party evaluation APIs.
* **📖 Clean Citation UX**: Clean, non-cluttered user interface that suppresses internal vector scores and auto-hides citations when context is missing.

---

## 🏗️ Architecture & Pipeline Flow


               +----------------------------------+
               |        User Input Query          |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  1. Security & Guardrail Check   |  ---> [Reject unsafe / redact PII]
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  2. Session Semantic Cache       |  ---> [Cache Hit -> Return Response]
               +----------------------------------+
                                |
                                v (Cache Miss)
               +----------------------------------+
               |  3. Qdrant Vector DB Search      |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  4. Relevance Threshold Check    |  ---> [Low Score -> Short Circuit]
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  5. Grounded LLM Generation      |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  6. Local Ragas Evaluation       |
               +----------------------------------+
                                |
                                v
               +----------------------------------+
               |  Streamlit UI Output & Citations |
               +----------------------------------+

📂 Project Structure
enterprise-rag-engine/
├── app.py                      # Main Streamlit Web Application UI
├── requirements.txt            # Python Dependencies
├── README.md                   # Project Documentation
└── src/
    ├── evaluation/
    │   └── ragas_eval.py       # Ragas Evaluator wrapper (Ollama-backed)
    ├── ingestion/
    │   └── ingestion_graph.py  # LangGraph ingestion & processing pipeline
    ├── pipeline/
    │   ├── confidence.py       # Confidence thresholding & short-circuiting
    │   └── generator.py        # Grounded LLM response synthesis
    ├── retrieval/
    │   ├── cache.py            # Session-aware Semantic Cache
    │   └── vector_store.py     # Qdrant Vector DB client wrapper
    ├── security/
    │   └── guardrails.py       # Input safety & PII sanitization
    └── telemetry/
        └── __init__.py         # OpenTelemetry tracing setup


🚀 Getting Started
Prerequisites
Python 3.10+ installed

Ollama running locally

1. Clone the Repository & Set Up Virtual Environment
git clone [https://github.com/your-username/enterprise-rag-engine.git](https://github.com/your-username/enterprise-rag-engine.git)
cd enterprise-rag-engine

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

2. Install Dependencies

pip install -r requirements.txt

Note: If requirements.txt is missing, install the core requirements directly:
pip install streamlit qdrant-client langchain-ollama ragas datasets langchain-community opentelemetry-api

3. Pull Required Local Models via Ollama
Make sure Ollama is running (ollama serve), then pull the LLM and Embedding models:

# LLM for Generation & Ragas evaluation
ollama pull qwen2.5:1.5b

# Embedding model for vector search & evaluation
ollama pull nomic-embed-text

(Optional: You can also pull bge-large or all-minilm depending on your ragas_eval.py configuration).

4. Launch Qdrant Vector Store
If using Docker for Qdrant:
docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant

(If running Qdrant in-memory or embedded mode, this step is handled automatically).

5. Run the Application
streamlit run app.py
Open http://localhost:8501 in your browser.

💻 Usage Guide
Upload Documents: Use the sidebar to upload .pdf, .docx, or .txt files.

Process File: Click Process Document to sanitize and index vectors into Qdrant.

Ask Questions: Type your query into the chat input.

Inspect Thought Process: Expand the 🧠 Bot Thought Process container to observe step-by-step execution times and pipeline decisions.

Review Quality Audit: Check the automatically generated Faithfulness and Answer Relevancy scores below the response.

⚙️ Configuration & Model Tweaks
LLM & Embeddings: Updated in src/evaluation/ragas_eval.py and src/pipeline/generator.py:

Default LLM: qwen2.5:1.5b

Default Embeddings: nomic-embed-text

Confidence Threshold: Adjusted in src/pipeline/confidence.py (Default: 0.50).

Cache Threshold: Adjusted in src/retrieval/cache.py (Default: 0.92).

