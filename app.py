import os
import tempfile
import uuid
import streamlit as st

from src.ingestion.ingestion_graph import ingestion_pipeline
from src.retrieval.cache import SemanticCache
from src.retrieval.vector_store import QdrantVectorStore
from src.pipeline.confidence import ConfidenceEvaluator
from src.pipeline.generator import GroundedGenerator
from src.security.guardrails import SecurityGuard
from src.evaluation.ragas_eval import RagasEvaluator

st.set_page_config(page_title="Enterprise RAG Engine", page_icon="⚡", layout="wide")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource
def load_rag_components():
    cache = SemanticCache(similarity_threshold=0.92)
    vector_store = QdrantVectorStore()
    evaluator = ConfidenceEvaluator(threshold=0.50)
    generator = GroundedGenerator()
    security_guard = SecurityGuard()
    ragas_eval = RagasEvaluator()
    return cache, vector_store, evaluator, generator, security_guard, ragas_eval


cache, vector_store, evaluator, generator, security_guard, ragas_eval = load_rag_components()

st.title("⚡ Enterprise RAG Engine")

# Sidebar - Document Ingestion
with st.sidebar:
    st.header("📄 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload PDF, DOCX, or Text file",
        type=["pdf", "docx", "txt"]
    )

    if uploaded_file and st.button("Process Document", type="primary"):
        with st.spinner("Processing & Indexing Document..."):
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                res = ingestion_pipeline.invoke({"file_path": tmp_path})
                cache.clear_session(st.session_state.session_id)
                st.success(f"Successfully processed '{uploaded_file.name}'!")
            except Exception as e:
                st.error(f"Error processing document: {str(e)}")
            finally:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

    st.markdown("---")
    if st.button("Reset Session & Clear Cache"):
        cache.clear_session(st.session_state.session_id)
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()


# Chat UI
st.subheader("💬 Context Query Assistant")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Ask a question about the uploaded document..."):
    # Display user query
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        response_text = ""
        retrieved_chunks = []
        is_rejected = False
        eval_scores = None

        # Dynamic Thought Process Container (Replaces static spinner)
        with st.status("🧠 Bot Thought Process...", expanded=True) as status:
            # Step 1: Input Safety Guardrails
            status.update(label="🛡️ Verifying input safety & PII redaction...")
            is_safe, safety_msg = security_guard.check_input_safety(prompt)
            sanitized_prompt = security_guard.sanitize_text(prompt)
            st.write("✓ Input safety and PII check passed.")

            if not is_safe:
                status.update(label="🚨 Security Violation Detected", state="error", expanded=True)
                response_text = f"🚨 **Security Alert**: {safety_msg}"
            else:
                # Step 2: Session Cache Lookup
                status.update(label="⚡ Checking session cache...")
                cached_response = cache.get(sanitized_prompt, session_id=st.session_state.session_id)

                if cached_response:
                    status.update(label="⚡ Retrieved from Session Cache!", state="complete", expanded=False)
                    response_text = cached_response
                else:
                    # Step 3: Vector Store Query
                    status.update(label="🔍 Querying vector store for relevant context...")
                    retrieved_chunks = vector_store.search(sanitized_prompt, top_k=5)
                    st.write(f"• Fetched {len(retrieved_chunks)} context chunk(s) from Qdrant.")

                    # Step 4: Relevance Confidence Threshold
                    status.update(label="📊 Evaluating relevance confidence thresholds...")
                    eval_status, top_score = evaluator.evaluate(retrieved_chunks)

                    if eval_status == "SHORT_CIRCUIT":
                        status.update(label="⚠️ Query short-circuited: Low context relevance.", state="complete", expanded=False)
                        response_text = f"The uploaded document does not contain relevant information regarding '{sanitized_prompt}'."
                        is_rejected = True
                    else:
                        # Step 5: Grounded Answer Generation
                        status.update(label="🤖 Generating grounded response with LLM...")
                        response_text = generator.generate(sanitized_prompt, retrieved_chunks)
                        cache.set(sanitized_prompt, response_text, session_id=st.session_state.session_id)

                        is_rejected = (
                            "does not contain" in response_text.lower() or 
                            "not mentioned" in response_text.lower()
                        )

                        if is_rejected:
                            status.update(label="⚠️ Topic missing from document context.", state="complete", expanded=False)
                        else:
                            # Step 6: Ragas Metrics Evaluation
                            status.update(label="🛡️ Running Ragas quality evaluation (Faithfulness & Relevancy)...")
                            eval_scores = ragas_eval.evaluate_response(
                                sanitized_prompt, response_text, retrieved_chunks
                            )
                            st.write(f"• Faithfulness score: `{eval_scores['faithfulness']:.2f}`")
                            st.write(f"• Answer Relevancy score: `{eval_scores['answer_relevancy']:.2f}`")

                            status.update(label="✅ Processing complete!", state="complete", expanded=False)

        # Output Answer
        st.write(response_text)

        # Sources & Quality Metrics UI (Suppressed if topic is missing)
        if retrieved_chunks and not is_rejected:
            st.caption("Sources referenced:")
            for idx, chunk in enumerate(retrieved_chunks, 1):
                chunk_text = chunk.get("text") or chunk.get("child_text", "")
                doc_id = chunk.get("doc_id", "Document")

                with st.expander(f"Source [{idx}] — {doc_id}"):
                    st.write(chunk_text)

            if eval_scores:
                st.caption(
                    f"🛡️ **Quality Audit**: Faithfulness = `{eval_scores['faithfulness']:.2f}` | "
                    f"Answer Relevancy = `{eval_scores['answer_relevancy']:.2f}`"
                )

    st.session_state.messages.append({"role": "assistant", "content": response_text})