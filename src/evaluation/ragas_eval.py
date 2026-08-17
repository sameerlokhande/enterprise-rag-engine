import sys
import types
from typing import List, Dict, Any
from unittest.mock import MagicMock

# ==============================================================================
# Compatibility Patch:
# Fixes legacy Ragas imports for deprecated/removed LangChain VertexAI modules.
# ==============================================================================
for legacy_module in [
    "langchain_community.chat_models.vertexai",
    "langchain_community.embeddings.vertexai",
]:
    if legacy_module not in sys.modules:
        dummy_mod = types.ModuleType(legacy_module)
        dummy_mod.ChatVertexAI = MagicMock
        dummy_mod.VertexAIEmbeddings = MagicMock
        sys.modules[legacy_module] = dummy_mod

# ==============================================================================
# Standard Imports
# ==============================================================================
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

# Telemetry Import with Fallback
try:
    from src.telemetry import get_tracer
    tracer = get_tracer()
except Exception:
    import contextlib
    class DummyTracer:
        @contextlib.contextmanager
        def start_as_current_span(self, name):
            class DummySpan:
                def set_attribute(self, key, value): pass
            yield DummySpan()
    tracer = DummyTracer()


class RagasEvaluator:
    """Evaluates RAG generation quality using Ragas metrics (Faithfulness & Answer Relevancy)."""

    def __init__(
        self,
        model_name: str = "qwen2.5:1.5b",
        embedding_model: str = "nomic-embed-text",
        host: str = "http://localhost:11434"
    ):
        """Initializes Ragas evaluation wrappers with local Ollama models."""
        ollama_llm = ChatOllama(model=model_name, base_url=host)
        self.eval_llm = LangchainLLMWrapper(ollama_llm)

        ollama_embeddings = OllamaEmbeddings(model=embedding_model, base_url=host)
        self.eval_embeddings = LangchainEmbeddingsWrapper(ollama_embeddings)

    def evaluate_response(
        self,
        query: str,
        response: str,
        retrieved_chunks: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Calculates Faithfulness and Answer Relevancy scores for a single query and response pair."""
        with tracer.start_as_current_span("ragas.evaluation") as span:
            contexts = [
                chunk.get("text") or chunk.get("child_text") or chunk.get("page_content", "")
                for chunk in retrieved_chunks
            ]
            contexts = [c for c in contexts if c.strip()]

            if not contexts or not response.strip():
                span.set_attribute("ragas.status", "SKIPPED_EMPTY_CONTEXT_OR_RESPONSE")
                return {"faithfulness": 0.0, "answer_relevancy": 0.0}

            dataset_dict = {
                "question": [query],
                "answer": [response],
                "contexts": [contexts]
            }
            dataset = Dataset.from_dict(dataset_dict)

            try:
                eval_results = evaluate(
                    dataset=dataset,
                    metrics=[faithfulness, answer_relevancy],
                    llm=self.eval_llm,
                    embeddings=self.eval_embeddings,
                    raise_exceptions=False
                )

                # Safely parse score using DataFrame conversion
                df = eval_results.to_pandas()

                faithfulness_score = float(df["faithfulness"].iloc[0]) if "faithfulness" in df.columns else 0.0
                relevancy_score = float(df["answer_relevancy"].iloc[0]) if "answer_relevancy" in df.columns else 0.0

                scores = {
                    "faithfulness": round(faithfulness_score, 4),
                    "answer_relevancy": round(relevancy_score, 4)
                }

                span.set_attribute("ragas.faithfulness", scores["faithfulness"])
                span.set_attribute("ragas.answer_relevancy", scores["answer_relevancy"])
                span.set_attribute("ragas.status", "SUCCESS")

                return scores

            except Exception as e:
                span.set_attribute("ragas.status", "FAILED")
                span.set_attribute("ragas.error", str(e))
                return {"faithfulness": 0.0, "answer_relevancy": 0.0}