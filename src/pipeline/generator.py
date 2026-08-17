import time
from typing import List, Dict, Any
import ollama

from src.telemetry import get_tracer

tracer = get_tracer()


class GroundedGenerator:
    def __init__(self, model_name: str = "qwen2.5:1.5b", host: str = "http://localhost:11434"):
        self.model_name = model_name
        self.client = ollama.Client(host=host)

    def generate(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Constructs a strictly grounded prompt with zero-tolerance for unmentioned concepts."""
        with tracer.start_as_current_span("ollama.generation") as span:
            start_time = time.time()

            if not context_chunks:
                return f"The uploaded document does not contain any information about '{query}'."

            context_blocks = []
            citations = []
            for idx, chunk in enumerate(context_chunks, 1):
                text = chunk.get("text") or chunk.get("child_text") or ""
                doc_id = chunk.get("doc_id", "Unknown")
                context_blocks.append(f"[{idx}] (Doc ID: {doc_id}):\n{text}")
                citations.append(f"[{idx}] Doc {doc_id}")

            formatted_context = "\n\n".join(context_blocks)

            prompt = f"""You are a strict, factual Enterprise RAG Assistant.

STRICT GROUNDING RULES:
1. First, check if the main subject of the question is mentioned anywhere in the Context below.
2. If the concept, term, or topic (e.g., HTML, CSS, JavaScript, etc.) is NOT mentioned or discussed in the Context, respond ONLY with: "The uploaded document does not contain any information about this topic."
3. DO NOT use pre-trained world knowledge or general knowledge under any circumstances.
4. If the context contains ONLY question titles or section headers without explanations, state: "The document lists questions regarding this topic, but provides no explanatory content."

Context:
{formatted_context}

Question: {query}
Answer:"""

            span.set_attribute("llm.model", self.model_name)
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.citations", ", ".join(citations))

            try:
                response = self.client.generate(
                    model=self.model_name,
                    prompt=prompt
                )
                answer_text = response.get("response", "").strip()
            except Exception as e:
                answer_text = f"Error communicating with local Ollama service ({str(e)}). Please ensure Ollama is running (`ollama serve`)."

            latency_ms = (time.time() - start_time) * 1000
            span.set_attribute("llm.latency_ms", latency_ms)

            return answer_text

    def generate_response(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        return self.generate(query, context_chunks)

    def run(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        return self.generate(query, context_chunks)