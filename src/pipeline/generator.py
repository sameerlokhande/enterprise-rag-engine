import requests
import time
from typing import List, Dict
from src.telemetry import get_tracer

tracer = get_tracer()

class GroundedGenerator:
    def __init__(self, ollama_url: str = "http://localhost:11434/api/generate", model: str = "qwen2.5:0.5b"):
        self.ollama_url = ollama_url
        self.model = model

    def generate_response(self, query: str, chunks: List[Dict]) -> Dict:
        with tracer.start_as_current_span("ollama.generation") as span:
            context_str = ""
            for c in chunks:
                context_str += f"\n--- DOCUMENT ID: {c['child_id']} ---\n{c['text']}\n"

            prompt = f"""You are an enterprise AI assistant. Answer using ONLY the provided contexts below.
Rules:
1. Do NOT assume or extrapolate facts.
2. Every fact MUST end with its source citation in brackets, e.g., [Doc: <child_id>].
3. If context lacks sufficient proof, reply: "I cannot answer this based on documentation."

Contexts:
{context_str}

User Query: {query}
Answer:"""

            span.set_attribute("llm.model", self.model)
            span.set_attribute("llm.prompt_length", len(prompt))
            span.set_attribute("llm.chunks_injected", len(chunks))

            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0}
            }

            start = time.time()
            res = requests.post(self.ollama_url, json=payload).json()
            latency_ms = (time.time() - start) * 1000

            citations = [c["child_id"] for c in chunks]
            
            span.set_attribute("llm.latency_ms", round(latency_ms, 2))
            span.set_attribute("llm.response_length", len(res.get("response", "")))
            span.set_attribute("llm.citations", citations)

            return {
                "answer": res["response"],
                "citations": citations
            }