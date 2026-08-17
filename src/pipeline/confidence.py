from typing import List, Dict, Tuple
from src.telemetry import get_tracer

tracer = get_tracer()

class ConfidenceEvaluator:
    def __init__(self, alpha_threshold: float = 0.35):
        self.alpha_threshold = alpha_threshold

    def evaluate(self, reranked_chunks: List[Dict]) -> Tuple[bool, float, str]:
        with tracer.start_as_current_span("confidence.evaluator") as span:
            span.set_attribute("evaluator.alpha_threshold", self.alpha_threshold)

            if not reranked_chunks:
                span.set_attribute("evaluator.decision", "SHORT_CIRCUIT")
                span.set_attribute("evaluator.reason", "NO_CONTEXT_RETRIEVED")
                return False, 0.0, "NO_CONTEXT_RETRIEVED"

            top_score = reranked_chunks[0].get("relevance_score", 0.0)
            span.set_attribute("evaluator.top_score", top_score)

            if top_score < self.alpha_threshold:
                span.set_attribute("evaluator.decision", "SHORT_CIRCUIT")
                span.set_attribute("evaluator.reason", "RELEVANCE_BELOW_THRESHOLD")
                return False, top_score, f"RELEVANCE_BELOW_THRESHOLD ({top_score:.3f} < {self.alpha_threshold})"

            span.set_attribute("evaluator.decision", "PASSED")
            span.set_attribute("evaluator.reason", "CONTEXT_VERIFIED")
            return True, top_score, "CONTEXT_VERIFIED"