from typing import List, Dict, Any, Tuple
from src.telemetry import get_tracer

tracer = get_tracer()


class ConfidenceEvaluator:
    def __init__(self, threshold: float = 0.50):
        self.threshold = threshold

    def evaluate(self, chunks: List[Dict[str, Any]]) -> Tuple[str, float]:
        """Evaluates whether the top candidate chunk meets the minimum relevance score threshold."""
        with tracer.start_as_current_span("confidence.evaluator") as span:
            span.set_attribute("evaluator.threshold", self.threshold)

            if not chunks:
                span.set_attribute("evaluator.decision", "SHORT_CIRCUIT")
                span.set_attribute("evaluator.top_score", 0.0)
                return "SHORT_CIRCUIT", 0.0

            top_chunk = chunks[0]
            # Supports both 'relevance_score' (from FlashRank) and 'score' (from vector store)
            top_score = top_chunk.get("relevance_score", top_chunk.get("score", 0.0))
            span.set_attribute("evaluator.top_score", top_score)

            if top_score >= self.threshold:
                span.set_attribute("evaluator.decision", "PASSED")
                return "PASSED", top_score
            else:
                span.set_attribute("evaluator.decision", "SHORT_CIRCUIT")
                return "SHORT_CIRCUIT", top_score