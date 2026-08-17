import re
import hashlib
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from src.telemetry import get_tracer

tracer = get_tracer()

@dataclass
class NormalizedDocument:
    doc_id: str
    clean_text: str
    raw_length: int
    clean_length: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class DocumentNormalizer:
    def __init__(self):
        self.zero_width_space_pattern = re.compile(r'[\u200b\u200c\u200d\ufeff]')
        self.excess_whitespace_pattern = re.compile(r'[ \t]+')
        self.excess_newlines_pattern = re.compile(r'\n{3,}')
        self.page_number_pattern = re.compile(r'(?i)^\s*page\s+\d+\s+of\s+\d+\s*$', re.MULTILINE)
        self.html_tag_pattern = re.compile(r'<[^>]+>')

    def normalize(self, raw_text: str, source_path: Optional[str] = None) -> NormalizedDocument:
        with tracer.start_as_current_span("ingestion.normalize_document") as span:
            if not raw_text or not raw_text.strip():
                span.record_exception(ValueError("Cannot normalize an empty document."))
                raise ValueError("Cannot normalize an empty document.")

            raw_length = len(raw_text)
            
            normalized = unicodedata.normalize('NFKC', raw_text)
            cleaned = self.zero_width_space_pattern.sub('', normalized)
            cleaned = self.html_tag_pattern.sub(' ', cleaned)
            cleaned = self.page_number_pattern.sub('', cleaned)
            cleaned = self.excess_whitespace_pattern.sub(' ', cleaned)
            clean_text = self.excess_newlines_pattern.sub('\n\n', cleaned).strip()

            seed = source_path if source_path else clean_text
            doc_id = f"DOC_{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:16]}"

            span.set_attribute("doc.id", doc_id)
            span.set_attribute("doc.raw_length", raw_length)
            span.set_attribute("doc.clean_length", len(clean_text))
            span.set_attribute("doc.compression_ratio", round(len(clean_text) / (raw_length or 1), 3))

            return NormalizedDocument(
                doc_id=doc_id,
                clean_text=clean_text,
                raw_length=raw_length,
                clean_length=len(clean_text),
                metadata={"source_path": source_path or "unknown"}
            )