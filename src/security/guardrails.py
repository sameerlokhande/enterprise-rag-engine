import re
from typing import Tuple
from src.telemetry import get_tracer

tracer = get_tracer()


class SecurityGuard:
    def __init__(self):
        # Regular expressions for sensitive personal data
        self.email_pattern = re.compile(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+')
        self.phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        self.api_key_pattern = re.compile(r'\b(?:sk-[a-zA-Z0-9]{32,}|ghp_[a-zA-Z0-9]{36}|AIza[0-9A-Za-z-_]{35})\b')
        
        # Injection and jailbreak detection keywords
        self.injection_keywords = [
            "ignore previous instructions",
            "system prompt",
            "bypass guardrails",
            "you are now in developer mode",
            "dan mode",
            "forget your rules"
        ]

    def sanitize_text(self, text: str) -> str:
        """Scrubs PII (Emails, Phones, API Keys) before vector indexing or LLM submission."""
        with tracer.start_as_current_span("security.sanitize_text") as span:
            sanitized = self.email_pattern.sub("[REDACTED_EMAIL]", text)
            sanitized = self.phone_pattern.sub("[REDACTED_PHONE]", sanitized)
            sanitized = self.api_key_pattern.sub("[REDACTED_API_KEY]", sanitized)
            
            pii_detected = sanitized != text
            span.set_attribute("security.pii_redacted", pii_detected)
            return sanitized

    def check_input_safety(self, query: str) -> Tuple[bool, str]:
        """Inspects user input queries for prompt injection or jailbreak attempts."""
        with tracer.start_as_current_span("security.check_input_safety") as span:
            query_lower = query.lower()
            for pattern in self.injection_keywords:
                if pattern in query_lower:
                    span.set_attribute("security.jailbreak_attempt", True)
                    span.set_attribute("security.blocked_pattern", pattern)
                    return False, f"Query blocked due to potential prompt injection attempt: '{pattern}'"
            
            span.set_attribute("security.jailbreak_attempt", False)
            return True, "Passed safety check"