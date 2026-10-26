"""Compatibility import; new integrations should use llm_service."""

from .llm_service import answer_question, sanitize_context

__all__ = ["answer_question", "sanitize_context"]
