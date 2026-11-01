"""
Base Agent class and standardized AgentResponse schema for the AI Finance Assistant.
All six specialized financial agents inherit from BaseAgent to guarantee consistent
error handling, RAG integration, telemetry, source attribution, and regulatory disclaimers.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.core.config import AppConfig, LLMClient, logger
from src.data.market_provider import MarketDataProvider
from src.rag.retriever import FinanceRAGRetriever


@dataclass
class AgentResponse:
    """Standardized output schema returned by every specialized agent."""
    agent_id: str
    agent_name: str
    query: str
    response_text: str
    structured_data: Dict[str, Any] = field(default_factory=dict)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    follow_up_suggestions: List[str] = field(default_factory=list)
    confidence_score: float = 0.92
    execution_time_ms: float = 0.0
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "query": self.query,
            "response_text": self.response_text,
            "structured_data": self.structured_data,
            "citations": self.citations,
            "follow_up_suggestions": self.follow_up_suggestions,
            "confidence_score": round(self.confidence_score, 3),
            "execution_time_ms": round(self.execution_time_ms, 2),
            "error": self.error,
        }


class BaseAgent(ABC):
    """Abstract base class for all specialized financial agents."""

    def __init__(
        self,
        agent_id: str,
        agent_name: str,
        description: str,
        config: Optional[AppConfig] = None,
        llm_client: Optional[LLMClient] = None,
        rag_retriever: Optional[FinanceRAGRetriever] = None,
        market_provider: Optional[MarketDataProvider] = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.description = description
        self.config = config or AppConfig.load()
        self.llm = llm_client or LLMClient(self.config)
        self.rag = rag_retriever or FinanceRAGRetriever(
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap,
        )
        self.market = market_provider or MarketDataProvider(
            cache_ttl_seconds=self.config.market_cache_ttl_seconds
        )

    @abstractmethod
    def _execute(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        """Subclasses implement domain-specific processing here."""

    def run(self, query: str, context: Optional[Dict[str, Any]] = None) -> AgentResponse:
        """Executes the agent with timing telemetry and graceful error fallback."""
        start_t = time.perf_counter()
        ctx = context if context is not None else {}
        try:
            if not query or not query.strip():
                raise ValueError("Query cannot be empty.")
            response = self._execute(query.strip(), ctx)
            response.execution_time_ms = (time.perf_counter() - start_t) * 1000.0
            return response
        except Exception as exc:
            elapsed = (time.perf_counter() - start_t) * 1000.0
            logger.error("Agent %s encountered an error: %s", self.agent_name, exc)
            fallback_text = (
                f"### {self.agent_name} Notice\n\n"
                f"I encountered an issue processing that specific request (`{exc}`). "
                "Please verify your input parameters or try asking about a specific financial concept, "
                "ticker symbol (e.g., `VTI`, `AAPL`), portfolio allocation, or savings goal.\n\n"
                f"> *{self.config.disclaimer_text}*"
            )
            return AgentResponse(
                agent_id=self.agent_id,
                agent_name=self.agent_name,
                query=query,
                response_text=fallback_text,
                structured_data={"status": "fallback", "reason": str(exc)},
                citations=[],
                follow_up_suggestions=[
                    "What is the Bogleheads Three-Fund Portfolio?",
                    "Analyze my beginner balanced portfolio",
                    "Compare Roth IRA vs Traditional 401(k)",
                ],
                confidence_score=0.50,
                execution_time_ms=elapsed,
                error=str(exc),
            )
