"""
Conversation and Workflow State Definitions for LangGraph Multi-Agent Orchestration.
Maintains multi-turn conversation history, user financial profile, active portfolio/goal
parameters, intermediate agent responses, and execution telemetry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TypedDict


class WorkflowState(TypedDict, total=False):
    """TypedDict representing the state flowing through the LangGraph StateGraph."""
    session_id: str
    user_query: str
    context: Dict[str, Any]
    user_profile: Dict[str, Any]
    conversation_history: List[Dict[str, Any]]
    classified_intents: List[str]
    routing_rationale: str
    routing_confidence: float
    agent_outputs: Dict[str, Dict[str, Any]]
    final_response: str
    citations: List[Dict[str, Any]]
    follow_up_suggestions: List[str]
    execution_trace: List[Dict[str, Any]]
    total_latency_ms: float
    error: Optional[str]


@dataclass
class ConversationSession:
    """Manages persistent session state and multi-turn context preservation across queries."""
    session_id: str = "default-session"
    user_profile: Dict[str, Any] = field(
        default_factory=lambda: {
            "risk_tolerance": "Moderate",
            "age": 32,
            "annual_income": 115000.0,
            "filing_status": "single",
            "experience_level": "Beginner",
            "active_portfolio_key": "beginner_balanced",
        }
    )
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_active_agents: List[str] = field(default_factory=list)
    last_structured_data: Dict[str, Any] = field(default_factory=dict)

    def add_turn(
        self,
        user_query: str,
        assistant_response: str,
        agents_invoked: List[str],
        citations: List[Dict[str, Any]],
        structured_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.history.append({
            "turn_index": len(self.history) + 1,
            "user_query": user_query,
            "assistant_response": assistant_response,
            "agents_invoked": agents_invoked,
            "citations": citations,
        })
        self.last_active_agents = list(agents_invoked)
        if structured_data is not None:
            self.last_structured_data = structured_data

    def get_recent_context_summary(self, max_turns: int = 3) -> str:
        if not self.history:
            return "No prior conversation turns."
        recent = self.history[-max_turns:]
        lines = []
        for turn in recent:
            agents_str = ", ".join(turn["agents_invoked"])
            lines.append(f"Turn {turn['turn_index']} (User): {turn['user_query']} -> Handled by [{agents_str}]")
        return "\n".join(lines)
