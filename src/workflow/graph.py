"""
LangGraph StateGraph Multi-Agent Orchestrator for the AI Finance Assistant.
Uses langgraph.graph.StateGraph when installed, and includes a drop-in NativeStateGraph
engine with identical node/edge/conditional-edge semantics so the workflow executes
seamlessly in any environment.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, List, Optional

from src.agents import (
    AgentResponse,
    FinanceQAAgent,
    GoalPlanningAgent,
    MarketAnalysisAgent,
    NewsSynthesizerAgent,
    PortfolioAnalysisAgent,
    TaxEducationAgent,
)
from src.core.config import AppConfig, LLMClient, logger
from src.data.market_provider import MarketDataProvider
from src.rag.retriever import FinanceRAGRetriever
from src.workflow.router import IntentRouter
from src.workflow.state import ConversationSession, WorkflowState

try:
    from langgraph.graph import END, StateGraph  # type: ignore
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    END = "__end__"


class NativeCompiledGraph:
    """Executes compiled StateGraph nodes, direct edges, and conditional edges deterministically."""

    def __init__(
        self,
        nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        edges: Dict[str, str],
        conditional_edges: Dict[str, Callable[[Dict[str, Any]], str]],
        entry_point: str,
    ) -> None:
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges
        self.entry_point = entry_point

    def invoke(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        state = dict(initial_state)
        current_node = self.entry_point
        steps = 0
        max_steps = 25

        while current_node != END and steps < max_steps:
            steps += 1
            node_fn = self.nodes.get(current_node)
            if node_fn is None:
                raise RuntimeError(f"Unknown workflow graph node: {current_node}")

            updates = node_fn(state)
            if updates:
                state.update(updates)

            if current_node in self.conditional_edges:
                next_node = self.conditional_edges[current_node](state)
            elif current_node in self.edges:
                next_node = self.edges[current_node]
            else:
                next_node = END
            current_node = next_node

        return state


class NativeStateGraph:
    """Drop-in compatible StateGraph implementation when langgraph package is not installed."""

    def __init__(self, state_schema: Any = None) -> None:
        self.state_schema = state_schema
        self.nodes: Dict[str, Callable[[Dict[str, Any]], Dict[str, Any]]] = {}
        self.edges: Dict[str, str] = {}
        self.conditional_edges: Dict[str, Callable[[Dict[str, Any]], str]] = {}
        self.entry_point: str = ""

    def add_node(self, name: str, action: Callable[[Dict[str, Any]], Dict[str, Any]]) -> None:
        self.nodes[name] = action

    def set_entry_point(self, name: str) -> None:
        self.entry_point = name

    def add_edge(self, start_key: str, end_key: str) -> None:
        self.edges[start_key] = end_key

    def add_conditional_edges(
        self,
        source: str,
        path_fn: Callable[[Dict[str, Any]], str],
        path_map: Optional[Dict[str, str]] = None,
    ) -> None:
        if path_map:
            self.conditional_edges[source] = lambda st: path_map.get(path_fn(st), END)
        else:
            self.conditional_edges[source] = path_fn

    def compile(self) -> NativeCompiledGraph:
        return NativeCompiledGraph(
            nodes=self.nodes,
            edges=self.edges,
            conditional_edges=self.conditional_edges,
            entry_point=self.entry_point,
        )


class FinanceAssistantWorkflow:
    """
    Orchestrates all 6 specialized agents via a StateGraph workflow:
    [context_loader] -> [intent_router] -> (conditional routing across 6 agent nodes) -> [response_synthesizer] -> END
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        rag_retriever: Optional[FinanceRAGRetriever] = None,
        market_provider: Optional[MarketDataProvider] = None,
    ) -> None:
        self.config = config or AppConfig.load()
        self.llm = LLMClient(self.config)
        self.rag = rag_retriever or FinanceRAGRetriever(
            chunk_size=self.config.rag_chunk_size,
            chunk_overlap=self.config.rag_chunk_overlap,
        )
        self.market = market_provider or MarketDataProvider(
            cache_ttl_seconds=self.config.market_cache_ttl_seconds
        )
        self.router = IntentRouter()
        self.sessions: Dict[str, ConversationSession] = {}

        shared_kwargs = {
            "config": self.config,
            "llm_client": self.llm,
            "rag_retriever": self.rag,
            "market_provider": self.market,
        }
        self.agents = {
            "finance_qa_agent": FinanceQAAgent(**shared_kwargs),
            "portfolio_analysis_agent": PortfolioAnalysisAgent(**shared_kwargs),
            "market_analysis_agent": MarketAnalysisAgent(**shared_kwargs),
            "goal_planning_agent": GoalPlanningAgent(**shared_kwargs),
            "news_synthesizer_agent": NewsSynthesizerAgent(**shared_kwargs),
            "tax_education_agent": TaxEducationAgent(**shared_kwargs),
        }
        self.compiled_graph = self._build_graph()

    def get_session(self, session_id: str = "default-session") -> ConversationSession:
        if session_id not in self.sessions:
            self.sessions[session_id] = ConversationSession(session_id=session_id)
        return self.sessions[session_id]

    def _context_loader_node(self, state: WorkflowState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        session_id = state.get("session_id", "default-session")
        session = self.get_session(session_id)
        ctx = dict(state.get("context") or {})

        # Propagate session user profile defaults into context if not explicitly overridden
        for k, v in session.user_profile.items():
            if k not in ctx:
                ctx[k] = v

        trace = list(state.get("execution_trace") or [])
        trace.append({
            "node": "context_loader",
            "status": "completed",
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            "history_turns": len(session.history),
        })
        return {
            "context": ctx,
            "user_profile": dict(session.user_profile),
            "conversation_history": list(session.history),
            "execution_trace": trace,
        }

    def _intent_router_node(self, state: WorkflowState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        session = self.get_session(state.get("session_id", "default-session"))
        decision = self.router.classify(
            query=state.get("user_query", ""),
            context=state.get("context"),
            last_active_agents=session.last_active_agents,
        )
        trace = list(state.get("execution_trace") or [])
        trace.append({
            "node": "intent_router",
            "status": "completed",
            "target_agents": decision.target_agents,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        })
        return {
            "classified_intents": decision.target_agents,
            "routing_rationale": decision.rationale,
            "routing_confidence": decision.confidence,
            "agent_outputs": {},
            "execution_trace": trace,
        }

    def _make_agent_node(self, agent_id: str) -> Callable[[WorkflowState], Dict[str, Any]]:
        def _node_fn(state: WorkflowState) -> Dict[str, Any]:
            t0 = time.perf_counter()
            agent = self.agents[agent_id]
            query = state.get("user_query", "")
            ctx = dict(state.get("context") or {})

            # Share upstream agent outputs in context for inter-agent collaboration
            existing_outputs = dict(state.get("agent_outputs") or {})
            if existing_outputs:
                ctx["upstream_agent_outputs"] = existing_outputs

            resp: AgentResponse = agent.run(query=query, context=ctx)
            existing_outputs[agent_id] = resp.to_dict()

            trace = list(state.get("execution_trace") or [])
            trace.append({
                "node": agent_id,
                "agent_name": agent.agent_name,
                "status": "completed" if not resp.error else "fallback",
                "confidence": resp.confidence_score,
                "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
            })
            return {
                "agent_outputs": existing_outputs,
                "execution_trace": trace,
            }
        return _node_fn

    def _next_step_selector(self, state: WorkflowState) -> str:
        """Determines the next unexecuted agent in classified_intents, or routes to synthesizer."""
        intents = state.get("classified_intents") or ["finance_qa_agent"]
        completed = set((state.get("agent_outputs") or {}).keys())
        for agent_id in intents:
            if agent_id in self.agents and agent_id not in completed:
                return agent_id
        return "response_synthesizer"

    def _response_synthesizer_node(self, state: WorkflowState) -> Dict[str, Any]:
        t0 = time.perf_counter()
        outputs = state.get("agent_outputs") or {}
        intents = state.get("classified_intents") or []
        session = self.get_session(state.get("session_id", "default-session"))

        combined_sections: List[str] = []
        all_citations: List[Dict[str, Any]] = []
        seen_urls = set()
        follow_ups: List[str] = []

        if len(outputs) > 1:
            agent_names = [outputs[a]["agent_name"] for a in intents if a in outputs]
            combined_sections.append(
                f"🔄 **Multi-Agent Collaborative Synthesis:** Coordinated insights across **{', '.join(agent_names)}**."
            )

        for agent_id in intents:
            out = outputs.get(agent_id)
            if not out:
                continue
            combined_sections.append(out["response_text"])
            for cit in out.get("citations", []):
                url = cit.get("source_url", "")
                title = cit.get("title", "")
                key = f"{title}|{url}"
                if key not in seen_urls:
                    seen_urls.add(key)
                    all_citations.append(cit)
            for sug in out.get("follow_up_suggestions", []):
                if sug not in follow_ups:
                    follow_ups.append(sug)

        final_text = "\n\n---\n\n".join(combined_sections) if combined_sections else "No response generated."

        # Update persistent conversation session
        session.add_turn(
            user_query=state.get("user_query", ""),
            assistant_response=final_text,
            agents_invoked=intents,
            citations=all_citations,
            structured_data={k: v.get("structured_data", {}) for k, v in outputs.items()},
        )

        trace = list(state.get("execution_trace") or [])
        trace.append({
            "node": "response_synthesizer",
            "status": "completed",
            "agents_merged": len(outputs),
            "citations_count": len(all_citations),
            "latency_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        })

        return {
            "final_response": final_text,
            "citations": all_citations,
            "follow_up_suggestions": follow_ups[:4],
            "conversation_history": list(session.history),
            "execution_trace": trace,
        }

    def _build_graph(self) -> Any:
        graph_cls = StateGraph if LANGGRAPH_AVAILABLE else NativeStateGraph
        builder = graph_cls(WorkflowState)

        builder.add_node("context_loader", self._context_loader_node)
        builder.add_node("intent_router", self._intent_router_node)
        for agent_id in self.agents:
            builder.add_node(agent_id, self._make_agent_node(agent_id))
        builder.add_node("response_synthesizer", self._response_synthesizer_node)

        builder.set_entry_point("context_loader")
        builder.add_edge("context_loader", "intent_router")

        path_map = {agent_id: agent_id for agent_id in self.agents}
        path_map["response_synthesizer"] = "response_synthesizer"

        builder.add_conditional_edges("intent_router", self._next_step_selector, path_map)
        for agent_id in self.agents:
            builder.add_conditional_edges(agent_id, self._next_step_selector, path_map)

        builder.add_edge("response_synthesizer", END)
        return builder.compile()

    def process_query(
        self,
        query: str,
        session_id: str = "default-session",
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Executes the full StateGraph workflow for a user query and returns the rich state."""
        t_start = time.perf_counter()
        initial_state: WorkflowState = {
            "session_id": session_id,
            "user_query": query,
            "context": context or {},
            "execution_trace": [],
        }
        result_state = self.compiled_graph.invoke(initial_state)
        result_state["total_latency_ms"] = round((time.perf_counter() - t_start) * 1000.0, 2)
        return result_state
