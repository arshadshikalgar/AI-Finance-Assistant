"""
Intelligent Multi-Intent Query Router for the AI Finance Assistant.
Classifies user queries into one or more of the 6 specialized agents and supports
context-aware follow-up resolution across multi-turn conversations.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class RouteDecision:
    """Represents the router's agent selection, confidence score, and rationale."""
    target_agents: List[str]
    confidence: float
    rationale: str
    is_multi_agent: bool


class IntentRouter:
    """Routes user queries to the optimal agent(s) using weighted keyword, ticker, and context signals."""

    AGENT_PATTERNS: Dict[str, List[str]] = {
        "portfolio_analysis_agent": [
            r"\bportfolio\b", r"\ballocation\b", r"\brebalanc\w*\b", r"\bholding\w*\b",
            r"\bsharpe\b", r"\bsortino\b", r"\bhhi\b", r"\bconcentration\b",
            r"\bmy stocks\b", r"\bmy funds\b", r"\bmy etfs\b", r"\banalyze my\b",
            r"\bdiversification score\b", r"\bfee drag\b", r"\bthree-fund\b",
        ],
        "market_analysis_agent": [
            r"\bquote\w*\b", r"\bprice\w*\b", r"\bticker\w*\b", r"\bsma\b", r"\brsi\b",
            r"\bmacd\b", r"\btechnical\w*\b", r"\btrading at\b", r"\bmarket overview\b",
            r"\bsector performance\b", r"\bvix\b", r"\byield curve\b",
            r"\b(?:spy|voo|vti|qqq|vxus|bnd|sgov|schd|vtip|vnq|aapl|msft|nvda|googl|amzn|jpm|jnj|arkk)\b",
        ],
        "goal_planning_agent": [
            r"\bgoal\w*\b", r"\bretire\w*\b", r"\bmonte carlo\b", r"\bsave for\b",
            r"\bsaving for\b", r"\bdown payment\b", r"\btarget amount\b", r"\bprojection\w*\b",
            r"\bhow much (?:do i need|should i save|to save|monthly)\b", r"\bfire\b",
            r"\b\d+\s*years?\b.*\b(?:\$\d+|\d+k|\d+m|million)\b",
        ],
        "news_synthesizer_agent": [
            r"\bnews\b", r"\bheadline\w*\b", r"\bsentiment\b", r"\bwhat(?:'s| is) happening\b",
            r"\bmarket update\b", r"\bcurrent events\b", r"\bfed(?:eral reserve)? announcement\b",
            r"\bearnings season\b", r"\bmedia\b",
        ],
        "tax_education_agent": [
            r"\btax\w*\b", r"\b401\(?k\)?\b", r"\broth\b", r"\bira\b", r"\bhsa\b",
            r"\b529\b", r"\bcapital gain\w*\b", r"\bwash[- ]sale\b", r"\bharvest\w*\b",
            r"\bdeduction\w*\b", r"\bbracket\w*\b", r"\brmd\w*\b", r"\basset location\b",
        ],
        "finance_qa_agent": [
            r"\bwhat is\b", r"\bwhat are\b", r"\bexplain\b", r"\bhow does\b", r"\bhow do\b",
            r"\bdifference between\b", r"\bdefine\b", r"\bmeaning of\b", r"\bbeginner\b",
            r"\bcompound interest\b", r"\binflation\b", r"\bdividend\b", r"\bbond\b",
            r"\betf\b", r"\bmutual fund\b", r"\bfiduciary\b", r"\bemergency fund\b",
        ],
    }

    FOLLOW_UP_PRONOUNS = {
        "it", "that", "those", "this", "them", "instead", "what if", "change it", "same"
    }

    def classify(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        last_active_agents: Optional[List[str]] = None,
    ) -> RouteDecision:
        ctx = context or {}
        if ctx.get("force_agent"):
            forced = ctx["force_agent"]
            agents = forced if isinstance(forced, list) else [forced]
            return RouteDecision(
                target_agents=agents,
                confidence=1.0,
                rationale=f"Explicitly routed to {', '.join(agents)} via context override.",
                is_multi_agent=len(agents) > 1,
            )

        q_low = (query or "").lower().strip()
        if not q_low:
            return RouteDecision(
                target_agents=["finance_qa_agent"],
                confidence=0.60,
                rationale="Empty query defaulted to Finance Q&A Educator Agent.",
                is_multi_agent=False,
            )

        scores: Dict[str, float] = {agent_id: 0.0 for agent_id in self.AGENT_PATTERNS}

        for agent_id, patterns in self.AGENT_PATTERNS.items():
            for pat in patterns:
                matches = re.findall(pat, q_low)
                if matches:
                    scores[agent_id] += 1.0 + (0.25 * (len(matches) - 1))

        # Contextual boosts if holdings or goal parameters are attached
        if ctx.get("holdings"):
            scores["portfolio_analysis_agent"] += 2.5
        if ctx.get("target_amount") or ctx.get("monthly_contribution"):
            scores["goal_planning_agent"] += 2.5

        # Follow-up memory boost if user asks a short follow-up like "What if I make it Aggressive?"
        if last_active_agents and len(q_low.split()) <= 10:
            if any(p in q_low for p in self.FOLLOW_UP_PRONOUNS) or max(scores.values()) == 0:
                for prev_agent in last_active_agents:
                    if prev_agent in scores:
                        scores[prev_agent] += 1.2

        sorted_agents = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_agent, top_score = sorted_agents[0]

        if top_score <= 0.0:
            return RouteDecision(
                target_agents=["finance_qa_agent"],
                confidence=0.78,
                rationale="General financial inquiry routed to Finance Q&A Educator Agent with RAG.",
                is_multi_agent=False,
            )

        # Select primary agent + any secondary agent with strong co-intent (score >= 1.0 and >= 60% of top)
        selected = [top_agent]
        for sec_agent, sec_score in sorted_agents[1:3]:
            if sec_score >= 1.25 and sec_score >= (top_score * 0.55):
                selected.append(sec_agent)

        confidence = min(0.98, 0.78 + (0.05 * top_score))
        rationale = (
            f"Multi-agent orchestration across {', '.join(selected)} based on compound query intent."
            if len(selected) > 1
            else f"Routed to {top_agent} (intent score: {top_score:.2f})."
        )
        return RouteDecision(
            target_agents=selected,
            confidence=round(confidence, 3),
            rationale=rationale,
            is_multi_agent=len(selected) > 1,
        )
