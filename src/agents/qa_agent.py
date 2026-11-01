"""
Finance Q&A Agent: Handles general financial education queries, concept explanations,
investing basics, ETFs vs. mutual funds, bond mechanics, and behavioral finance using RAG.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from src.agents.base_agent import AgentResponse, BaseAgent


class FinanceQAAgent(BaseAgent):
    """Specialized agent for financial literacy, concept definitions, and RAG-grounded Q&A."""

    SYSTEM_PROMPT = (
        "You are the Finance Q&A Agent, an empathetic and rigorous financial educator for beginners. "
        "Explain concepts clearly without jargon, use concrete numerical examples, cite verified sources "
        "from the retrieved knowledge base, and always distinguish education from fiduciary advice."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_id="finance_qa_agent",
            agent_name="Finance Q&A Educator Agent",
            description="Answers financial literacy questions using semantic RAG over 108 curated articles and 51 glossary terms.",
            **kwargs,
        )

    def _execute(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        category_filter: Optional[str] = context.get("category")
        rag_bundle = self.rag.build_context_prompt(
            query=query,
            top_k=self.config.rag_top_k,
            category=category_filter,
        )
        results = rag_bundle["results"]
        citations = rag_bundle["citations"]
        glossary_matches = rag_bundle["glossary_matches"]

        if results:
            top_hit = results[0].chunk
            headline = f"Financial Concept Guide: {top_hit.title}"
            summary = (
                f"To help answer your question (*\"{query}\"*), here is a clear breakdown grounded in "
                f"**{top_hit.source}** and our curated financial education library."
            )
        else:
            headline = "Financial Education Overview"
            summary = (
                f"Here is a beginner-friendly guide to **{query}** based on core evidence-based "
                "investing principles (diversification, cost minimization, and long-term compounding)."
            )

        glossary_md = ""
        if glossary_matches:
            rows = [f"- **{g['term']}**: {g['definition']}" for g in glossary_matches[:3]]
            glossary_md = "#### Quick Glossary Definitions\n" + "\n".join(rows)

        recommendations = [
            "Keep investment fees low by favoring broad-market index ETFs charging under 0.10% annually.",
            "Match your asset allocation to your time horizon—keep money needed within 3 years in liquid cash/T-Bills.",
            "Automate monthly contributions (Dollar-Cost Averaging) to remove emotion from market volatility.",
        ]

        context_payload = {
            "headline": headline,
            "summary": summary,
            "rag_results": results,
            "metrics_markdown": glossary_md,
            "recommendations": recommendations,
            "citations": citations,
        }

        response_text = self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=f"Query: {query}\n\nRetrieved Context:\n{rag_bundle['context_text']}",
            context_data=context_payload,
        )

        confidence = min(0.98, 0.75 + (results[0].score * 0.25)) if results else 0.72

        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            query=query,
            response_text=response_text,
            structured_data={
                "matched_articles": [r.to_dict() for r in results],
                "glossary_matches": glossary_matches,
                "knowledge_base_size": self.rag.total_articles,
            },
            citations=citations,
            follow_up_suggestions=[
                "How does the Bogleheads Three-Fund Portfolio work?",
                "What is the difference between an ETF and a Mutual Fund?",
                "How do bond prices react when interest rates change?",
            ],
            confidence_score=confidence,
        )
