"""
News Synthesizer Agent: Summarizes and contextualizes financial news, evaluates sentiment,
and helps beginners separate long-term macroeconomic signal from short-term media noise.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.agents.base_agent import AgentResponse, BaseAgent


class NewsSynthesizerAgent(BaseAgent):
    """Specialized agent for financial news synthesis, sentiment scoring, and macro education."""

    SYSTEM_PROMPT = (
        "You are the News Synthesizer Agent. Summarize current financial and macroeconomic news, "
        "evaluate market sentiment, and translate complex headlines into calm, long-term educational "
        "takeaways for beginner investors."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_id="news_synthesizer_agent",
            agent_name="Financial News & Sentiment Synthesizer Agent",
            description="Synthesizes market headlines, computes sentiment scores, and explains beginner impact.",
            **kwargs,
        )

    def _execute(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        ticker_filter: Optional[str] = context.get("ticker")
        news_items = self.market.get_financial_news(ticker=ticker_filter, limit=5)

        avg_sentiment = (
            sum(float(n.get("sentiment_score", 0.0)) for n in news_items) / len(news_items)
            if news_items
            else 0.0
        )
        if avg_sentiment >= 0.35:
            overall_mood = "Constructive / Bullish"
        elif avg_sentiment <= -0.20:
            overall_mood = "Cautious / Bearish"
        else:
            overall_mood = "Balanced / Neutral-Constructive"

        rag_bundle = self.rag.build_context_prompt(
            query="reading financial news critically sentiment federal reserve inflation noise",
            top_k=3,
            category="market_intelligence_and_macro",
        )

        news_blocks: List[str] = ["#### Curated Financial News & Beginner Impact Analysis\n"]
        news_citations: List[Dict[str, Any]] = list(rag_bundle["citations"])

        for idx, item in enumerate(news_items, start=1):
            t_badges = ", ".join(f"`{t}`" for t in item.get("tickers", []))
            news_blocks.append(
                f"{idx}. **[{item['headline']}]({item['source_url']})** — *{item['source']}* "
                f"| Sentiment: **{item['sentiment']}** ({item['sentiment_score']:+.2f}) | Tickers: {t_badges}\n"
                f"   - **Summary:** {item['summary']}\n"
                f"   - **What It Means for You:** {item['beginner_context']}"
            )
            news_citations.append({
                "rank": len(news_citations) + 1,
                "article_id": item["id"],
                "title": item["headline"],
                "category": item["category"],
                "source": item["source"],
                "source_url": item["source_url"],
                "score": 0.95,
                "markdown_link": f"[{item['source']}]({item['source_url']})",
            })

        recommendations = [
            "**Separate Signal from Noise:** Ask whether a headline alters 10-year corporate earnings or your personal retirement timeline before making any trade.",
            "**Avoid Recency Bias:** Sensational headlines often trigger FOMO at market tops or panic selling at bottoms; stick to your written Investment Policy Statement (IPS).",
            "**Rebalance on Rules, Not Headlines:** Let portfolio drift thresholds (+/- 5%), rather than breaking news, dictate when you adjust holdings.",
        ]

        context_payload = {
            "headline": f"Market News Synthesis & Sentiment Briefing ({overall_mood})",
            "summary": (
                f"Synthesized **{len(news_items)}** key macroeconomic and market developments. "
                f"Aggregate News Sentiment Score is **{avg_sentiment:+.2f}** (**{overall_mood}**)."
            ),
            "rag_results": rag_bundle["results"],
            "metrics_markdown": "\n\n".join(news_blocks),
            "recommendations": recommendations,
            "citations": news_citations[:6],
        }

        response_text = self.llm.generate(
            system_prompt=self.SYSTEM_PROMPT,
            user_prompt=query,
            context_data=context_payload,
        )

        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.agent_name,
            query=query,
            response_text=response_text,
            structured_data={
                "news_items": news_items,
                "aggregate_sentiment_score": round(avg_sentiment, 3),
                "overall_market_mood": overall_mood,
            },
            citations=news_citations[:6],
            follow_up_suggestions=[
                "How do Federal Reserve interest rate decisions impact my bond ETFs?",
                "How can I protect my portfolio from behavioral biases like FOMO?",
                "Show me current market quotes for VTI, VXUS, and BND",
            ],
            confidence_score=0.93,
        )
