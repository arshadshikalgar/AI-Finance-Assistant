"""
Market Analysis Agent: Provides real-time/cached stock & ETF quotes, technical indicators
(SMA-50, SMA-200, RSI-14, MACD), valuation metrics, and macroeconomic sector intelligence.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List

from src.agents.base_agent import AgentResponse, BaseAgent
from src.data.market_provider import BASELINE_MARKET_CATALOG


class MarketAnalysisAgent(BaseAgent):
    """Specialized agent for live market quotes, technicals, and sector trend interpretation."""

    SYSTEM_PROMPT = (
        "You are the Market Analysis Agent. Deliver accurate stock/ETF quote metrics, "
        "technical indicators (50/200-day moving averages, RSI-14, MACD trend), and macro context "
        "while reminding beginners that short-term movements do not replace long-term diversification."
    )

    COMMON_WORDS_UPPER = {
        "ETF", "IRA", "HSA", "ROTH", "IRS", "SEC", "FED", "GDP", "CPI", "PCE",
        "THE", "AND", "FOR", "HOW", "WHAT", "ARE", "CAN", "YOU", "WITH", "FROM",
        "PE", "EPS", "RSI", "SMA", "MACD", "NAV", "YTD", "APR", "APY", "USD",
    }

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_id="market_analysis_agent",
            agent_name="Market Intelligence & Technicals Agent",
            description="Retrieves stock/ETF quotes, 30-min cached market data, technical indicators, and sector performance.",
            **kwargs,
        )

    def extract_tickers(self, query: str, context: Dict[str, Any]) -> List[str]:
        if context.get("tickers"):
            return [str(t).upper() for t in context["tickers"]]

        found: List[str] = []
        # Check explicit known catalog tickers mentioned case-insensitively
        q_upper_words = set(re.findall(r"\b[A-Za-z.]{2,5}\b", query.upper()))
        for known in BASELINE_MARKET_CATALOG.keys():
            if known in q_upper_words:
                found.append(known)

        # Also check name mappings
        name_map = {
            "apple": "AAPL", "microsoft": "MSFT", "nvidia": "NVDA", "alphabet": "GOOGL",
            "google": "GOOGL", "amazon": "AMZN", "berkshire": "BRK.B", "jpmorgan": "JPM",
            "s&p": "SPY", "sp500": "SPY", "nasdaq": "QQQ", "vanguard total": "VTI",
            "bond": "BND", "treasury": "SGOV",
        }
        q_low = query.lower()
        for keyword, tkr in name_map.items():
            if keyword in q_low and tkr not in found:
                found.append(tkr)

        # Check uppercase ticker patterns in original query
        explicit_caps = re.findall(r"\$?([A-Z]{2,5})\b", query)
        for cand in explicit_caps:
            if cand not in self.COMMON_WORDS_UPPER and cand not in found:
                found.append(cand)

        if not found:
            found = ["SPY", "QQQ", "VTI", "BND"]
        return found[:5]

    def _execute(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        tickers = self.extract_tickers(query, context)
        quotes = self.market.get_multiple_quotes(tickers)
        overview = self.market.get_market_overview()
        macro = overview["macro_indicators"]

        rag_bundle = self.rag.build_context_prompt(
            query=f"market valuation technical indicators SMA RSI VIX {query}",
            top_k=3,
            category="market_intelligence_and_macro",
        )

        rows = [
            "#### Live & Cached Security Quotes & Technicals\n",
            "| Ticker | Name | Price | Day Change | P/E Ratio | Div Yield | SMA-50 / SMA-200 | RSI-14 | Trend Signal | Freshness |",
            "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | :--- | :--- |",
        ]
        for tkr, q in quotes.items():
            pe_str = f"{q.pe_ratio:.1f}x" if q.pe_ratio is not None else "N/A (Bond)"
            rows.append(
                f"| **{q.ticker}** | {q.name} | ${q.price:,.2f} | {q.change:+.2f} ({q.change_pct:+.2f}%) | "
                f"{pe_str} | {q.dividend_yield*100:.2f}% | ${q.sma_50:,.2f} / ${q.sma_200:,.2f} | "
                f"{q.rsi_14:.1f} | {q.macd_signal.split(' (')[0]} | `{q.freshness}` |"
            )

        macro_md = (
            "\n\n#### Macroeconomic & Yield Curve Backdrop\n"
            f"- **Federal Funds Target Rate:** {macro['fed_funds_rate']} | **10Y Treasury Yield:** {macro['us_10y_treasury_yield']} | **2Y Treasury Yield:** {macro['us_2y_treasury_yield']}\n"
            f"- **Inflation (CPI / Core PCE YoY):** {macro['cpi_inflation_yoy']} / {macro['core_pce_yoy']} | **VIX Volatility Index:** {macro['vix_volatility_index']} ({macro['market_regime']})"
        )

        metrics_md = "\n".join(rows) + macro_md

        recommendations = [
            "**Trend & Momentum Context:** Prices trading above both their 50-day and 200-day Simple Moving Averages indicate an established uptrend, whereas RSI-14 readings above 70 signal short-term overbought conditions.",
            "**Valuation Discipline:** Compare a stock or ETF's P/E ratio against the broad S&P 500 (~27.4x) and international equities (`VXUS` ~15.6x) to avoid overpaying for growth.",
            "**Dollar-Cost Averaging:** Rather than timing daily technical swings, invest on a recurring monthly schedule.",
        ]

        context_payload = {
            "headline": f"Market Intelligence & Technical Snapshot ({', '.join(quotes.keys())})",
            "summary": (
                f"Retrieved real-time/cached market metrics for **{', '.join(quotes.keys())}** "
                f"alongside current Federal Reserve and Treasury yield curve indicators."
            ),
            "rag_results": rag_bundle["results"],
            "metrics_markdown": metrics_md,
            "recommendations": recommendations,
            "citations": rag_bundle["citations"],
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
                "quotes": {k: v.to_dict() for k, v in quotes.items()},
                "macro_indicators": macro,
                "sector_performance": overview["sector_performance"],
            },
            citations=rag_bundle["citations"],
            follow_up_suggestions=[
                "Compare VTI vs VXUS valuation and dividend yields",
                "What does an RSI above 70 or a Golden Cross mean?",
                "How do current interest rates affect bond ETFs like BND?",
            ],
            confidence_score=0.94,
        )
