"""
Portfolio Analysis Agent: Evaluates user portfolios for asset allocation, sector concentration,
expense ratio fee drag, Sharpe/Sortino risk-adjusted metrics, and target rebalancing trades.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from src.agents.base_agent import AgentResponse, BaseAgent
from src.utils.financial_math import analyze_portfolio


class PortfolioAnalysisAgent(BaseAgent):
    """Specialized agent for quantitative portfolio diagnostics and rebalancing guidance."""

    SYSTEM_PROMPT = (
        "You are the Portfolio Analysis Agent. Analyze the user's holdings using Modern Portfolio Theory "
        "(Sharpe ratio, Sortino ratio, Beta, volatility, HHI diversification score, and expense ratio drag) "
        "and provide objective, educational rebalancing insights."
    )

    SAMPLE_PORTFOLIOS_PATH = (
        Path(__file__).resolve().parent.parent / "data" / "portfolios" / "sample_portfolios.json"
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_id="portfolio_analysis_agent",
            agent_name="Portfolio Analysis & MPT Specialist Agent",
            description="Computes portfolio allocation, risk/return ratios, concentration scores, fee drag, and rebalancing trades.",
            **kwargs,
        )
        self.sample_portfolios: Dict[str, Any] = {}
        if self.SAMPLE_PORTFOLIOS_PATH.exists():
            self.sample_portfolios = json.loads(
                self.SAMPLE_PORTFOLIOS_PATH.read_text(encoding="utf-8")
            )

    def _resolve_holdings_and_profile(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Selects user-supplied holdings from context or matches sample portfolio from query."""
        if context.get("holdings"):
            return {
                "name": context.get("portfolio_name", "Custom User Portfolio"),
                "risk_tolerance": context.get("risk_tolerance", "Moderate"),
                "holdings": context["holdings"],
            }

        q_low = query.lower()
        if "tech" in q_low or "aggressive" in q_low or "nvda" in q_low:
            key = "tech_concentrated_growth"
        elif "conservative" in q_low or "income" in q_low or "retire" in q_low:
            key = "conservative_income"
        else:
            key = context.get("portfolio_key", "beginner_balanced")

        selected = self.sample_portfolios.get(key) or next(iter(self.sample_portfolios.values()))
        return {
            "name": selected["name"],
            "risk_tolerance": context.get("risk_tolerance", selected.get("risk_tolerance", "Moderate")),
            "holdings": selected["holdings"],
        }

    def _execute(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        resolved = self._resolve_holdings_and_profile(query, context)
        holdings = resolved["holdings"]
        risk_tolerance = resolved["risk_tolerance"]
        portfolio_name = resolved["name"]

        # Refresh live/cached prices from MarketDataProvider if not explicitly overridden
        enriched_input: List[Dict[str, Any]] = []
        for h in holdings:
            item = dict(h)
            if "current_price" not in item or context.get("refresh_prices", False):
                quote = self.market.get_quote(item["ticker"])
                item["current_price"] = quote.price
            enriched_input.append(item)

        analysis = analyze_portfolio(
            holdings=enriched_input,
            risk_tolerance=risk_tolerance,
            risk_free_rate=self.config.risk_free_rate,
        )
        m = analysis["metrics"]

        rag_bundle = self.rag.build_context_prompt(
            query="portfolio diversification asset allocation rebalancing expense ratio sharpe ratio",
            top_k=3,
            category="portfolio_management",
        )

        # Build clean markdown scorecard table
        alloc_str = ", ".join(f"**{k}**: {v:.1f}%" for k, v in analysis["asset_allocation"].items())
        metrics_md = (
            f"#### Quantitative Scorecard — {portfolio_name}\n\n"
            "| Metric | Value | Benchmark / Interpretation |\n"
            "| :--- | ---: | :--- |\n"
            f"| **Total Market Value** | ${analysis['total_value']:,.2f} | Cost Basis: ${analysis['total_cost_basis']:,.2f} ({analysis['total_unrealized_gain_pct']:+.2f}%) |\n"
            f"| **Diversification Score** | {m['diversification_score']}/100 | HHI Concentration: {m['hhi_concentration']:.3f} |\n"
            f"| **Expected Annual Return** | {m['expected_annual_return_pct']:.2f}% | Annualized Volatility: {m['annualized_volatility_pct']:.2f}% |\n"
            f"| **Sharpe Ratio / Sortino Ratio** | {m['sharpe_ratio']:.2f} / {m['sortino_ratio']:.2f} | Risk-Free Rate Baseline: {self.config.risk_free_rate*100:.2f}% |\n"
            f"| **Portfolio Beta** | {m['portfolio_beta']:.2f} | S&P 500 Benchmark = 1.00 ({m['assessed_risk_level']} Risk) |\n"
            f"| **Weighted Expense Ratio** | {m['weighted_expense_ratio_pct']:.3f}% | ${m['annual_fee_dollars']:,.2f}/yr (10-Yr Fee Drag: ${m['fee_drag_10y_dollars']:,.2f}) |\n"
            f"| **Dividend Yield** | {m['weighted_dividend_yield_pct']:.2f}% | Est. Annual Income: ${m['estimated_annual_dividend_income']:,.2f}/yr |\n\n"
            f"**Current Asset Class Breakdown:** {alloc_str}"
        )

        recommendations: List[str] = []
        if m["diversification_score"] < 65:
            recommendations.append(
                f"**Reduce Concentration Risk:** Your Diversification Score is **{m['diversification_score']}/100**. "
                "Consider trimming oversized single-stock positions (<10% per company) and reallocating into broad index ETFs like `VTI` and `VXUS`."
            )
        else:
            recommendations.append(
                f"**Strong Diversification:** Your Diversification Score of **{m['diversification_score']}/100** indicates healthy spread across asset classes."
            )

        if m["weighted_expense_ratio_pct"] > 0.15:
            recommendations.append(
                f"**Lower Expense Drag:** Your weighted expense ratio is **{m['weighted_expense_ratio_pct']:.3f}%**, which projects to **${m['fee_drag_30y_dollars']:,.2f}** in compounded 30-year fee drag. Replacing high-fee funds (>0.40%) with low-cost index ETFs (<0.05%) preserves capital."
            )
        else:
            recommendations.append(
                f"**Ultra-Low Cost Efficiency:** Your **{m['weighted_expense_ratio_pct']:.3f}%** weighted expense ratio keeps fee drag minimal (${m['annual_fee_dollars']:,.2f}/yr)."
            )

        for action in analysis["rebalancing_actions"][:3]:
            recommendations.append(
                f"**Rebalancing Trade ({action['asset_class']}):** {action['action']} **${action['amount_dollars']:,.2f}** "
                f"to move from {action['current_pct']:.1f}% toward your {risk_tolerance} target of {action['target_pct']:.1f}%."
            )

        context_payload = {
            "headline": f"Portfolio Diagnostic & Risk Review: {portfolio_name}",
            "summary": (
                f"Your portfolio is currently valued at **${analysis['total_value']:,.2f}** "
                f"(unrealized gain of **${analysis['total_unrealized_gain']:,.2f}** or **{analysis['total_unrealized_gain_pct']:+.2f}%**). "
                f"Its volatility profile (**{m['annualized_volatility_pct']:.2f}%**) classifies it as **{m['assessed_risk_level']}** "
                f"compared to your stated **{m['target_risk_tolerance']}** preference."
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
            structured_data=analysis,
            citations=rag_bundle["citations"],
            follow_up_suggestions=[
                "How would rebalancing to a Moderate profile change my volatility?",
                "Analyze the Aggressive Tech-Heavy Growth sample portfolio",
                "How can I harvest tax losses when rebalancing my portfolio?",
            ],
            confidence_score=0.95,
        )
