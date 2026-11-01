"""
Tax Education Agent: Explains tax-advantaged accounts (Traditional vs. Roth 401k/IRA, HSA, 529),
marginal vs. effective tax rates, short-term vs. long-term capital gains, asset location,
and tax-loss harvesting (including the IRS Wash-Sale Rule).
"""
from __future__ import annotations

import re
from typing import Any, Dict

from src.agents.base_agent import AgentResponse, BaseAgent
from src.utils.financial_math import calculate_tax_scenarios


class TaxEducationAgent(BaseAgent):
    """Specialized agent for tax-efficient investing education and account type comparisons."""

    SYSTEM_PROMPT = (
        "You are the Tax Education Agent. Explain tax-advantaged accounts (401k, Roth IRA, HSA, 529), "
        "capital gains rules, asset location, and tax-loss harvesting clearly using IRS rules, "
        "while noting that users should consult a CPA for personalized tax filing advice."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_id="tax_education_agent",
            agent_name="Tax Education & Account Optimization Agent",
            description="Explains 401(k)/IRA/HSA/529 accounts, marginal vs. effective brackets, capital gains, and tax-loss harvesting.",
            **kwargs,
        )

    def _parse_tax_inputs(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        income = float(context.get("annual_income", 0.0))
        if income <= 0:
            inc_match = re.search(r"\$?(\d{2,3}(?:,\d{3})|\d{2,3}k)\b", query.lower())
            if inc_match:
                raw = inc_match.group(1).replace(",", "")
                income = float(raw[:-1]) * 1000.0 if raw.endswith("k") else float(raw)
            else:
                income = 115000.0

        filing_status = context.get(
            "filing_status",
            "married" if "married" in query.lower() or "joint" in query.lower() else "single",
        )
        return {
            "annual_income": income,
            "filing_status": filing_status,
            "annual_contribution": float(context.get("annual_contribution", 7000.0)),
            "capital_gain_amount": float(context.get("capital_gain_amount", 10000.0)),
            "holding_period_months": int(context.get("holding_period_months", 18)),
            "unrealized_loss_harvest": float(context.get("unrealized_loss_harvest", 3000.0)),
        }

    def _execute(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        tax_inputs = self._parse_tax_inputs(query, context)
        tax_calc = calculate_tax_scenarios(**tax_inputs)
        cg = tax_calc["capital_gains_analysis"]
        acc = tax_calc["account_comparison"]
        tlh = tax_calc["tax_loss_harvesting"]

        rag_bundle = self.rag.build_context_prompt(
            query=f"tax 401k roth ira hsa capital gains wash sale {query}",
            top_k=3,
            category="tax_and_retirement",
        )

        metrics_md = (
            f"#### Tax Efficiency & Account Comparison ({tax_calc['filing_status']}, ${tax_calc['annual_income']:,.0f} Gross Income)\n\n"
            "| Tax Metric / Scenario | Estimated Value | Educational Explanation |\n"
            "| :--- | ---: | :--- |\n"
            f"| **Taxable Income (After ${tax_calc['standard_deduction']:,.0f} Std Deduction)** | ${tax_calc['taxable_income']:,.2f} | Est. Federal Income Tax: ${tax_calc['estimated_federal_income_tax']:,.2f} |\n"
            f"| **Marginal vs. Effective Tax Rate** | **{tax_calc['marginal_tax_rate_pct']:.1f}%** Marginal | **{tax_calc['effective_tax_rate_pct']:.2f}%** Effective (Average) Rate |\n"
            f"| **Pre-Tax 401(k)/IRA Upfront Tax Savings** | **${acc['traditional_upfront_tax_savings']:,.2f}/yr** | On a ${acc['annual_contribution']:,.0f} pre-tax contribution |\n"
            f"| **Roth Account 20-Yr Tax-Free Balance** | **${acc['roth_20yr_tax_free_value_at_8pct']:,.2f}** | Contributing ${acc['annual_contribution']:,.0f}/yr at 8% (100% tax-free in retirement) |\n"
            f"| **Capital Gains Tax (${cg['capital_gain_amount']:,.0f} Gain)** | **${cg['long_term_tax_dollars']:,.2f}** (Long-Term >1yr) | vs. ${cg['short_term_tax_dollars']:,.2f} Short-Term (Saves **${cg['tax_saved_by_long_term_holding']:,.2f}**) |\n"
            f"| **Tax-Loss Harvesting Offset (${tlh['harvested_loss_amount']:,.0f} Loss)** | **${tlh['estimated_tax_savings_dollars']:,.2f}** Saved | Offsets realized gains + up to $3,000/yr ordinary income |\n\n"
            "#### Tax-Advantaged Account Priority Order ('The Tax Waterfall')\n"
            "1. **401(k) up to Employer Match:** Captures an instant 50%–100% risk-free return.\n"
            "2. **Health Savings Account (HSA):** Triple tax-advantaged (deductible going in, tax-free growth, tax-free medical withdrawals).\n"
            "3. **Roth IRA / Traditional IRA:** Tax-free growth and flexible investment choices.\n"
            "4. **Remaining 401(k) Space & Taxable Brokerage:** Hold broad index ETFs (`VTI`, `VXUS`) in taxable accounts for long-term capital gains rates and Foreign Tax Credits."
        )

        recommendations = [
            f"**Account Selection Strategy:** {acc['recommendation']}",
            f"**Hold >12 Months for Preferential Rates:** Holding appreciated investments in a taxable account longer than 1 year drops your tax rate from **{tax_calc['marginal_tax_rate_pct']:.1f}%** to **{cg['applied_rate_pct']:.1f}%**, saving **${cg['tax_saved_by_long_term_holding']:,.2f}** per $10,000 gain.",
            "**Observe the 30-Day IRS Wash-Sale Rule:** When harvesting losses in a taxable account, avoid repurchasing a 'substantially identical' security within 30 days before or after the sale across any of your accounts (including IRAs).",
        ]

        context_payload = {
            "headline": "Tax-Efficient Investing & Account Optimization Guide",
            "summary": (
                f"Based on an illustrative **${tax_calc['annual_income']:,.0f}** ({tax_calc['filing_status']}) profile, "
                f"your top marginal federal bracket is **{tax_calc['marginal_tax_rate_pct']:.1f}%**, while your true effective tax rate is only **{tax_calc['effective_tax_rate_pct']:.2f}%**."
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
            structured_data=tax_calc,
            citations=rag_bundle["citations"],
            follow_up_suggestions=[
                "Should I choose Roth 401(k) or Traditional 401(k)?",
                "How does the IRS Wash-Sale Rule work in Tax-Loss Harvesting?",
                "What is Asset Location across taxable vs IRA accounts?",
            ],
            confidence_score=0.95,
        )
