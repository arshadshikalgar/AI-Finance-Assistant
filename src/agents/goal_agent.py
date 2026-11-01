"""
Goal Planning Agent: Assists users with financial goal setting, retirement / FIRE targets,
home down payments, inflation adjustments, and 1,000-trial Monte Carlo simulations.
"""
from __future__ import annotations

import re
from typing import Any, Dict

from src.agents.base_agent import AgentResponse, BaseAgent
from src.utils.financial_math import project_financial_goal


class GoalPlanningAgent(BaseAgent):
    """Specialized agent for financial goal projections and Monte Carlo wealth modeling."""

    SYSTEM_PROMPT = (
        "You are the Goal Planning Agent. Help users translate financial aspirations into "
        "concrete, inflation-adjusted savings plans using deterministic compounding and "
        "1,000-trial Monte Carlo simulations matched to their risk appetite."
    )

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(
            agent_id="goal_planning_agent",
            agent_name="Financial Goal & Monte Carlo Planner Agent",
            description="Projects compound wealth trajectories, required monthly savings, inflation adjustments, and Monte Carlo success probabilities.",
            **kwargs,
        )

    def _parse_goal_parameters(self, query: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extracts goal parameters from context dict or natural language query."""
        goal_name = context.get("goal_name", "Long-Term Financial Independence / Retirement")
        q_low = query.lower()
        if "house" in q_low or "home" in q_low or "down payment" in q_low:
            goal_name = "Home Down Payment Goal"
        elif "college" in q_low or "education" in q_low or "529" in q_low:
            goal_name = "Education / 529 Savings Goal"
        elif "emergency" in q_low:
            goal_name = "Emergency Reserve Goal"

        # Risk tolerance
        if context.get("risk_tolerance"):
            risk = str(context["risk_tolerance"]).capitalize()
        elif "conservative" in q_low or "safe" in q_low or "low risk" in q_low:
            risk = "Conservative"
        elif "aggressive" in q_low or "high growth" in q_low:
            risk = "Aggressive"
        else:
            risk = "Moderate"

        # Parse years if mentioned (e.g. "in 20 years" or "15-year")
        years = int(context.get("years", 0))
        if years <= 0:
            yr_match = re.search(r"(\d{1,2})\s*(?:-|\s)?\s*year", q_low)
            years = int(yr_match.group(1)) if yr_match else 20

        # Parse dollar amounts in query if context does not specify them
        target_amount = float(context.get("target_amount", 0.0))
        current_savings = float(context.get("current_savings", -1.0))
        monthly_contribution = float(context.get("monthly_contribution", -1.0))

        if target_amount <= 0 or current_savings < 0 or monthly_contribution < 0:
            # Find numbers like $1,000,000 or 500k or $1500
            raw_amounts = re.findall(r"\$?\b(\d{1,3}(?:,\d{3})+|\d+)(k|m|million|thousand)?\b", q_low)
            parsed_nums = []
            for num_str, suffix in raw_amounts:
                val = float(num_str.replace(",", ""))
                if suffix in ("k", "thousand"):
                    val *= 1_000.0
                elif suffix in ("m", "million"):
                    val *= 1_000_000.0
                if val >= 100.0:  # ignore small numbers like years/percentages
                    parsed_nums.append(val)

            parsed_nums.sort(reverse=True)
            if target_amount <= 0:
                target_amount = parsed_nums[0] if parsed_nums else 1_000_000.0
            if current_savings < 0:
                current_savings = parsed_nums[1] if len(parsed_nums) > 1 and parsed_nums[1] < target_amount else 25_000.0
            if monthly_contribution < 0:
                monthly_contribution = parsed_nums[2] if len(parsed_nums) > 2 and parsed_nums[2] < 20_000 else 1_200.0

        return {
            "goal_name": goal_name,
            "target_amount": target_amount,
            "current_savings": current_savings,
            "monthly_contribution": monthly_contribution,
            "years": max(1, min(60, years)),
            "risk_tolerance": risk,
        }

    def _execute(self, query: str, context: Dict[str, Any]) -> AgentResponse:
        params = self._parse_goal_parameters(query, context)
        projection = project_financial_goal(
            goal_name=params["goal_name"],
            target_amount=params["target_amount"],
            current_savings=params["current_savings"],
            monthly_contribution=params["monthly_contribution"],
            years=params["years"],
            risk_tolerance=params["risk_tolerance"],
            inflation_rate=self.config.default_inflation_rate,
            num_simulations=1000,
        )
        mc = projection["monte_carlo"]

        rag_bundle = self.rag.build_context_prompt(
            query="financial goals compound interest monte carlo safe withdrawal rate time horizon",
            top_k=3,
            category="personal_finance_and_goals",
        )

        alloc_str = ", ".join(
            f"**{k}**: {v*100:.0f}%" for k, v in projection["recommended_allocation"].items()
        )
        metrics_md = (
            f"#### Goal Projection & 1,000-Trial Monte Carlo Summary — {projection['goal_name']}\n\n"
            "| Parameter / Metric | Projected Value | Notes |\n"
            "| :--- | ---: | :--- |\n"
            f"| **Target Goal (Nominal / Inflation-Adj.)** | ${projection['target_amount_nominal']:,.2f} | ${projection['target_amount_inflation_adjusted']:,.2f} in future dollars ({projection['inflation_rate_pct']:.1f}% CPI) |\n"
            f"| **Starting Capital & Monthly Savings** | ${projection['current_savings']:,.2f} | +${projection['monthly_contribution']:,.2f}/month for {projection['years']} years |\n"
            f"| **Risk Profile & Expected Return** | {projection['risk_tolerance']} ({projection['expected_annual_return_pct']:.2f}%/yr) | Volatility: {projection['assumed_volatility_pct']:.2f}% (Doubles every ~{projection['rule_of_72_doubling_years']} yrs) |\n"
            f"| **Deterministic Projected Balance** | **${projection['projected_final_nominal']:,.2f}** | Real Purchasing Power: **${projection['projected_final_real']:,.2f}** |\n"
            f"| **Total Principal vs. Compounding Gain** | ${projection['total_contributions']:,.2f} Principal | **+${projection['total_investment_growth']:,.2f}** Compounded Growth |\n"
            f"| **Required Monthly for Nominal Target** | ${projection['required_monthly_for_nominal_target']:,.2f}/mo | Required for Inflation-Adj Target: ${projection['required_monthly_for_inflation_adj_target']:,.2f}/mo |\n"
            f"| **Monte Carlo Success Probability** | **{mc['success_probability_nominal_pct']:.1f}%** (Nominal) | **{mc['success_probability_inflation_adj_pct']:.1f}%** (Inflation-Adjusted) |\n"
            f"| **Monte Carlo Outcomes (10th / 50th / 90th)** | ${mc['p10_bear_case']:,.0f} (Bear) | **${mc['p50_median_case']:,.0f}** (Median) • ${mc['p90_bull_case']:,.0f} (Bull) |\n\n"
            f"**Recommended Target Allocation ({projection['risk_tolerance']}):** {alloc_str}"
        )

        recommendations = []
        if projection["monthly_gap_nominal"] > 0:
            recommendations.append(
                f"**Close the Monthly Savings Gap:** Increasing your monthly contribution by **${projection['monthly_gap_nominal']:,.2f}/mo** "
                f"(to **${projection['required_monthly_for_nominal_target']:,.2f}/mo**) puts you on track for your nominal target."
            )
        else:
            recommendations.append(
                f"**On Track for Nominal Goal:** Your current **${projection['monthly_contribution']:,.2f}/mo** contribution exceeds the required **${projection['required_monthly_for_nominal_target']:,.2f}/mo**."
            )

        recommendations.append(
            f"**Protect Against Inflation:** To preserve full purchasing power against {projection['inflation_rate_pct']:.1f}% inflation, target **${projection['required_monthly_for_inflation_adj_target']:,.2f}/mo** or increase contributions by 3% annually as your income grows."
        )
        recommendations.append(
            "**Tax-Shelter Your Compounding:** Contribute first up to your employer's 401(k) match, then max out your HSA and Roth IRA to eliminate tax drag over your horizon."
        )

        context_payload = {
            "headline": f"Financial Goal Roadmap: {projection['goal_name']} ({projection['years']}-Year Horizon)",
            "summary": (
                f"Starting with **${projection['current_savings']:,.2f}** and investing **${projection['monthly_contribution']:,.2f}/month** "
                f"in a **{projection['risk_tolerance']}** portfolio ({projection['expected_annual_return_pct']:.2f}% expected return) "
                f"projects to **${projection['projected_final_nominal']:,.2f}** in {projection['years']} years, with a "
                f"**{mc['success_probability_nominal_pct']:.1f}%** Monte Carlo probability of reaching **${projection['target_amount_nominal']:,.2f}**."
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
            structured_data=projection,
            citations=rag_bundle["citations"],
            follow_up_suggestions=[
                "What happens if I increase my monthly savings by $500?",
                "How does the 4% Safe Withdrawal Rule apply to my retirement target?",
                "Which accounts (401k, Roth IRA, Taxable) should hold this goal?",
            ],
            confidence_score=0.95,
        )
