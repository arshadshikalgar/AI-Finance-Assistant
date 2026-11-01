"""
Quantitative Financial Math & Portfolio Analytics Engine.
Implements Modern Portfolio Theory (MPT) metrics, concentration scoring (HHI),
fee drag projections, goal compound growth & Monte Carlo simulations, and tax calculators.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


# Asset class expected annual returns and volatilities based on long-term capital market assumptions
ASSET_CLASS_DEFAULTS: Dict[str, Dict[str, float]] = {
    "US Equity": {"expected_return": 0.088, "volatility": 0.160, "beta": 1.00},
    "International Equity": {"expected_return": 0.078, "volatility": 0.170, "beta": 0.88},
    "Fixed Income": {"expected_return": 0.046, "volatility": 0.058, "beta": 0.05},
    "Real Estate": {"expected_return": 0.075, "volatility": 0.180, "beta": 0.85},
    "Cash Equivalents": {"expected_return": 0.045, "volatility": 0.005, "beta": 0.00},
}

# Target asset class weights by investor risk profile
TARGET_ALLOCATIONS: Dict[str, Dict[str, float]] = {
    "Conservative": {
        "US Equity": 0.25,
        "International Equity": 0.10,
        "Fixed Income": 0.50,
        "Cash Equivalents": 0.15,
    },
    "Moderate": {
        "US Equity": 0.48,
        "International Equity": 0.17,
        "Fixed Income": 0.28,
        "Cash Equivalents": 0.07,
    },
    "Aggressive": {
        "US Equity": 0.65,
        "International Equity": 0.25,
        "Fixed Income": 0.08,
        "Cash Equivalents": 0.02,
    },
}


def analyze_portfolio(
    holdings: List[Dict[str, Any]],
    risk_tolerance: str = "Moderate",
    risk_free_rate: float = 0.0435,
) -> Dict[str, Any]:
    """
    Computes comprehensive portfolio analytics:
    - Total market value, total cost basis, unrealized gain/loss ($ and %)
    - Holding weights, asset class allocation, sector breakdown
    - Weighted expense ratio & 10-year/30-year compounded fee drag
    - Portfolio Beta, Expected Annual Return, Annualized Volatility (with correlation discount)
    - Sharpe Ratio, Sortino Ratio, 95% 1-Year Value-at-Risk (VaR)
    - Herfindahl-Hirschman Index (HHI) and 0-100 Diversification Score
    - Specific rebalancing trade recommendations to align with target risk profile
    """
    if not holdings:
        raise ValueError("Portfolio holdings list cannot be empty.")

    enriched_holdings: List[Dict[str, Any]] = []
    total_value = 0.0
    total_cost = 0.0

    for item in holdings:
        shares = float(item.get("shares", 0.0))
        if shares <= 0:
            continue
        price = float(item.get("current_price", item.get("price", 100.0)))
        avg_cost = float(item.get("avg_cost", price))
        market_val = shares * price
        cost_val = shares * avg_cost
        gain_dlr = market_val - cost_val
        gain_pct = ((market_val / cost_val) - 1.0) * 100.0 if cost_val > 0 else 0.0

        asset_cls = item.get("asset_class", "US Equity")
        defaults = ASSET_CLASS_DEFAULTS.get(asset_cls, ASSET_CLASS_DEFAULTS["US Equity"])

        enriched_holdings.append({
            "ticker": str(item.get("ticker", "UNKNOWN")).upper(),
            "name": item.get("name", item.get("ticker", "Security")),
            "shares": round(shares, 4),
            "avg_cost": round(avg_cost, 2),
            "current_price": round(price, 2),
            "market_value": round(market_val, 2),
            "cost_basis": round(cost_val, 2),
            "unrealized_gain": round(gain_dlr, 2),
            "unrealized_gain_pct": round(gain_pct, 2),
            "asset_class": asset_cls,
            "sector": item.get("sector", "Broad Market"),
            "expense_ratio": float(item.get("expense_ratio", 0.0005)),
            "dividend_yield": float(item.get("dividend_yield", 0.015)),
            "beta": float(item.get("beta", defaults["beta"])),
            "expected_return": float(item.get("expected_return", defaults["expected_return"])),
            "volatility": float(item.get("volatility", defaults["volatility"])),
        })
        total_value += market_val
        total_cost += cost_val

    if total_value <= 0:
        raise ValueError("Total portfolio market value must be greater than zero.")

    # Calculate weights and aggregate metrics
    asset_allocation: Dict[str, float] = {}
    sector_allocation: Dict[str, float] = {}
    weighted_expense_ratio = 0.0
    weighted_dividend_yield = 0.0
    portfolio_beta = 0.0
    expected_return = 0.0
    weighted_vol_sum = 0.0
    hhi_sum = 0.0

    for h in enriched_holdings:
        w = h["market_value"] / total_value
        h["weight"] = round(w, 4)
        h["weight_pct"] = round(w * 100.0, 2)
        ac = h["asset_class"]
        sec = h["sector"]
        asset_allocation[ac] = round(asset_allocation.get(ac, 0.0) + (w * 100.0), 2)
        sector_allocation[sec] = round(sector_allocation.get(sec, 0.0) + (w * 100.0), 2)

        weighted_expense_ratio += w * h["expense_ratio"]
        weighted_dividend_yield += w * h["dividend_yield"]
        portfolio_beta += w * h["beta"]
        expected_return += w * h["expected_return"]
        weighted_vol_sum += w * h["volatility"]
        hhi_sum += w * w

    # Account for cross-asset correlation diversification benefit
    num_asset_classes = len(asset_allocation)
    correlation_discount = max(0.72, 1.0 - (0.065 * (num_asset_classes - 1)) - (0.02 * (1.0 - hhi_sum)))
    portfolio_volatility = max(0.005, weighted_vol_sum * correlation_discount)

    # Sharpe & Sortino Ratios
    excess_return = expected_return - risk_free_rate
    sharpe_ratio = excess_return / portfolio_volatility if portfolio_volatility > 0 else 0.0
    downside_volatility = portfolio_volatility * 0.71  # semi-deviation approximation
    sortino_ratio = excess_return / downside_volatility if downside_volatility > 0 else 0.0

    # 95% 1-year Parametric Value at Risk (VaR)
    z_95 = 1.64485
    var_95_pct = max(0.0, (z_95 * portfolio_volatility) - expected_return)
    var_95_dollars = total_value * var_95_pct

    # Diversification Score (0 to 100) combining HHI, asset class breadth, and ETF vs single-stock mix
    broad_etf_tickers = {"VTI", "VOO", "SPY", "VXUS", "BND", "SGOV", "VTIP", "SCHD", "VNQ", "QQQ", "VT", "AGG"}
    etf_weight = sum(h["weight"] for h in enriched_holdings if h["ticker"] in broad_etf_tickers)
    base_hhi_score = (1.0 - hhi_sum) * 75.0
    asset_class_bonus = min(20.0, num_asset_classes * 5.5)
    etf_breadth_bonus = etf_weight * 22.0
    diversification_score = min(98.0, max(12.0, base_hhi_score + asset_class_bonus + etf_breadth_bonus))

    # Risk Assessment Label
    if portfolio_volatility < 0.095:
        assessed_risk_level = "Conservative"
    elif portfolio_volatility < 0.148:
        assessed_risk_level = "Moderate"
    else:
        assessed_risk_level = "Aggressive"

    # Fee Drag Calculation over 10 and 30 years
    annual_fee_dollars = total_value * weighted_expense_ratio
    gross_10y = total_value * ((1.0 + expected_return) ** 10)
    net_10y = total_value * ((1.0 + expected_return - weighted_expense_ratio) ** 10)
    fee_drag_10y = gross_10y - net_10y

    gross_30y = total_value * ((1.0 + expected_return) ** 30)
    net_30y = total_value * ((1.0 + expected_return - weighted_expense_ratio) ** 30)
    fee_drag_30y = gross_30y - net_30y

    # Rebalancing trade calculation against target profile
    target_key = risk_tolerance.capitalize() if risk_tolerance.capitalize() in TARGET_ALLOCATIONS else "Moderate"
    targets = TARGET_ALLOCATIONS[target_key]
    rebalancing_actions: List[Dict[str, Any]] = []
    all_classes = sorted(set(list(asset_allocation.keys()) + list(targets.keys())))
    for ac in all_classes:
        curr_pct = asset_allocation.get(ac, 0.0)
        targ_pct = targets.get(ac, 0.0) * 100.0
        diff_pct = targ_pct - curr_pct
        diff_dlr = (diff_pct / 100.0) * total_value
        if abs(diff_pct) >= 2.0:
            action = "BUY" if diff_dlr > 0 else "TRIM / SELL"
            rebalancing_actions.append({
                "asset_class": ac,
                "current_pct": round(curr_pct, 2),
                "target_pct": round(targ_pct, 2),
                "drift_pct": round(diff_pct, 2),
                "action": action,
                "amount_dollars": round(abs(diff_dlr), 2),
            })

    total_gain = total_value - total_cost
    total_gain_pct = ((total_value / total_cost) - 1.0) * 100.0 if total_cost > 0 else 0.0

    return {
        "total_value": round(total_value, 2),
        "total_cost_basis": round(total_cost, 2),
        "total_unrealized_gain": round(total_gain, 2),
        "total_unrealized_gain_pct": round(total_gain_pct, 2),
        "holdings": enriched_holdings,
        "asset_allocation": asset_allocation,
        "sector_allocation": sector_allocation,
        "metrics": {
            "weighted_expense_ratio_pct": round(weighted_expense_ratio * 100.0, 3),
            "annual_fee_dollars": round(annual_fee_dollars, 2),
            "fee_drag_10y_dollars": round(fee_drag_10y, 2),
            "fee_drag_30y_dollars": round(fee_drag_30y, 2),
            "weighted_dividend_yield_pct": round(weighted_dividend_yield * 100.0, 2),
            "estimated_annual_dividend_income": round(total_value * weighted_dividend_yield, 2),
            "portfolio_beta": round(portfolio_beta, 2),
            "expected_annual_return_pct": round(expected_return * 100.0, 2),
            "annualized_volatility_pct": round(portfolio_volatility * 100.0, 2),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "sortino_ratio": round(sortino_ratio, 2),
            "var_95_1y_pct": round(var_95_pct * 100.0, 2),
            "var_95_1y_dollars": round(var_95_dollars, 2),
            "hhi_concentration": round(hhi_sum, 4),
            "diversification_score": round(diversification_score, 1),
            "assessed_risk_level": assessed_risk_level,
            "target_risk_tolerance": target_key,
        },
        "rebalancing_actions": rebalancing_actions,
    }


def project_financial_goal(
    goal_name: str,
    target_amount: float,
    current_savings: float,
    monthly_contribution: float,
    years: int,
    risk_tolerance: str = "Moderate",
    inflation_rate: float = 0.028,
    num_simulations: int = 1000,
) -> Dict[str, Any]:
    """
    Projects deterministic and Monte Carlo stochastic wealth paths for a financial goal,
    calculating inflation-adjusted targets, required monthly savings, and success probability.
    """
    if years <= 0:
        raise ValueError("Time horizon (years) must be greater than zero.")
    if target_amount <= 0:
        raise ValueError("Target amount must be greater than zero.")
    if current_savings < 0 or monthly_contribution < 0:
        raise ValueError("Current savings and monthly contribution must be non-negative.")

    profile_params = {
        "Conservative": {"mean_return": 0.056, "volatility": 0.068},
        "Moderate": {"mean_return": 0.076, "volatility": 0.115},
        "Aggressive": {"mean_return": 0.092, "volatility": 0.158},
    }
    rk = risk_tolerance.capitalize() if risk_tolerance.capitalize() in profile_params else "Moderate"
    mean_ret = profile_params[rk]["mean_return"]
    vol = profile_params[rk]["volatility"]

    # Inflation-adjusted target in future nominal dollars
    inflation_adjusted_target = target_amount * ((1.0 + inflation_rate) ** years)

    # Deterministic year-by-year compounding (monthly compounding)
    monthly_rate = mean_ret / 12.0
    months = years * 12
    yearly_schedule: List[Dict[str, Any]] = []
    balance = float(current_savings)
    total_principal = float(current_savings)

    for yr in range(1, years + 1):
        for _ in range(12):
            balance = balance * (1.0 + monthly_rate) + monthly_contribution
            total_principal += monthly_contribution
        real_balance = balance / ((1.0 + inflation_rate) ** yr)
        yearly_schedule.append({
            "year": yr,
            "principal_contributed": round(total_principal, 2),
            "projected_nominal_balance": round(balance, 2),
            "projected_real_balance": round(real_balance, 2),
            "investment_growth": round(balance - total_principal, 2),
        })

    # Required monthly contribution to hit nominal target_amount & inflation_adjusted_target
    future_val_current = current_savings * ((1.0 + monthly_rate) ** months)
    annuity_factor = (((1.0 + monthly_rate) ** months) - 1.0) / monthly_rate if monthly_rate > 0 else float(months)

    shortfall_nominal = max(0.0, target_amount - future_val_current)
    required_monthly_nominal = shortfall_nominal / annuity_factor if annuity_factor > 0 else 0.0

    shortfall_real_adj = max(0.0, inflation_adjusted_target - future_val_current)
    required_monthly_inflation_adj = shortfall_real_adj / annuity_factor if annuity_factor > 0 else 0.0

    # Monte Carlo Simulation (1,000 trials)
    rng = np.random.default_rng(42)
    annual_Contribution = monthly_contribution * 12.0
    sim_paths = np.zeros((num_simulations, years + 1), dtype=np.float64)
    sim_paths[:, 0] = current_savings

    # Log-normal geometric Brownian motion step
    mu_drift = mean_ret - 0.5 * (vol ** 2)
    random_shocks = rng.normal(loc=0.0, scale=1.0, size=(num_simulations, years))

    for t in range(1, years + 1):
        annual_growth_factors = np.exp(mu_drift + vol * random_shocks[:, t - 1])
        sim_paths[:, t] = sim_paths[:, t - 1] * annual_growth_factors + annual_Contribution

    final_balances = sim_paths[:, -1]
    prob_nominal = float(np.mean(final_balances >= target_amount) * 100.0)
    prob_inflation_adj = float(np.mean(final_balances >= inflation_adjusted_target) * 100.0)

    percentiles_by_year: List[Dict[str, Any]] = []
    for t in range(0, years + 1):
        col = sim_paths[:, t]
        percentiles_by_year.append({
            "year": t,
            "p10": round(float(np.percentile(col, 10)), 2),
            "p25": round(float(np.percentile(col, 25)), 2),
            "p50_median": round(float(np.percentile(col, 50)), 2),
            "p75": round(float(np.percentile(col, 75)), 2),
            "p90": round(float(np.percentile(col, 90)), 2),
        })

    rule_of_72_years = round(72.0 / (mean_ret * 100.0), 1)

    return {
        "goal_name": goal_name,
        "target_amount_nominal": round(target_amount, 2),
        "target_amount_inflation_adjusted": round(inflation_adjusted_target, 2),
        "current_savings": round(current_savings, 2),
        "monthly_contribution": round(monthly_contribution, 2),
        "years": years,
        "risk_tolerance": rk,
        "expected_annual_return_pct": round(mean_ret * 100.0, 2),
        "assumed_volatility_pct": round(vol * 100.0, 2),
        "inflation_rate_pct": round(inflation_rate * 100.0, 2),
        "rule_of_72_doubling_years": rule_of_72_years,
        "projected_final_nominal": round(balance, 2),
        "projected_final_real": round(balance / ((1.0 + inflation_rate) ** years), 2),
        "total_contributions": round(total_principal, 2),
        "total_investment_growth": round(balance - total_principal, 2),
        "required_monthly_for_nominal_target": round(required_monthly_nominal, 2),
        "required_monthly_for_inflation_adj_target": round(required_monthly_inflation_adj, 2),
        "monthly_gap_nominal": round(max(0.0, required_monthly_nominal - monthly_contribution), 2),
        "monte_carlo": {
            "num_simulations": num_simulations,
            "success_probability_nominal_pct": round(prob_nominal, 1),
            "success_probability_inflation_adj_pct": round(prob_inflation_adj, 1),
            "p10_bear_case": percentiles_by_year[-1]["p10"],
            "p50_median_case": percentiles_by_year[-1]["p50_median"],
            "p90_bull_case": percentiles_by_year[-1]["p90"],
            "percentiles_by_year": percentiles_by_year,
        },
        "yearly_schedule": yearly_schedule,
        "recommended_allocation": TARGET_ALLOCATIONS[rk],
    }


def calculate_tax_scenarios(
    annual_income: float = 115000.0,
    filing_status: str = "single",
    annual_contribution: float = 7000.0,
    capital_gain_amount: float = 10000.0,
    holding_period_months: int = 18,
    unrealized_loss_harvest: float = 3000.0,
) -> Dict[str, Any]:
    """
    Calculates marginal vs. effective income tax rates, short-term vs. long-term capital gains tax,
    Traditional vs. Roth account tax advantage comparison, and Tax-Loss Harvesting savings.
    """
    if annual_income < 0:
        raise ValueError("Annual income cannot be negative.")

    is_married = "married" in filing_status.lower() or "joint" in filing_status.lower()
    standard_deduction = 30000.0 if is_married else 15000.0
    taxable_income = max(0.0, annual_income - standard_deduction)

    # 2025/2026 Federal Income Tax Brackets
    brackets = (
        [
            (23850.0, 0.10),
            (96950.0, 0.12),
            (206700.0, 0.22),
            (394600.0, 0.24),
            (501050.0, 0.32),
            (751600.0, 0.35),
            (float("inf"), 0.37),
        ]
        if is_married
        else [
            (11925.0, 0.10),
            (48475.0, 0.12),
            (103350.0, 0.22),
            (197300.0, 0.24),
            (250525.0, 0.32),
            (626350.0, 0.35),
            (float("inf"), 0.37),
        ]
    )

    total_tax = 0.0
    prev_limit = 0.0
    marginal_rate = 0.10
    for limit, rate in brackets:
        if taxable_income > prev_limit:
            taxed_slice = min(taxable_income, limit) - prev_limit
            total_tax += taxed_slice * rate
            marginal_rate = rate
            prev_limit = limit
        else:
            break

    effective_rate = (total_tax / annual_income) if annual_income > 0 else 0.0

    # Long-term capital gains rate
    ltcg_0_limit = 96700.0 if is_married else 48350.0
    ltcg_15_limit = 600050.0 if is_married else 533400.0
    if taxable_income <= ltcg_0_limit:
        ltcg_rate = 0.0
    elif taxable_income <= ltcg_15_limit:
        ltcg_rate = 0.15
    else:
        ltcg_rate = 0.20

    is_long_term = holding_period_months > 12
    applied_cg_rate = ltcg_rate if is_long_term else marginal_rate
    stcg_tax = capital_gain_amount * marginal_rate
    ltcg_tax = capital_gain_amount * ltcg_rate
    cg_tax_saved_by_holding_1yr = stcg_tax - ltcg_tax

    # Tax-Loss Harvesting benefit
    tlh_ordinary_offset = min(3000.0, unrealized_loss_harvest)
    tlh_tax_savings = tlh_ordinary_offset * marginal_rate

    # Traditional Pre-Tax 401(k)/IRA upfront tax savings
    pretax_upfront_tax_savings = annual_contribution * marginal_rate

    return {
        "annual_income": round(annual_income, 2),
        "filing_status": "Married Filing Jointly" if is_married else "Single",
        "standard_deduction": round(standard_deduction, 2),
        "taxable_income": round(taxable_income, 2),
        "marginal_tax_rate_pct": round(marginal_rate * 100.0, 1),
        "effective_tax_rate_pct": round(effective_rate * 100.0, 2),
        "estimated_federal_income_tax": round(total_tax, 2),
        "capital_gains_analysis": {
            "capital_gain_amount": round(capital_gain_amount, 2),
            "holding_period_months": holding_period_months,
            "classification": "Long-Term (>12 months)" if is_long_term else "Short-Term (<=12 months)",
            "applied_rate_pct": round(applied_cg_rate * 100.0, 1),
            "short_term_tax_dollars": round(stcg_tax, 2),
            "long_term_tax_dollars": round(ltcg_tax, 2),
            "tax_saved_by_long_term_holding": round(cg_tax_saved_by_holding_1yr, 2),
        },
        "account_comparison": {
            "annual_contribution": round(annual_contribution, 2),
            "traditional_upfront_tax_savings": round(pretax_upfront_tax_savings, 2),
            "roth_20yr_tax_free_value_at_8pct": round(annual_contribution * (((1.08 ** 20) - 1.0) / 0.08), 2),
            "recommendation": (
                "Prioritize Traditional 401(k) for current bracket reduction (24%+ marginal bracket) "
                "paired with Backdoor Roth IRA and HSA for tax diversification."
                if marginal_rate >= 0.24
                else "Prioritize Roth IRA and Roth 401(k) to lock in today's lower 10%-22% tax bracket "
                     "after capturing full employer 401(k) match."
            ),
        },
        "tax_loss_harvesting": {
            "harvested_loss_amount": round(unrealized_loss_harvest, 2),
            "ordinary_income_offset_allowed": round(tlh_ordinary_offset, 2),
            "carryforward_loss": round(max(0.0, unrealized_loss_harvest - 3000.0), 2),
            "estimated_tax_savings_dollars": round(tlh_tax_savings, 2),
        },
    }
