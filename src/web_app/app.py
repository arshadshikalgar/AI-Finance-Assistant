"""
Streamlit Multi-Tab Web Application for the AI Finance Assistant.
Features 6 interactive tabs:
1. Conversational AI Assistant (with multi-turn session state, agent badges, citations, & quick prompts)
2. Portfolio Analysis & MPT Dashboard (allocation pie/bar charts, Sharpe/Sortino, HHI score, rebalancing)
3. Real-Time Market Intelligence (live/cached quotes, 30-day trend lines, SMA/RSI technicals, sector heatmap)
4. Financial Goal & Monte Carlo Planner (1,000-trial fan chart, inflation-adjusted targets, savings gap)
5. Tax Education & Knowledge Base Explorer (108 curated articles, 51 glossary terms, tax calculator)
6. Multi-Agent Telemetry & MCP Server Inspector (LangGraph execution trace & live MCP tool tester)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.mcp_server.server import FinanceMCPServer
from src.utils.financial_math import (
    analyze_portfolio,
    calculate_tax_scenarios,
    project_financial_goal,
)
from src.workflow.graph import FinanceAssistantWorkflow


def get_app_metadata() -> Dict[str, Any]:
    """Returns structured metadata about the web app tabs and capabilities for testing & inspection."""
    return {
        "title": "AI Finance Assistant — Multi-Agent Financial Literacy & Wealth Platform",
        "tabs": [
            "💬 AI Chat Assistant",
            "📊 Portfolio Analytics",
            "📈 Market Intelligence",
            "🎯 Goal & Monte Carlo Planner",
            "🏛️ Tax & Knowledge Base",
            "⚙️ LangGraph & MCP Telemetry",
        ],
        "agents_count": 6,
    }


def render_streamlit_app() -> None:
    """Renders the full Streamlit multi-tab interface when invoked via `streamlit run src/web_app/app.py`."""
    import streamlit as st  # type: ignore

    st.set_page_config(
        page_title="AI Finance Assistant | Multi-Agent Platform",
        page_icon="💹",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    if "workflow" not in st.session_state:
        st.session_state.workflow = FinanceAssistantWorkflow()
    if "mcp_server" not in st.session_state:
        st.session_state.mcp_server = FinanceMCPServer(st.session_state.workflow)
    if "session_id" not in st.session_state:
        st.session_state.session_id = "streamlit-user-session"

    workflow: FinanceAssistantWorkflow = st.session_state.workflow
    session = workflow.get_session(st.session_state.session_id)

    # Sidebar: User Profile & Global Settings
    with st.sidebar:
        st.title("💹 AI Finance Assistant")
        st.caption("Multi-Agent LangGraph + FAISS RAG + MCP")
        st.divider()
        st.subheader("👤 Investor Profile")
        risk_tol = st.selectbox(
            "Risk Tolerance",
            ["Conservative", "Moderate", "Aggressive"],
            index=["Conservative", "Moderate", "Aggressive"].index(
                session.user_profile.get("risk_tolerance", "Moderate")
            ),
        )
        annual_inc = st.number_input(
            "Annual Gross Income ($)",
            min_value=20000.0,
            max_value=2000000.0,
            value=float(session.user_profile.get("annual_income", 115000.0)),
            step=5000.0,
        )
        filing_stat = st.selectbox("Tax Filing Status", ["single", "married"], index=0)
        session.user_profile.update({
            "risk_tolerance": risk_tol,
            "annual_income": annual_inc,
            "filing_status": filing_stat,
        })
        st.divider()
        st.markdown(
            f"**Knowledge Base:** `{workflow.rag.total_articles}` Articles (`{workflow.rag.total_chunks}` Chunks)\n\n"
            f"**Glossary:** `{len(workflow.rag.glossary)}` Financial Terms\n\n"
            "**Active Agents:** `6 Specialized Agents`"
        )
        st.info(workflow.config.disclaimer_text)

    tabs = st.tabs(get_app_metadata()["tabs"])

    # TAB 1: Conversational AI Assistant
    with tabs[0]:
        st.subheader("💬 Multi-Agent Financial Literacy & Advisory Chat")
        col_q1, col_q2, col_q3 = st.columns(3)
        prompt_override = None
        if col_q1.button("📘 Explain the Bogleheads 3-Fund Portfolio"):
            prompt_override = "How does the Bogleheads Three-Fund Portfolio work and why are low fees important?"
        if col_q2.button("📊 Analyze my portfolio & tax rebalancing"):
            prompt_override = "Analyze my beginner balanced portfolio and explain how to minimize capital gains tax when rebalancing."
        if col_q3.button("🎯 Plan $1M retirement in 20 years"):
            prompt_override = "Help me plan for a $1,000,000 retirement goal in 20 years with $25,000 saved and $1,200/month."

        for turn in session.history:
            with st.chat_message("user"):
                st.markdown(turn["user_query"])
            with st.chat_message("assistant"):
                badges = " ".join(f"`{a}`" for a in turn["agents_invoked"])
                st.caption(f"Handled by: {badges}")
                st.markdown(turn["assistant_response"])

        user_input = st.chat_input("Ask about investing concepts, portfolios, tickers (VTI, NVDA), goals, or taxes...")
        active_query = prompt_override or user_input
        if active_query:
            with st.chat_message("user"):
                st.markdown(active_query)
            with st.chat_message("assistant"):
                result = workflow.process_query(active_query, session_id=st.session_state.session_id)
                badges = " ".join(f"`{a}`" for a in result["classified_intents"])
                st.caption(
                    f"Routed to: {badges} | Confidence: {result['routing_confidence']*100:.0f}% | "
                    f"Latency: {result['total_latency_ms']:.1f} ms"
                )
                st.markdown(result["final_response"])

    # TAB 2: Portfolio Analysis Dashboard
    with tabs[1]:
        st.subheader("📊 Portfolio Diagnostics & Modern Portfolio Theory Dashboard")
        p_agent = workflow.agents["portfolio_analysis_agent"]
        preset_key = st.selectbox(
            "Select Portfolio Preset",
            list(p_agent.sample_portfolios.keys()),
            format_func=lambda k: p_agent.sample_portfolios[k]["name"],
        )
        preset = p_agent.sample_portfolios[preset_key]
        analysis = analyze_portfolio(preset["holdings"], risk_tolerance=risk_tol)
        m = analysis["metrics"]

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total Value", f"${analysis['total_value']:,.2f}", f"{analysis['total_unrealized_gain_pct']:+.2f}%")
        k2.metric("Diversification Score", f"{m['diversification_score']}/100", f"HHI: {m['hhi_concentration']:.3f}")
        k3.metric("Sharpe / Sortino", f"{m['sharpe_ratio']:.2f} / {m['sortino_ratio']:.2f}", f"Beta: {m['portfolio_beta']:.2f}")
        k4.metric("Expense Ratio", f"{m['weighted_expense_ratio_pct']:.3f}%", f"${m['annual_fee_dollars']:,.2f}/yr fee")

        st.dataframe(pd.DataFrame(analysis["holdings"]), use_container_width=True)
        if analysis["rebalancing_actions"]:
            st.markdown("#### Recommended Rebalancing Trades")
            st.dataframe(pd.DataFrame(analysis["rebalancing_actions"]), use_container_width=True)

    # TAB 3: Market Intelligence
    with tabs[2]:
        st.subheader("📈 Real-Time & Cached Market Intelligence")
        overview = workflow.market.get_market_overview()
        tkr_input = st.text_input("Lookup Ticker Symbol", value="VTI")
        if tkr_input:
            quote = workflow.market.get_quote(tkr_input)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric(f"{quote.ticker} ({quote.name})", f"${quote.price:,.2f}", f"{quote.change:+.2f} ({quote.change_pct:+.2f}%)")
            c2.metric("SMA-50 / SMA-200", f"${quote.sma_50:,.2f}", f"SMA-200: ${quote.sma_200:,.2f}")
            c3.metric("RSI-14 Momentum", f"{quote.rsi_14:.1f}", quote.macd_signal.split(" (")[0])
            c4.metric("Dividend Yield / Expense", f"{quote.dividend_yield*100:.2f}%", f"Freshness: {quote.freshness}")
            st.line_chart(pd.DataFrame({"30-Day Price ($)": quote.price_history_30d}))
        st.markdown("#### GICS Sector Performance")
        st.dataframe(pd.DataFrame(overview["sector_performance"]), use_container_width=True)

    # TAB 4: Goal & Monte Carlo Planner
    with tabs[3]:
        st.subheader("🎯 Financial Goal & 1,000-Trial Monte Carlo Simulator")
        g1, g2, g3, g4 = st.columns(4)
        g_target = g1.number_input("Target Goal ($)", value=1000000.0, step=50000.0)
        g_curr = g2.number_input("Current Savings ($)", value=25000.0, step=5000.0)
        g_mo = g3.number_input("Monthly Contribution ($)", value=1200.0, step=100.0)
        g_yrs = g4.slider("Time Horizon (Years)", min_value=3, max_value=45, value=20)

        proj = project_financial_goal("Custom Financial Milestone", g_target, g_curr, g_mo, g_yrs, risk_tol)
        mc = proj["monte_carlo"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Projected Nominal Balance", f"${proj['projected_final_nominal']:,.0f}", f"Real: ${proj['projected_final_real']:,.0f}")
        m2.metric("Monte Carlo Success Prob.", f"{mc['success_probability_nominal_pct']:.1f}%", f"Median: ${mc['p50_median_case']:,.0f}")
        m3.metric("Required Monthly Savings", f"${proj['required_monthly_for_nominal_target']:,.0f}/mo", f"Inflation-Adj: ${proj['required_monthly_for_inflation_adj_target']:,.0f}/mo")
        mc_df = pd.DataFrame(mc["percentiles_by_year"]).set_index("year")
        st.line_chart(mc_df[["p10", "p50_median", "p90"]])

    # TAB 5: Tax & Knowledge Base Explorer
    with tabs[4]:
        st.subheader("🏛️ Tax Optimization & 108-Article Financial Knowledge Base")
        tax_res = calculate_tax_scenarios(annual_income=annual_inc, filing_status=filing_stat)
        st.json(tax_res, expanded=False)
        search_q = st.text_input("Search Knowledge Base Articles & Glossary", value="Roth IRA vs Traditional 401k")
        if search_q:
            hits = workflow.rag.retrieve(search_q, top_k=4)
            for h in hits:
                with st.expander(f"{h.chunk.title} ({h.chunk.category}) — Score: {h.score:.2f}"):
                    st.write(h.chunk.content)
                    st.markdown(f"**Source:** {h.citation}")

    # TAB 6: Telemetry & MCP Inspector
    with tabs[5]:
        st.subheader("⚙️ LangGraph Execution Trace & MCP Server Tools")
        mcp_tools = st.session_state.mcp_server.list_tools()
        st.dataframe(pd.DataFrame(mcp_tools)[["name", "description"]], use_container_width=True)


if __name__ == "__main__":
    render_streamlit_app()
