"""
Comprehensive Unit & Integration Test Suite for the AI Finance Assistant.
Covers:
- Knowledge Base (108 articles, 51 glossary terms) & RAG Vector Store (chunking, embedding, retrieval, category filtering, citations)
- Core Config, Rate Limiter, and LLM Client fallback & synthesis
- Market Data Provider (30-min TTL memory/disk caching, backoff retries, technicals, custom ticker synthesis, news feed)
- Quantitative Financial Math (MPT metrics, Sharpe/Sortino, HHI diversification score, rebalancing, goal projection, 1000-trial Monte Carlo, tax calculator)
- All 6 Specialized Agents (Q&A, Portfolio, Market, Goal, News, Tax) + error handling fallbacks + edge cases
- LangGraph StateGraph Workflow Orchestrator (single-agent routing, multi-agent collaboration, multi-turn conversation memory)
- Model Context Protocol (MCP) Server JSON-RPC 2.0 protocol, stdio transport, & all 6 financial tools
- Web App metadata verification
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents import (
    FinanceQAAgent,
    GoalPlanningAgent,
    MarketAnalysisAgent,
    NewsSynthesizerAgent,
    PortfolioAnalysisAgent,
    TaxEducationAgent,
)
from src.core.config import AppConfig, LLMClient, SlidingWindowRateLimiter
from src.data.market_provider import MarketDataProvider
from src.mcp_server.server import FinanceMCPServer
from src.rag.retriever import FinanceRAGRetriever
from src.rag.vector_store import FAISSVectorStore, HybridEmbedder, TextChunker
from src.utils.financial_math import (
    analyze_portfolio,
    calculate_tax_scenarios,
    project_financial_goal,
)
from src.web_app.app import get_app_metadata
from src.workflow.graph import FinanceAssistantWorkflow, NativeStateGraph
from src.workflow.router import IntentRouter


class TestKnowledgeBaseAndRAG(unittest.TestCase):
    """Tests knowledge base curation (100+ articles) and RAG vector retrieval."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = FinanceRAGRetriever()

    def test_knowledge_base_has_over_100_articles_and_50_glossary_terms(self) -> None:
        self.assertGreaterEqual(self.retriever.total_articles, 100)
        self.assertGreaterEqual(self.retriever.total_chunks, 100)
        self.assertGreaterEqual(len(self.retriever.glossary), 50)
        categories = self.retriever.get_categories()
        self.assertGreaterEqual(len(categories), 7)

    def test_chunker_and_embedder_edge_cases(self) -> None:
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=0)
        with self.assertRaises(ValueError):
            TextChunker(chunk_size=100, chunk_overlap=100)

        chunker = TextChunker(chunk_size=120, chunk_overlap=30)
        long_text = (
            "Compound interest accelerates wealth creation over multi-decade investment horizons. "
            "Diversification across low-cost index funds reduces idiosyncratic company risk significantly. "
            "Tax-advantaged accounts like the Roth IRA and HSA shelter investment returns from annual taxation."
        )
        chunks = chunker.chunk_text(long_text)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunker.chunk_text("   "), [])

        embedder = HybridEmbedder(max_features=64)
        with self.assertRaises(RuntimeError):
            embedder.encode(["unfitted test"])

        store = FAISSVectorStore()
        self.assertEqual(store.build_index([]), 0)
        self.assertEqual(store.search(""), [])

    def test_semantic_retrieval_and_category_filtering(self) -> None:
        results = self.retriever.retrieve("What is the Bogleheads Three-Fund Portfolio?", top_k=3)
        self.assertEqual(len(results), 3)
        self.assertIn("Three-Fund", results[0].chunk.title)
        self.assertGreater(results[0].score, 0.20)
        self.assertTrue(results[0].citation.startswith("["))
        self.assertIn("citation", results[0].to_dict())

        tax_results = self.retriever.retrieve(
            "wash sale rule and tax loss harvesting",
            top_k=3,
            category="tax_and_retirement",
        )
        self.assertTrue(all(r.chunk.category == "tax_and_retirement" for r in tax_results))
        self.assertEqual(self.retriever.retrieve("   "), [])

    def test_glossary_lookup_and_context_prompt(self) -> None:
        hits = self.retriever.lookup_glossary("Sharpe Ratio")
        self.assertGreaterEqual(len(hits), 1)
        self.assertEqual(hits[0]["term"], "Sharpe Ratio")
        default_hits = self.retriever.lookup_glossary("", limit=3)
        self.assertEqual(len(default_hits), 3)

        bundle = self.retriever.build_context_prompt("How does an HSA triple tax advantage work?")
        self.assertIn("context_text", bundle)
        self.assertGreaterEqual(len(bundle["citations"]), 1)


class TestConfigRateLimiterAndMarketData(unittest.TestCase):
    """Tests configuration loading, rate limiter, and MarketDataProvider caching/fallbacks."""

    def test_config_and_rate_limiter_and_llm_client(self) -> None:
        cfg = AppConfig.load()
        self.assertEqual(cfg.app_name, "AI Finance Assistant")
        self.assertEqual(cfg.market_cache_ttl_seconds, 1800)

        with self.assertRaises(ValueError):
            SlidingWindowRateLimiter(0)

        limiter = SlidingWindowRateLimiter(max_requests_per_minute=2)
        self.assertTrue(limiter.acquire())
        self.assertTrue(limiter.acquire())
        self.assertFalse(limiter.acquire())
        self.assertEqual(limiter.current_usage, 2)

        llm = LLMClient(cfg)
        fallback_txt = llm.generate("Sys", "General question with empty context", {})
        self.assertIn("Educational Disclaimer", fallback_txt)

    def test_market_provider_caching_retries_and_custom_ticker_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            provider = MarketDataProvider(cache_ttl_seconds=1800, cache_dir=Path(tmpdir), max_retries=2)
            with self.assertRaises(ValueError):
                provider.get_quote("   ")

            q1 = provider.get_quote("VTI")
            self.assertEqual(q1.ticker, "VTI")
            self.assertEqual(len(q1.price_history_30d), 30)
            self.assertEqual(q1.price_history_30d[-1], q1.price)

            # Second fetch should come from memory cache
            q2 = provider.get_quote("vti")
            self.assertIn("MEMORY_CACHE", q2.freshness)

            # Clear memory cache to verify disk cache hit
            provider._memory_cache.clear()
            q3 = provider.get_quote("VTI")
            self.assertIn("DISK_CACHE", q3.freshness)

            # Unknown ticker deterministic fallback
            q_custom = provider.get_quote("XYZC")
            self.assertEqual(q_custom.ticker, "XYZC")
            self.assertEqual(q_custom.freshness, "SYNTHESIZED_FALLBACK")

            # Retry with backoff helper test
            attempts = {"count": 0}
            def flaky_fn() -> str:
                attempts["count"] += 1
                if attempts["count"] < 2:
                    raise RuntimeError("Transient error")
                return "recovered"

            self.assertEqual(provider._retry_with_backoff(flaky_fn), "recovered")

            with self.assertRaises(RuntimeError):
                provider._retry_with_backoff(lambda: (_ for _ in ()).throw(RuntimeError("Permanent")))

            # Market overview & news filters
            overview = provider.get_market_overview()
            self.assertIn("SPY", overview["benchmarks"])
            self.assertEqual(len(overview["sector_performance"]), 11)

            news_nvda = provider.get_financial_news(ticker="NVDA", category="Technology")
            self.assertGreaterEqual(len(news_nvda), 1)


class TestFinancialMathEngine(unittest.TestCase):
    """Tests portfolio MPT analytics, goal projections, Monte Carlo, and tax calculations."""

    def test_portfolio_analysis_metrics_and_edge_cases(self) -> None:
        with self.assertRaises(ValueError):
            analyze_portfolio([])
        with self.assertRaises(ValueError):
            analyze_portfolio([{"ticker": "VTI", "shares": 0, "current_price": 100.0}])

        holdings = [
            {"ticker": "VTI", "shares": 100, "avg_cost": 220.0, "current_price": 278.50, "asset_class": "US Equity", "expense_ratio": 0.0003},
            {"ticker": "VXUS", "shares": 150, "avg_cost": 55.0, "current_price": 62.40, "asset_class": "International Equity", "expense_ratio": 0.0007},
            {"ticker": "BND", "shares": 120, "avg_cost": 74.0, "current_price": 73.20, "asset_class": "Fixed Income", "expense_ratio": 0.0003},
        ]
        res = analyze_portfolio(holdings, risk_tolerance="Moderate")
        self.assertAlmostEqual(res["total_value"], 27850.0 + 9360.0 + 8784.0, places=2)
        self.assertGreater(res["metrics"]["diversification_score"], 70.0)
        self.assertGreater(res["metrics"]["sharpe_ratio"], 0.15)
        self.assertLess(res["metrics"]["weighted_expense_ratio_pct"], 0.06)

        # Conservative low-volatility portfolio branch
        cons_res = analyze_portfolio(
            [{"ticker": "SGOV", "shares": 500, "avg_cost": 100.0, "current_price": 100.35, "asset_class": "Cash Equivalents", "volatility": 0.005}],
            risk_tolerance="Conservative",
        )
        self.assertEqual(cons_res["metrics"]["assessed_risk_level"], "Conservative")

    def test_goal_projection_and_monte_carlo(self) -> None:
        with self.assertRaises(ValueError):
            project_financial_goal("Invalid", 100000, 1000, 100, years=0)
        with self.assertRaises(ValueError):
            project_financial_goal("Invalid", -500, 1000, 100, years=10)
        with self.assertRaises(ValueError):
            project_financial_goal("Invalid", 100000, -10, 100, years=10)

        for profile in ["Conservative", "Moderate", "Aggressive"]:
            proj = project_financial_goal(
                goal_name="Retirement",
                target_amount=1_000_000.0,
                current_savings=50_000.0,
                monthly_contribution=1_500.0,
                years=20,
                risk_tolerance=profile,
                num_simulations=200,
            )
            self.assertEqual(len(proj["yearly_schedule"]), 20)
            self.assertGreater(proj["projected_final_nominal"], proj["total_contributions"])
            mc = proj["monte_carlo"]
            self.assertLess(mc["p10_bear_case"], mc["p50_median_case"])
            self.assertLess(mc["p50_median_case"], mc["p90_bull_case"])

    def test_tax_scenario_calculator_all_brackets(self) -> None:
        with self.assertRaises(ValueError):
            calculate_tax_scenarios(annual_income=-100.0)

        # Single moderate income long-term capital gain
        tax_single = calculate_tax_scenarios(
            annual_income=125000.0,
            filing_status="single",
            annual_contribution=7000.0,
            capital_gain_amount=10000.0,
            holding_period_months=18,
            unrealized_loss_harvest=4000.0,
        )
        self.assertGreater(tax_single["marginal_tax_rate_pct"], tax_single["effective_tax_rate_pct"])
        self.assertEqual(tax_single["tax_loss_harvesting"]["carryforward_loss"], 1000.0)

        # Married high-income short-term capital gain (20% LTCG vs 37% marginal)
        tax_high = calculate_tax_scenarios(
            annual_income=850000.0,
            filing_status="married filing jointly",
            annual_contribution=23500.0,
            capital_gain_amount=50000.0,
            holding_period_months=6,
            unrealized_loss_harvest=2000.0,
        )
        self.assertEqual(tax_high["marginal_tax_rate_pct"], 37.0)
        self.assertIn("Short-Term", tax_high["capital_gains_analysis"]["classification"])

        # Low income 0% LTCG bracket
        tax_low = calculate_tax_scenarios(annual_income=45000.0, filing_status="single")
        self.assertEqual(tax_low["capital_gains_analysis"]["long_term_tax_dollars"], 0.0)


class TestSixAgentsWorkflowAndMCP(unittest.TestCase):
    """Tests all 6 specialized agents, LangGraph StateGraph workflow, and MCP Server."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.workflow = FinanceAssistantWorkflow()
        cls.mcp = FinanceMCPServer(cls.workflow)

    def test_all_six_agents_individual_execution_and_branches(self) -> None:
        queries = {
            "finance_qa_agent": "Explain compound interest and the Rule of 72",
            "portfolio_analysis_agent": "Analyze my tech concentrated growth portfolio",
            "market_analysis_agent": "What is the current price and RSI of NVDA, Apple, and BND?",
            "goal_planning_agent": "Plan for a $150k home down payment in 5 years with $20k saved and $2,500/mo conservative",
            "news_synthesizer_agent": "Summarize the latest financial market news and sentiment",
            "tax_education_agent": "Compare Roth IRA vs Traditional 401(k) and capital gains tax on $95k married income",
        }
        for agent_id, q in queries.items():
            agent = self.workflow.agents[agent_id]
            resp = agent.run(q)
            self.assertIsNone(resp.error)
            self.assertEqual(resp.agent_id, agent_id)
            self.assertIn("Educational Disclaimer", resp.response_text)
            self.assertIn("agent_name", resp.to_dict())

        # Portfolio custom holdings & conservative branch
        p_agent = self.workflow.agents["portfolio_analysis_agent"]
        custom_resp = p_agent.run(
            "Analyze my conservative income portfolio",
            context={"refresh_prices": True},
        )
        self.assertIsNone(custom_resp.error)

        custom_holdings_resp = p_agent.run(
            "Analyze custom holdings",
            context={
                "portfolio_name": "Custom Test",
                "holdings": [{"ticker": "VTI", "shares": 50, "avg_cost": 250.0, "current_price": 278.5}],
            },
        )
        self.assertIsNone(custom_holdings_resp.error)

        # Goal agent college & emergency branches
        g_agent = self.workflow.agents["goal_planning_agent"]
        self.assertIsNone(g_agent.run("Save $80,000 for college 529 in 10 years aggressive").error)
        self.assertIsNone(g_agent.run("Build $30,000 emergency fund in 3 years").error)

        # Market agent default tickers when none specified
        m_agent = self.workflow.agents["market_analysis_agent"]
        self.assertIsNone(m_agent.run("Show general market overview").error)

        # Test graceful fallback on empty query
        err_resp = self.workflow.agents["finance_qa_agent"].run("   ")
        self.assertIsNotNone(err_resp.error)
        self.assertEqual(err_resp.structured_data.get("status"), "fallback")

    def test_router_and_langgraph_workflow_collaboration(self) -> None:
        router = IntentRouter()
        forced = router.classify("Any query", context={"force_agent": "tax_education_agent"})
        self.assertEqual(forced.target_agents, ["tax_education_agent"])
        empty_route = router.classify("")
        self.assertEqual(empty_route.target_agents, ["finance_qa_agent"])
        unmatched_route = router.classify("Hello there friend")
        self.assertEqual(unmatched_route.target_agents, ["finance_qa_agent"])

        # Multi-agent query combining portfolio analysis and tax education
        state = self.workflow.process_query(
            "Analyze my portfolio allocation and explain how to reduce capital gains tax with tax-loss harvesting",
            session_id="test-multi-session",
        )
        intents = state["classified_intents"]
        self.assertIn("portfolio_analysis_agent", intents)
        self.assertIn("tax_education_agent", intents)
        self.assertGreaterEqual(len(state["agent_outputs"]), 2)
        self.assertIn("Multi-Agent Collaborative Synthesis", state["final_response"])

        # Follow-up turn verifying session memory preservation
        follow_state = self.workflow.process_query(
            "What if I change it to Aggressive?",
            session_id="test-multi-session",
        )
        self.assertEqual(len(follow_state["conversation_history"]), 2)
        session = self.workflow.get_session("test-multi-session")
        self.assertIn("Turn 1", session.get_recent_context_summary())

        # NativeStateGraph direct edge test
        g = NativeStateGraph()
        g.add_node("step1", lambda s: {"val": s.get("val", 0) + 1})
        g.add_node("step2", lambda s: {"val": s["val"] * 2})
        g.set_entry_point("step1")
        g.add_edge("step1", "step2")
        compiled = g.compile()
        out = compiled.invoke({"val": 3})
        self.assertEqual(out["val"], 8)

    def test_mcp_server_jsonrpc_stdio_and_all_tools(self) -> None:
        init_resp = self.mcp.handle_jsonrpc_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
        self.assertEqual(init_resp["result"]["serverInfo"]["name"], "ai-finance-assistant-mcp")

        ping_resp = self.mcp.handle_jsonrpc_request({"jsonrpc": "2.0", "id": 99, "method": "ping"})
        self.assertEqual(ping_resp["result"]["status"], "pong")

        list_resp = self.mcp.handle_jsonrpc_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
        tools = list_resp["result"]["tools"]
        self.assertEqual(len(tools), 6)

        for tool_call in [
            ("search_financial_knowledge", {"query": "bond duration"}),
            ("analyze_investment_portfolio", {"portfolio_key": "conservative_income"}),
            ("get_market_quote_and_technicals", {"ticker": "AAPL"}),
            ("project_financial_goal_monte_carlo", {"target_amount": 500000, "years": 15}),
            ("calculate_tax_and_account_strategy", {"annual_income": 100000}),
            ("ask_multi_agent_finance_assistant", {"query": "What is the latest market news?"}),
        ]:
            resp = self.mcp.handle_jsonrpc_request({
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": tool_call[0], "arguments": tool_call[1]},
            })
            self.assertIn("result", resp)
            self.assertFalse(resp["result"]["isError"])

        # Test stdio loop with valid and invalid JSON lines
        fake_in = io.StringIO('{"jsonrpc": "2.0", "id": 10, "method": "ping"}\ninvalid_json\n\n')
        fake_out = io.StringIO()
        with patch.object(sys, "stdin", fake_in), patch.object(sys, "stdout", fake_out):
            self.mcp.run_stdio()
        lines = [line for line in fake_out.getvalue().splitlines() if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertIn("pong", lines[0])
        self.assertIn("Parse error", lines[1])

        # Unknown method & unknown tool error handling
        bad_method = self.mcp.handle_jsonrpc_request({"jsonrpc": "2.0", "id": 4, "method": "unknown/method"})
        self.assertIn("error", bad_method)
        bad_tool = self.mcp.handle_jsonrpc_request({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {"name": "nonexistent_tool"},
        })
        self.assertIn("error", bad_tool)

    def test_web_app_metadata(self) -> None:
        meta = get_app_metadata()
        self.assertEqual(meta["agents_count"], 6)
        self.assertEqual(len(meta["tabs"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
