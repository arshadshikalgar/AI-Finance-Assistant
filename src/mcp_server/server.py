"""
Model Context Protocol (MCP) Server for Claude Desktop & External Agent Integration.
Implements JSON-RPC 2.0 MCP protocol (`initialize`, `tools/list`, `tools/call`, `resources/list`)
over stdio or direct programmatic dispatch, exposing 6 financial tools to Claude Desktop.
"""
from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Optional

from src.utils.financial_math import (
    analyze_portfolio,
    calculate_tax_scenarios,
    project_financial_goal,
)
from src.workflow.graph import FinanceAssistantWorkflow


class FinanceMCPServer:
    """Model Context Protocol (MCP) Server exposing AI Finance Assistant capabilities."""

    SERVER_INFO = {
        "name": "ai-finance-assistant-mcp",
        "version": "2.0.0",
        "protocolVersion": "2024-11-05",
    }

    def __init__(self, workflow: Optional[FinanceAssistantWorkflow] = None) -> None:
        self.workflow = workflow or FinanceAssistantWorkflow()

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns MCP-compliant tool definitions with JSON Schema input specifications."""
        return [
            {
                "name": "search_financial_knowledge",
                "description": "Searches the 108-article financial education RAG database and 51-term glossary with source citations.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Financial question or concept to search"},
                        "category": {"type": "string", "description": "Optional category filter (e.g. investing_basics, tax_and_retirement)"},
                        "top_k": {"type": "integer", "default": 4},
                    },
                    "required": ["query"],
                },
            },
            {
                "name": "analyze_investment_portfolio",
                "description": "Calculates Modern Portfolio Theory metrics (Sharpe, Sortino, Beta, HHI diversification score, fee drag) and rebalancing trades.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "portfolio_key": {
                            "type": "string",
                            "description": "Preset portfolio key (beginner_balanced, tech_concentrated_growth, conservative_income)",
                            "default": "beginner_balanced",
                        },
                        "risk_tolerance": {
                            "type": "string",
                            "enum": ["Conservative", "Moderate", "Aggressive"],
                            "default": "Moderate",
                        },
                        "holdings": {
                            "type": "array",
                            "description": "Optional custom list of holdings [{ticker, shares, avg_cost, current_price, asset_class}]",
                        },
                    },
                },
            },
            {
                "name": "get_market_quote_and_technicals",
                "description": "Retrieves real-time/cached stock & ETF quotes, SMA-50, SMA-200, RSI-14, MACD trend, and macro indicators.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "ticker": {"type": "string", "description": "Ticker symbol (e.g. VTI, SPY, AAPL, NVDA, BND)"},
                    },
                    "required": ["ticker"],
                },
            },
            {
                "name": "project_financial_goal_monte_carlo",
                "description": "Runs deterministic compound growth and 1,000-trial Monte Carlo simulations for a financial goal.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "goal_name": {"type": "string", "default": "Retirement Independence"},
                        "target_amount": {"type": "number", "default": 1000000},
                        "current_savings": {"type": "number", "default": 25000},
                        "monthly_contribution": {"type": "number", "default": 1200},
                        "years": {"type": "integer", "default": 20},
                        "risk_tolerance": {"type": "string", "default": "Moderate"},
                    },
                    "required": ["target_amount", "years"],
                },
            },
            {
                "name": "calculate_tax_and_account_strategy",
                "description": "Calculates marginal vs effective federal tax rates, capital gains taxes, Roth vs Traditional benefits, and tax-loss harvesting.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "annual_income": {"type": "number", "default": 115000},
                        "filing_status": {"type": "string", "default": "single"},
                        "annual_contribution": {"type": "number", "default": 7000},
                        "capital_gain_amount": {"type": "number", "default": 10000},
                        "holding_period_months": {"type": "integer", "default": 18},
                    },
                },
            },
            {
                "name": "ask_multi_agent_finance_assistant",
                "description": "Invokes the full 6-agent LangGraph orchestrator to answer complex multi-domain financial queries.",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Natural language financial question"},
                        "session_id": {"type": "string", "default": "mcp-session"},
                    },
                    "required": ["query"],
                },
            },
        ]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Executes a registered MCP tool and returns an MCP content response."""
        args = arguments or {}
        if name == "search_financial_knowledge":
            bundle = self.workflow.rag.build_context_prompt(
                query=str(args.get("query", "")),
                top_k=int(args.get("top_k", 4)),
                category=args.get("category"),
            )
            payload = {
                "query": bundle["query"],
                "citations": bundle["citations"],
                "glossary_matches": bundle["glossary_matches"],
                "context_text": bundle["context_text"],
            }
        elif name == "analyze_investment_portfolio":
            holdings = args.get("holdings")
            if not holdings:
                p_key = args.get("portfolio_key", "beginner_balanced")
                p_Agent = self.workflow.agents["portfolio_analysis_agent"]
                sample = p_Agent.sample_portfolios.get(p_key) or next(iter(p_Agent.sample_portfolios.values()))
                holdings = sample["holdings"]
            payload = analyze_portfolio(
                holdings=holdings,
                risk_tolerance=str(args.get("risk_tolerance", "Moderate")),
            )
        elif name == "get_market_quote_and_technicals":
            ticker = str(args.get("ticker", "SPY"))
            quote = self.workflow.market.get_quote(ticker)
            payload = quote.to_dict()
        elif name == "project_financial_goal_monte_carlo":
            payload = project_financial_goal(
                goal_name=str(args.get("goal_name", "Retirement Goal")),
                target_amount=float(args.get("target_amount", 1000000.0)),
                current_savings=float(args.get("current_savings", 25000.0)),
                monthly_contribution=float(args.get("monthly_contribution", 1200.0)),
                years=int(args.get("years", 20)),
                risk_tolerance=str(args.get("risk_tolerance", "Moderate")),
            )
        elif name == "calculate_tax_and_account_strategy":
            payload = calculate_tax_scenarios(
                annual_income=float(args.get("annual_income", 115000.0)),
                filing_status=str(args.get("filing_status", "single")),
                annual_contribution=float(args.get("annual_contribution", 7000.0)),
                capital_gain_amount=float(args.get("capital_gain_amount", 10000.0)),
                holding_period_months=int(args.get("holding_period_months", 18)),
            )
        elif name == "ask_multi_agent_finance_assistant":
            res = self.workflow.process_query(
                query=str(args.get("query", "")),
                session_id=str(args.get("session_id", "mcp-session")),
            )
            payload = {
                "final_response": res["final_response"],
                "classified_intents": res["classified_intents"],
                "citations": res["citations"],
                "total_latency_ms": res["total_latency_ms"],
            }
        else:
            raise ValueError(f"Unknown MCP tool: {name}")

        return {
            "content": [{"type": "text", "text": json.dumps(payload, indent=2)}],
            "structured": payload,
            "isError": False,
        }

    def handle_jsonrpc_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handles a standard JSON-RPC 2.0 MCP message."""
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": self.SERVER_INFO["protocolVersion"],
                    "serverInfo": self.SERVER_INFO,
                    "capabilities": {"tools": {"listChanged": False}},
                }
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                tool_name = params.get("name", "")
                tool_args = params.get("arguments") or {}
                result = self.call_tool(tool_name, tool_args)
            elif method == "ping":
                result = {"status": "pong"}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }
            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32602, "message": str(exc)},
            }

    def run_stdio(self) -> None:
        """Runs the MCP server loop over standard input/output for Claude Desktop."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_jsonrpc_request(req)
                sys.stdout.write(json.dumps(resp) + "\n")
                sys.stdout.flush()
            except Exception as exc:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


if __name__ == "__main__":
    server = FinanceMCPServer()
    server.run_stdio()
