# AI Finance Assistant — Multi-Agent Financial Literacy & Wealth Platform
Deployed link: https://ai-multi-agent-finance-assistant.streamlit.app/

An intelligent, multi-agent conversational financial education and portfolio analytics platform built with **LangGraph (`StateGraph`)**, **FAISS / Hybrid Semantic RAG**, **Google Gemini 2.0 Flash**, **Real-Time/Cached Market Data (`yFinance` & `Alpha Vantage`)**, **Streamlit**, and a **Model Context Protocol (MCP) Server** for Claude Desktop integration.

---

## 🏗️ System Architecture & Key Highlights

```text
User Query / Streamlit UI / Claude Desktop (MCP)
        │
        ▼
┌──────────────────────────────────────────────────────────┐
│            LangGraph StateGraph Orchestrator             │
│  1. Context Loader Node (Session Memory & User Profile)  │
│  2. Multi-Intent Router Node (Single & Co-Intent Route)  │
└───────┬──────────┬──────────┬──────────┬─────────┬───────┘
        │          │          │          │         │
        ▼          ▼          ▼          ▼         ▼
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
   │ Finance ││Portfolio││ Market  ││  Goal   ││  News   ││   Tax   │
   │Q&A Agent││Analysis ││Analysis ││Planning ││Synthesiz││Education│
   │  (RAG)  ││ (MPT)   ││ (Quotes)││(Monte C)││(Sentim.)││ (IRS)   │
   └────┬────┘└────┬────┘└────┬────┘└────┬────┘└────┬────┘└────┬────┘
        └──────────┴──────────┴────┬─────┴──────────┴──────────┘
                                   ▼
                   ┌───────────────────────────────┐
                   │   Response Synthesizer Node   │
                   │ (Citations, Disclaimers, Log) │
                   └───────────────────────────────┘
```

### Six Specialized Financial Agents
1. **Finance Q&A Educator Agent (`src/agents/qa_agent.py`)**: Answers beginner-to-intermediate financial literacy questions grounded in **108 curated articles** across 9 categories and **51 financial glossary terms** with inline source citations.
2. **Portfolio Analysis & MPT Specialist Agent (`src/agents/portfolio_agent.py`)**: Evaluates holdings using Modern Portfolio Theory—calculating total value, unrealized P&L, asset/sector allocation, weighted expense ratio, 10-year/30-year compounded fee drag, Portfolio Beta, Sharpe Ratio, Sortino Ratio, 95% Value-at-Risk (VaR), Herfindahl-Hirschman Index (HHI) 0–100 Diversification Score, and exact rebalancing trades.
3. **Market Intelligence & Technicals Agent (`src/agents/market_agent.py`)**: Retrieves live or 30-minute TTL cached quotes (`src/data/market_provider.py`), 50-day and 200-day Simple Moving Averages (`SMA-50`, `SMA-200`), 14-day Relative Strength Index (`RSI-14`), MACD trend signals, 11 GICS sector returns, and Federal Reserve / Treasury yield curve indicators.
4. **Financial Goal & Monte Carlo Planner Agent (`src/agents/goal_agent.py`)**: Models retirement (FIRE 25x rule), home down payment, education (529), and emergency fund milestones using both deterministic monthly compounding and **1,000-trial geometric Brownian motion Monte Carlo simulations** (reporting 10th, 25th, 50th, 75th, and 90th percentile wealth trajectories and inflation-adjusted targets).
5. **Financial News & Sentiment Synthesizer Agent (`src/agents/news_agent.py`)**: Synthesizes financial headlines, computes aggregate sentiment scores (`-1.0` to `+1.0`), and translates macroeconomic announcements into calm, long-term beginner takeaways.
6. **Tax Education & Account Optimization Agent (`src/agents/tax_agent.py`)**: Computes progressive federal income tax brackets (marginal vs. effective tax rates), short-term vs. long-term capital gains tax savings, Traditional vs. Roth 401(k)/IRA comparisons, HSA triple-tax advantages, and Tax-Loss Harvesting savings under IRS Section 1091 (Wash-Sale Rule).

---

## 📁 Project Directory Structure

```text
ai_finance_assistant/
├── src/
│   ├── agents/
│   │   ├── base_agent.py          # Abstract BaseAgent & AgentResponse schema
│   │   ├── qa_agent.py            # Finance Q&A Educator Agent (RAG + Glossary)
│   │   ├── portfolio_agent.py     # Portfolio Analysis & MPT Specialist Agent
│   │   ├── market_agent.py        # Market Intelligence & Technicals Agent
│   │   ├── goal_agent.py          # Financial Goal & Monte Carlo Planner Agent
│   │   ├── news_agent.py          # Financial News & Sentiment Synthesizer Agent
│   │   └── tax_agent.py           # Tax Education & Account Optimization Agent
│   ├── core/
│   │   └── config.py              # AppConfig, SlidingWindowRateLimiter (60 RPM), LLMClient
│   ├── data/
│   │   ├── knowledge_base/
│   │   │   ├── articles.json      # 108 curated financial education articles (9 categories)
│   │   │   └── glossary.json      # 51 core financial terms and definitions
│   │   ├── portfolios/
│   │   │   └── sample_portfolios.json # Beginner 3-Fund, Tech Growth, Conservative Income
│   │   ├── market_cache/          # 30-minute TTL disk cache for ticker quotes
│   │   └── market_provider.py     # yFinance / Alpha Vantage client + caching + retries
│   ├── rag/
│   │   ├── vector_store.py        # TextChunker, HybridEmbedder, FAISSVectorStore
│   │   └── retriever.py           # FinanceRAGRetriever with category filter & citations
│   ├── utils/
│   │   └── financial_math.py      # MPT portfolio analytics, Monte Carlo, Tax calculator
│   ├── workflow/
│   │   ├── state.py               # WorkflowState TypedDict & ConversationSession memory
│   │   ├── router.py              # Multi-intent query router & follow-up resolver
│   │   └── graph.py               # LangGraph StateGraph orchestrator
│   ├── mcp_server/
│   │   └── server.py              # JSON-RPC 2.0 Model Context Protocol (MCP) server
│   └── web_app/
│       └── app.py                 # 6-tab Streamlit application
├── tests/
│   └── test_ai_finance_assistant.py # Unit & integration test suite (96.3% statement coverage)
├── docs/
│   └── TECHNICAL_DESIGN.md        # Detailed architecture, RAG, and protocol specification
├── config.yaml
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
└── README.md
```

---

## 🚀 Quick Start & Setup Instructions

### 1. Local Python Setup
```bash
cd ai_finance_assistant
pip install -r requirements.txt
cp .env.example .env
# Optional: Add your GEMINI_API_KEY and ALPHA_VANTAGE_API_KEY in .env
```
> **Zero-Key Evaluation Mode:** The entire platform—including all 6 agents, RAG vector search, portfolio analytics, Monte Carlo simulations, market technicals, and tests—runs out-of-the-box even without external API keys using its deterministic synthesis engine and verified market snapshot cache.

### 2. Run the Multi-Tab Streamlit Web Interface
```bash
streamlit run src/web_app/app.py
```

### 3. Run the Test Suite & Coverage Check
```bash
python3 -m unittest discover -s tests -v
```

### 4. Claude Desktop MCP Server Integration
Add the following entry to your `claude_desktop_config.json`:
```json
{
  "mcpServers": {
    "ai-finance-assistant": {
      "command": "python3",
      "args": ["/absolute/path/to/ai_finance_assistant/src/mcp_server/server.py"]
    }
  }
}
```

---

## 🛡️ Regulatory & Educational Disclaimer
This system is built strictly for **financial education and informational purposes**. It does not provide personalized fiduciary investment, legal, or tax advice. Every agent response automatically embeds clear disclaimers and source citations (`SEC Investor.gov`, `FINRA`, `IRS`, `Bogleheads`, `CFA Institute`).
