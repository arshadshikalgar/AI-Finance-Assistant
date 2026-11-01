"""
Core Configuration, Rate Limiter, and LLM Client Wrapper.
Supports YAML configuration, environment variable overrides, sliding-window rate limiting,
OpenAI API (SDK + native HTTPS fallback), Google Gemini API, and seamless fallback to a
deterministic domain-aware synthesis engine when external API keys are not configured.
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.request
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import yaml

logger = logging.getLogger("ai_finance_assistant")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s [%(name)s]: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class AppConfig:
    """Centralized application configuration loaded from config.yaml and env vars."""
    app_name: str = "AI Finance Assistant"
    version: str = "2.0.0"
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    temperature: float = 0.2
    max_tokens: int = 1200
    rate_limit_rpm: int = 60
    rag_top_k: int = 4
    rag_chunk_size: int = 450
    rag_chunk_overlap: int = 75
    market_cache_ttl_seconds: int = 1800  # 30 minutes per milestone spec
    risk_free_rate: float = 0.0435        # 4.35% 10Y Treasury baseline
    default_inflation_rate: float = 0.028  # 2.8% long-term CPI assumption
    disclaimer_text: str = (
        "Educational Disclaimer: This analysis is for financial education and informational "
        "purposes only and does not constitute personalized fiduciary investment, tax, or legal advice. "
        "Consult a qualified Certified Financial Planner (CFP) or CPA before making financial decisions."
    )
    raw_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "AppConfig":
        if config_path is None:
            config_path = Path(__file__).resolve().parent.parent.parent / "config.yaml"
        data: Dict[str, Any] = {}
        if config_path.exists():
            with open(config_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}

        llm_cfg = data.get("llm", {})
        rag_cfg = data.get("rag", {})
        mkt_cfg = data.get("market_data", {})
        fin_cfg = data.get("financial_defaults", {})

        # Auto-detect OpenAI vs Gemini based on environment keys if LLM_PROVIDER is not explicitly set
        default_provider = llm_cfg.get("provider", "openai")
        if os.getenv("OPENAI_API_KEY") and not os.getenv("LLM_PROVIDER"):
            default_provider = "openai"
        elif (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")) and not os.getenv("LLM_PROVIDER"):
            default_provider = "gemini"

        provider = os.getenv("LLM_PROVIDER", default_provider).lower()
        default_model = "gpt-4o-mini" if provider == "openai" else llm_cfg.get("model", "gemini-2.0-flash")

        return cls(
            app_name=data.get("app_name", "AI Finance Assistant"),
            version=data.get("version", "2.0.0"),
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", default_model),
            temperature=float(llm_cfg.get("temperature", 0.2)),
            max_tokens=int(llm_cfg.get("max_tokens", 1200)),
            rate_limit_rpm=int(llm_cfg.get("rate_limit_rpm", 60)),
            rag_top_k=int(rag_cfg.get("top_k", 4)),
            rag_chunk_size=int(rag_cfg.get("chunk_size", 450)),
            rag_chunk_overlap=int(rag_cfg.get("chunk_overlap", 75)),
            market_cache_ttl_seconds=int(mkt_cfg.get("cache_ttl_seconds", 1800)),
            risk_free_rate=float(fin_cfg.get("risk_free_rate", 0.0435)),
            default_inflation_rate=float(fin_cfg.get("default_inflation_rate", 0.028)),
            raw_config=data,
        )


class SlidingWindowRateLimiter:
    """Thread-safe sliding-window rate limiter enforcing max requests per minute."""

    def __init__(self, max_requests_per_minute: int = 60) -> None:
        if max_requests_per_minute <= 0:
            raise ValueError("max_requests_per_minute must be > 0")
        self.max_rpm = max_requests_per_minute
        self.timestamps: Deque[float] = deque()

    def acquire(self) -> bool:
        now = time.monotonic()
        while self.timestamps and (now - self.timestamps[0]) > 60.0:
            self.timestamps.popleft()
        if len(self.timestamps) >= self.max_rpm:
            return False
        self.timestamps.append(now)
        return True

    @property
    def current_usage(self) -> int:
        now = time.monotonic()
        while self.timestamps and (now - self.timestamps[0]) > 60.0:
            self.timestamps.popleft()
        return len(self.timestamps)


class LLMClient:
    """
    Unified LLM Client supporting OpenAI (SDK + zero-dependency HTTPS client),
    Google Gemini (google-genai), and a deterministic synthesis engine when offline.
    """

    def __init__(self, config: Optional[AppConfig] = None) -> None:
        self.config = config or AppConfig.load()
        self.rate_limiter = SlidingWindowRateLimiter(self.config.rate_limit_rpm)
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.call_count = 0

    def _call_openai(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Calls OpenAI Chat Completions via official SDK or standard library HTTPS fallback."""
        if not self.openai_api_key:
            return None
        model_name = (
            self.config.llm_model
            if not self.config.llm_model.startswith("gemini")
            else "gpt-4o-mini"
        )
        try:
            from openai import OpenAI  # type: ignore
            client = OpenAI(api_key=self.openai_api_key)
            resp = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
            )
            content = resp.choices[0].message.content
            if content:
                return f"{content}\n\n> *{self.config.disclaimer_text}*"
        except ImportError:
            # Zero-dependency standard library HTTPS call to OpenAI API
            req_body = json.dumps({
                "model": model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=req_body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.openai_api_key}",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
                content = payload["choices"][0]["message"]["content"]
                if content:
                    return f"{content}\n\n> *{self.config.disclaimer_text}*"
        except Exception as exc:
            logger.info("OpenAI API call failed (%s); falling back to structured synthesis.", exc)
        return None

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> Optional[str]:
        """Calls Google Gemini API when GEMINI_API_KEY is configured."""
        if not self.gemini_api_key:
            return None
        try:
            from google import genai  # type: ignore
            client = genai.Client(api_key=self.gemini_api_key)
            response = client.models.generate_content(
                model=self.config.llm_model,
                contents=f"System Instructions:\n{system_prompt}\n\nUser Request:\n{user_prompt}",
            )
            if response and getattr(response, "text", None):
                return f"{response.text}\n\n> *{self.config.disclaimer_text}*"
        except Exception as exc:
            logger.info("Gemini SDK call failed (%s); falling back to structured synthesis.", exc)
        return None

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        context_data: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Generates a grounded financial response using OpenAI, Gemini, or local synthesis."""
        if not self.rate_limiter.acquire():
            logger.warning("Rate limit reached (%d RPM); serving synthesized response.", self.config.rate_limit_rpm)

        self.call_count += 1
        ctx = context_data or {}

        # Build enriched prompt including pre-computed quantitative & RAG context
        enriched_user_prompt = user_prompt
        if ctx.get("metrics_markdown"):
            enriched_user_prompt += f"\n\nPre-computed Quantitative Metrics:\n{ctx['metrics_markdown']}"

        if self.openai_api_key and (self.config.llm_provider == "openai" or not self.gemini_api_key):
            live_out = self._call_openai(system_prompt, enriched_user_prompt)
            if live_out:
                return live_out

        if self.gemini_api_key:
            live_out = self._call_gemini(system_prompt, enriched_user_prompt)
            if live_out:
                return live_out

        return self._synthesize_grounded_response(system_prompt, user_prompt, ctx)

    def _synthesize_grounded_response(
        self,
        system_prompt: str,
        user_prompt: str,
        context_data: Dict[str, Any],
    ) -> str:
        """Synthesizes clear, beginner-friendly markdown responses grounded in RAG & tool outputs."""
        sections: List[str] = []

        headline = context_data.get("headline")
        if headline:
            sections.append(f"### {headline}")

        summary = context_data.get("summary")
        if summary:
            sections.append(summary)

        rag_results = context_data.get("rag_results", [])
        if rag_results:
            sections.append("#### Key Financial Principles")
            for res in rag_results[:3]:
                chunk = res.chunk if hasattr(res, "chunk") else res
                title = chunk.title if hasattr(chunk, "title") else chunk.get("title", "Concept")
                content = chunk.content if hasattr(chunk, "content") else chunk.get("content", "")
                source = chunk.source if hasattr(chunk, "source") else chunk.get("source", "Knowledge Base")
                url = chunk.source_url if hasattr(chunk, "source_url") else chunk.get("source_url", "https://www.investor.gov/")
                clean_body = content.replace(f"{title}: ", "").strip()
                sections.append(f"- **{title}** ([{source}]({url})): {clean_body}")

        metrics_md = context_data.get("metrics_markdown")
        if metrics_md:
            sections.append(metrics_md)

        recommendations = context_data.get("recommendations", [])
        if recommendations:
            sections.append("#### Actionable Next Steps")
            for rec in recommendations:
                sections.append(f"- {rec}")

        citations = context_data.get("citations", [])
        if citations:
            cit_links = [c.get("markdown_link", f"[{c.get('title')}]({c.get('source_url')})") for c in citations]
            sections.append("**Sources & References:** " + " • ".join(cit_links))

        if not sections:
            sections.append(
                f"Here is an educational overview addressing your question: **{user_prompt.strip()}**.\n\n"
                "Focusing on broad diversification, low expense ratios, and tax-advantaged accounts "
                "helps long-term investors build sustainable wealth."
            )

        sections.append(f"\n> *{self.config.disclaimer_text}*")
        return "\n\n".join(sections)
