"""
High-level RAG Knowledge Retriever integrating the 108-article knowledge base,
51-term financial glossary, category filtering, and formatted source citations.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.rag.vector_store import FAISSVectorStore, RetrievalResult


class FinanceRAGRetriever:
    """Manages loading, indexing, semantic search, and glossary lookup."""

    DEFAULT_KB_PATH = (
        Path(__file__).resolve().parent.parent / "data" / "knowledge_base" / "articles.json"
    )
    DEFAULT_GLOSSARY_PATH = (
        Path(__file__).resolve().parent.parent / "data" / "knowledge_base" / "glossary.json"
    )

    def __init__(
        self,
        kb_path: Optional[Path] = None,
        glossary_path: Optional[Path] = None,
        chunk_size: int = 450,
        chunk_overlap: int = 75,
    ) -> None:
        self.kb_path = kb_path or self.DEFAULT_KB_PATH
        self.glossary_path = glossary_path or self.DEFAULT_GLOSSARY_PATH
        self.vector_store = FAISSVectorStore(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.articles: List[Dict[str, Any]] = []
        self.glossary: List[Dict[str, str]] = []
        self._load_and_index()

    def _load_and_index(self) -> None:
        if self.kb_path.exists():
            self.articles = json.loads(self.kb_path.read_text(encoding="utf-8"))
            self.vector_store.build_index(self.articles)
        if self.glossary_path.exists():
            self.glossary = json.loads(self.glossary_path.read_text(encoding="utf-8"))

    @property
    def total_articles(self) -> int:
        return len(self.articles)

    @property
    def total_chunks(self) -> int:
        return len(self.vector_store.chunks)

    def get_categories(self) -> List[str]:
        return sorted({a.get("category", "general") for a in self.articles})

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        category: Optional[str] = None,
        min_score: float = 0.05,
    ) -> List[RetrievalResult]:
        return self.vector_store.search(
            query=query,
            top_k=top_k,
            category_filter=category,
            min_score=min_score,
        )

    def lookup_glossary(self, query: str, limit: int = 5) -> List[Dict[str, str]]:
        if not query or not query.strip():
            return self.glossary[:limit]
        q_lower = query.lower().strip()
        matches: List[Dict[str, str]] = []
        for item in self.glossary:
            term_lower = item["term"].lower()
            def_lower = item["definition"].lower()
            if q_lower in term_lower or term_lower in q_lower:
                matches.insert(0, item)
            elif any(w in term_lower or w in def_lower for w in q_lower.split() if len(w) > 3):
                matches.append(item)
        return matches[:limit]

    def build_context_prompt(
        self,
        query: str,
        top_k: int = 4,
        category: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Returns retrieved chunks, formatted context block, and structured citations."""
        results = self.retrieve(query=query, top_k=top_k, category=category)
        glossary_hits = self.lookup_glossary(query=query, limit=2)

        context_Blocks = []
        citations = []
        for res in results:
            context_Blocks.append(
                f"[Source {res.rank}: {res.chunk.title} | Category: {res.chunk.category} | "
                f"Relevance: {res.score:.2f}]\n{res.chunk.content}"
            )
            citations.append({
                "rank": res.rank,
                "article_id": res.chunk.article_id,
                "title": res.chunk.title,
                "category": res.chunk.category,
                "source": res.chunk.source,
                "source_url": res.chunk.source_url,
                "score": round(res.score, 4),
                "markdown_link": res.citation,
            })

        return {
            "query": query,
            "results": results,
            "context_text": "\n\n".join(context_Blocks),
            "citations": citations,
            "glossary_matches": glossary_hits,
        }
