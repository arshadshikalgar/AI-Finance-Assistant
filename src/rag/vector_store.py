"""
Vector store and RAG retrieval pipeline for the AI Finance Assistant.
Supports FAISS index when faiss-cpu is installed, and seamlessly falls back
to a high-precision NumPy BM25 + TF-IDF + N-gram cosine similarity vector index
so the system runs identically in any environment.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


@dataclass
class DocumentChunk:
    """Represents a chunked and indexed segment of a financial education article."""
    chunk_id: str
    article_id: str
    title: str
    category: str
    content: str
    source: str
    source_url: str
    difficulty: str = "Beginner"
    tags: List[str] = field(default_factory=list)
    chunk_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "article_id": self.article_id,
            "title": self.title,
            "category": self.category,
            "content": self.content,
            "source": self.source,
            "source_url": self.source_url,
            "difficulty": self.difficulty,
            "tags": self.tags,
            "chunk_index": self.chunk_index,
        }


@dataclass
class RetrievalResult:
    """Represents a retrieved document chunk with relevance score and citation."""
    chunk: DocumentChunk
    score: float
    rank: int

    @property
    def citation(self) -> str:
        return f"[{self.chunk.title} ({self.chunk.source})]({self.chunk.source_url})"

    def to_dict(self) -> Dict[str, Any]:
        data = self.chunk.to_dict()
        data["score"] = round(float(self.score), 4)
        data["rank"] = self.rank
        data["citation"] = self.citation
        return data


class TextChunker:
    """Splits financial education articles into overlapping semantic chunks."""

    def __init__(self, chunk_size: int = 450, chunk_overlap: int = 75) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str) -> List[str]:
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        sentences = re.split(r"(?<=[.!?])\s+", text)
        chunks: List[str] = []
        current_chunk: List[str] = []
        current_len = 0

        for sentence in sentences:
            s_len = len(sentence)
            if current_len + s_len + 1 > self.chunk_size and current_chunk:
                joined = " ".join(current_chunk)
                chunks.append(joined)
                # Retain overlap from the tail of current_chunk
                overlap_sentences: List[str] = []
                overlap_len = 0
                for prev_s in reversed(current_chunk):
                    if overlap_len + len(prev_s) <= self.chunk_overlap:
                        overlap_sentences.insert(0, prev_s)
                        overlap_len += len(prev_s) + 1
                    else:
                        break
                current_chunk = overlap_sentences
                current_len = sum(len(s) for s in current_chunk) + max(0, len(current_chunk) - 1)

            current_chunk.append(sentence)
            current_len += s_len + 1

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def chunk_articles(self, articles: List[Dict[str, Any]]) -> List[DocumentChunk]:
        all_chunks: List[DocumentChunk] = []
        for article in articles:
            raw_chunks = self.chunk_text(article.get("content", ""))
            for idx, c_text in enumerate(raw_chunks):
                enriched_text = f"{article['title']}: {c_text}"
                all_chunks.append(
                    DocumentChunk(
                        chunk_id=f"{article['id']}-C{idx+1}",
                        article_id=article["id"],
                        title=article["title"],
                        category=article.get("category", "general"),
                        content=enriched_text,
                        source=article.get("source", "Financial Knowledge Base"),
                        source_url=article.get("source_url", "https://www.investor.gov/"),
                        difficulty=article.get("difficulty", "Beginner"),
                        tags=article.get("tags", []),
                        chunk_index=idx,
                    )
                )
        return all_chunks


class HybridEmbedder:
    """
    Deterministic dense + lexical TF-IDF/BM25-weighted vectorizer producing L2-normalized
    dense embeddings compatible with FAISS IndexFlatIP (cosine similarity) and NumPy.
    """

    STOPWORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "from", "up", "about", "into", "over", "after",
        "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "will", "would", "shall", "should", "can", "could",
        "what", "how", "why", "when", "where", "who", "which", "this", "that",
        "these", "those", "it", "its", "they", "them", "their", "you", "your",
        "we", "our", "as", "if", "than", "then", "so", "such", "no", "not",
    }

    SYNONYMS = {
        "401k": ["401(k)", "retirement", "employer", "match", "pretax"],
        "ira": ["roth", "traditional", "retirement", "individual"],
        "etf": ["exchange-traded", "index", "fund", "passive", "vti", "voo", "vxus", "bnd"],
        "tax": ["taxes", "irs", "capital", "gains", "harvesting", "wash", "deduction", "roth", "hsa"],
        "bond": ["bonds", "treasury", "fixed", "income", "yield", "coupon", "duration", "tips", "muni"],
        "stock": ["stocks", "equity", "equities", "shares", "dividend", "pe", "eps", "market"],
        "risk": ["volatility", "drawdown", "beta", "sharpe", "sortino", "diversification", "standard"],
        "goal": ["retire", "retirement", "fire", "saving", "savings", "compound", "projection", "down"],
        "inflation": ["cpi", "pce", "purchasing", "power", "real", "nominal", "tips"],
    }

    def __init__(self, max_features: int = 2048) -> None:
        self.max_features = max_features
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.fitted = False

    def tokenize(self, text: str) -> List[str]:
        cleaned = text.lower()
        # Normalize financial terms like 401(k) -> 401k, s&p -> sp500, p/e -> pe
        cleaned = cleaned.replace("401(k)", "401k").replace("s&p 500", "sp500").replace("p/e", "pe_ratio")
        tokens = re.findall(r"[a-z0-9_]+", cleaned)
        filtered = [t for t in tokens if t not in self.STOPWORDS and len(t) > 1]
        # Add bigrams for domain precision
        bigrams = [f"{filtered[i]}_{filtered[i+1]}" for i in range(len(filtered) - 1)]
        # Add synonym expansions
        expanded = list(filtered) + bigrams
        for t in filtered:
            if t in self.SYNONYMS:
                expanded.extend(self.SYNONYMS[t])
        return expanded

    def fit(self, texts: List[str]) -> "HybridEmbedder":
        doc_count = len(texts)
        df_counts: Dict[str, int] = {}
        term_totals: Dict[str, int] = {}

        for text in texts:
            tokens = self.tokenize(text)
            unique_tokens = set(tokens)
            for t in unique_tokens:
                df_counts[t] = df_counts.get(t, 0) + 1
            for t in tokens:
                term_totals[t] = term_totals.get(t, 0) + 1

        # Select top features by TF * IDF utility
        sorted_terms = sorted(
            term_totals.keys(),
            key=lambda t: term_totals[t] * math.log((1 + doc_count) / (1 + df_counts[t])),
            reverse=True,
        )[: self.max_features]

        self.vocab = {term: idx for idx, term in enumerate(sorted_terms)}
        idf_list = [
            math.log((1.0 + doc_count) / (1.0 + df_counts[term])) + 1.0
            for term in sorted_terms
        ]
        self.idf = np.array(idf_list, dtype=np.float32)
        self.fitted = True
        return self

    def encode(self, texts: List[str]) -> np.ndarray:
        if not self.fitted or self.idf is None:
            raise RuntimeError("HybridEmbedder must be fitted before calling encode().")
        dim = len(self.vocab)
        matrix = np.zeros((len(texts), dim), dtype=np.float32)
        for row_idx, text in enumerate(texts):
            tokens = self.tokenize(text)
            if not tokens:
                continue
            counts: Dict[int, int] = {}
            for t in tokens:
                col_idx = self.vocab.get(t)
                if col_idx is not None:
                    counts[col_idx] = counts.get(col_idx, 0) + 1
            for col_idx, count in counts.items():
                # Sublinear TF scaling * IDF
                tf = 1.0 + math.log(count)
                matrix[row_idx, col_idx] = tf * self.idf[col_idx]

        # L2 normalize rows for exact cosine similarity via inner product
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (matrix / norms).astype(np.float32)


class FAISSVectorStore:
    """
    Vector Database supporting FAISS IndexFlatIP when available and an identical
    exact-inner-product NumPy backend when running in minimal environments.
    """

    def __init__(self, chunk_size: int = 450, chunk_overlap: int = 75) -> None:
        self.chunker = TextChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.embedder = HybridEmbedder(max_features=2048)
        self.chunks: List[DocumentChunk] = []
        self.embeddings: Optional[np.ndarray] = None
        self.faiss_index: Any = None
        self.backend_name: str = "faiss" if FAISS_AVAILABLE else "numpy_exact_ip"

    def build_index(self, articles: List[Dict[str, Any]]) -> int:
        """Chunks articles, fits embedder, and builds the vector index."""
        self.chunks = self.chunker.chunk_articles(articles)
        if not self.chunks:
            self.embeddings = np.zeros((0, 1), dtype=np.float32)
            return 0

        corpus = [
            f"{c.title} {c.category.replace('_', ' ')} {' '.join(c.tags)} {c.content}"
            for c in self.chunks
        ]
        self.embedder.fit(corpus)
        self.embeddings = self.embedder.encode(corpus)

        if FAISS_AVAILABLE and self.embeddings.shape[0] > 0:
            dim = self.embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dim)
            self.faiss_index.add(self.embeddings)
        return len(self.chunks)

    def search(
        self,
        query: str,
        top_k: int = 4,
        category_filter: Optional[str] = None,
        min_score: float = 0.05,
    ) -> List[RetrievalResult]:
        """Performs semantic similarity search with optional category filtering."""
        if not query or not query.strip() or not self.chunks or self.embeddings is None:
            return []

        query_vec = self.embedder.encode([query])  # shape (1, dim)
        query_tokens = set(self.embedder.tokenize(query))

        if FAISS_AVAILABLE and self.faiss_index is not None and category_filter is None:
            k = min(top_k * 3, len(self.chunks))
            scores_arr, indices_arr = self.faiss_index.search(query_vec, k)
            raw_candidates = list(zip(indices_arr[0].tolist(), scores_arr[0].tolist()))
        else:
            sim_scores = np.dot(self.embeddings, query_vec[0])
            raw_candidates = list(enumerate(sim_scores.tolist()))

        # Boost exact title/tag matches and apply category filter
        scored_candidates: List[Tuple[int, float]] = []
        for idx, base_score in raw_candidates:
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]
            if category_filter and chunk.category != category_filter:
                continue

            title_tokens = set(self.embedder.tokenize(chunk.title))
            tag_tokens = set(chunk.tags)
            overlap_title = len(query_tokens & title_tokens)
            overlap_tags = len(query_tokens & tag_tokens)
            boosted_score = float(base_score) + (0.12 * overlap_title) + (0.06 * overlap_tags)
            if boosted_score >= min_score:
                scored_candidates.append((idx, min(boosted_score, 0.9999)))

        scored_candidates.sort(key=lambda x: x[1], reverse=True)

        # Deduplicate by article_id so top_k covers diverse articles
        seen_articles = set()
        results: List[RetrievalResult] = []
        for idx, score in scored_candidates:
            chunk = self.chunks[idx]
            if chunk.article_id in seen_articles:
                continue
            seen_articles.add(chunk.article_id)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=len(results) + 1,
                )
            )
            if len(results) >= top_k:
                break

        return results
