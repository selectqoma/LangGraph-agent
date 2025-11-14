from __future__ import annotations

import html
import os
import random
import re
import time
from dataclasses import dataclass
from urllib.parse import urlparse

import numpy as np
import requests
from sentence_transformers import SentenceTransformer

try:
    from tavily import TavilyClient
except Exception:  # pragma: no cover
    TavilyClient = None

try:
    from duckduckgo_search import DDGS
except Exception:  # pragma: no cover
    DDGS = None
try:
    from duckduckgo_search.exceptions import RatelimitException as DDGRatelimit
except Exception:  # pragma: no cover
    DDGRatelimit = Exception


def _clean_html(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


PREFERRED_DOMAINS = (
    "langchain-ai.github.io",
    "docs.langchain.com",
    "python.langchain.com",
)


def _expand_queries(query: str) -> list[str]:
    q = query.strip()
    variants = [q]
    if "langgraph" not in q.lower():
        variants.append(f"{q} LangGraph")

    seen: set[str] = set()
    out: list[str] = []
    for v in variants:
        if v not in seen:
            out.append(v)
            seen.add(v)
    return out


@dataclass
class OnlineRetriever:
    tavily_api_key: str | None = None
    timeout_s: int = 15
    embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"

    def __post_init__(self) -> None:
        try:
            local_only = os.getenv("LOCAL_EMBEDDINGS_ONLY", "1") == "1"
            self._embedder = SentenceTransformer(self.embed_model_name, local_files_only=local_only)
        except Exception:
            self._embedder = None

    def _search_tavily(
        self, query: str, k: int, include_domains: list[str] | None = None
    ) -> list[dict[str, str]]:
        if TavilyClient is None:
            return []
        if not self.tavily_api_key:
            return []
        client = TavilyClient(api_key=self.tavily_api_key)
        kwargs = {"query": query, "max_results": k}
        if include_domains:
            kwargs["include_domains"] = include_domains
        resp = client.search(**kwargs)
        results = []
        for r in resp.get("results", []):
            url = r.get("url") or r.get("link")
            title = r.get("title") or ""
            snippet = r.get("content") or r.get("snippet") or ""
            text = f"{title}\n{snippet}".strip()
            results.append({"text": _clean_html(text), "source": url or "unknown"})
        return results

    def _search_ddg(
        self, query: str, k: int, site_filters: list[str] | None = None
    ) -> list[dict[str, str]]:
        if DDGS is None:
            return []
        out: list[dict[str, str]] = []
        queries: list[str] = []
        if site_filters:
            for sf in site_filters:
                queries.append(f"site:{sf} {query}")
        else:
            queries.append(query)
        seen = set()
        with DDGS() as ddgs:
            for q in queries:
                try:
                    for r in ddgs.text(q, max_results=min(k, 10)):
                        url = r.get("href") or r.get("link")
                        if not url or url in seen:
                            continue
                        seen.add(url)
                        title = r.get("title") or ""
                        body = r.get("body") or r.get("snippet") or ""
                        text = f"{title}\n{body}".strip()
                        out.append({"text": _clean_html(text), "source": url})
                except DDGRatelimit:
                    time.sleep(0.5 + random.random() * 0.5)
                    continue
        return out

    def _fetch_url_text(self, url: str) -> str | None:
        try:
            headers = {"User-Agent": "LangGraphHelperAgent/0.1"}
            resp = requests.get(url, headers=headers, timeout=self.timeout_s)
            if resp.ok and resp.text:
                return _clean_html(resp.text)
        except Exception:
            return None
        return None

    def _prefer_docs_first(self, results: list[dict[str, str]]) -> list[dict[str, str]]:
        def score(r: dict[str, str]) -> int:
            url = r.get("source") or ""
            try:
                host = urlparse(url).netloc
            except Exception:
                host = ""
            return 0 if any(h in host for h in PREFERRED_DOMAINS) else 1

        return sorted(results, key=score)

    def _filter_allowed(
        self,
        results: list[dict[str, str]],
        allowed_domains: list[str] | None,
        allowed_path_contains: list[str] | None,
    ) -> list[dict[str, str]]:
        if not results:
            return results
        if not allowed_domains and not allowed_path_contains:
            return results
        filtered: list[dict[str, str]] = []
        for r in results:
            url = r.get("source") or ""
            if not url.startswith("http"):
                continue
            try:
                parsed = urlparse(url)
                host = parsed.netloc
                path = parsed.path or ""
            except Exception:
                continue
            domain_ok = True
            if allowed_domains:
                domain_ok = any(ad == host or host.endswith(ad) for ad in allowed_domains)
            path_ok = True
            if allowed_path_contains:
                path_ok = any(s in path for s in allowed_path_contains)
            if domain_ok and path_ok:
                filtered.append(r)
        return filtered

    def _filter_langgraph_context(self, results: list[dict[str, str]]) -> list[dict[str, str]]:
        if not results:
            return results
        filtered: list[dict[str, str]] = []
        for r in results:
            url = (r.get("source") or "").lower()
            text = (r.get("text") or "").lower()
            if "langgraph" in url or "langgraph" in text:
                filtered.append(r)
        return filtered

    def _embed_rerank(
        self, query: str, items: list[dict[str, str]], fetch_pages: bool, max_chars: int
    ) -> list[dict[str, str]]:
        if not items:
            return items
        if self._embedder is None:
            return items
        texts: list[str] = []
        enriched: list[dict[str, str]] = []
        for r in items:
            url = r.get("source") or ""
            text = r.get("text") or ""
            if fetch_pages and url.startswith("http"):
                page = self._fetch_url_text(url)
                if page:
                    enriched.append({"text": page[:max_chars], "source": url})
                    texts.append(page[:max_chars])
                    continue
            enriched.append(r)
            texts.append(text)
        q_emb = self._embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)[0]
        doc_embs = self._embedder.encode(
            texts, convert_to_numpy=True, normalize_embeddings=True
        ).astype(np.float32)
        sims = doc_embs @ q_emb
        ranked = sorted(range(len(sims)), key=lambda i: sims[i], reverse=True)
        return [enriched[i] for i in ranked]

    def _simple_retrieve(
        self,
        query: str,
        k: int,
        fetch_pages: bool,
        prefer_docs: bool,
        max_chars: int,
        allowed_domains: list[str] | None,
        allowed_path_contains: list[str] | None,
        langgraph_only: bool,
    ) -> list[dict[str, str]]:
        results = self._search_tavily(query, k, include_domains=allowed_domains)
        if not results:
            results = self._search_ddg(query, k, site_filters=allowed_domains)
        results = self._filter_allowed(results, allowed_domains, allowed_path_contains)
        if langgraph_only:
            results = self._filter_langgraph_context(results)
        if prefer_docs and results:
            results = self._prefer_docs_first(results)
        if fetch_pages:
            enriched: list[dict[str, str]] = []
            for r in results:
                url = r.get("source")
                page = self._fetch_url_text(url) if url and url.startswith("http") else None
                if page:
                    enriched.append({"text": page[:max_chars], "source": url or ""})
                else:
                    enriched.append(r)
            return enriched[:k]
        return results[:k]

    def _thorough_retrieve(
        self,
        query: str,
        k: int,
        fetch_pages: bool,
        prefer_docs: bool,
        max_chars: int,
        allowed_domains: list[str] | None,
        allowed_path_contains: list[str] | None,
        langgraph_only: bool,
    ) -> list[dict[str, str]]:
        queries = _expand_queries(query)
        pool: list[dict[str, str]] = []
        seen_urls: set[str] = set()
        for q in queries:
            t_res = self._search_tavily(q, max(k * 3, 15), include_domains=allowed_domains)
            d_res = self._search_ddg(q, max(k * 3, 15), site_filters=allowed_domains)
            for r in t_res + d_res:
                u = r.get("source") or ""
                if not u or u in seen_urls:
                    continue
                seen_urls.add(u)
                pool.append(r)
        if allowed_domains or allowed_path_contains:
            pool = self._filter_allowed(pool, allowed_domains, allowed_path_contains)
        if langgraph_only:
            pool = self._filter_langgraph_context(pool)
        ranked = self._embed_rerank(query, pool, fetch_pages=fetch_pages, max_chars=max_chars)
        if prefer_docs:
            ranked = self._prefer_docs_first(ranked)
        return ranked[:k]

    def retrieve(
        self,
        query: str,
        k: int = 6,
        fetch_pages: bool = False,
        prefer_docs: bool = True,
        max_chars: int = 9000,
        allowed_domains: list[str] | None = None,
        allowed_path_contains: list[str] | None = None,
        thorough: bool = False,
        langgraph_only: bool = True,
    ) -> list[dict[str, str]]:
        if not thorough:
            return self._simple_retrieve(
                query=query,
                k=k,
                fetch_pages=fetch_pages,
                prefer_docs=prefer_docs,
                max_chars=max_chars,
                allowed_domains=allowed_domains,
                allowed_path_contains=allowed_path_contains,
                langgraph_only=langgraph_only,
            )
        return self._thorough_retrieve(
            query=query,
            k=k,
            fetch_pages=fetch_pages,
            prefer_docs=prefer_docs,
            max_chars=max_chars,
            allowed_domains=allowed_domains,
            allowed_path_contains=allowed_path_contains,
            langgraph_only=langgraph_only,
        )
