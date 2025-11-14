from __future__ import annotations

import json
import os
import re
import time
from contextlib import suppress
from pathlib import Path
from typing import Literal, cast

import google.generativeai as genai

from agent.state import AgentState, Mode
from retrievers.online_search import OnlineRetriever
from retrievers.vector_faiss import VectorFaissRetriever
from utils.env import get_env_str, require_env_str


def router(state: AgentState) -> Literal["offline", "online"]:
    mode = cast(Mode, state.get("mode", "auto"))
    if mode == "online":
        return "online"
    return "offline"


def _classify_query_llm(question: str) -> tuple[bool, bool, str | None]:
    """Primary guardrail: use an LLM classifier to decide scope + meta-ness.

    Returns (in_scope, is_meta, reason_if_out_of_scope).
    Falls back to heuristic classification on failure.
    """
    q = (question or "").strip()
    if not q:
        return True, False, None

    api_key = get_env_str("GOOGLE_API_KEY")
    if not api_key:
        return True, False, None

    model_name = get_env_str("GEMINI_MODEL", "gemini-1.5-flash")
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        prompt = (
            "You are a strict classifier for a LangGraph/LangChain developer assistant.\n"
            "Given a user question, decide:\n"
            "- in_scope: true if the question is about LangGraph, LangChain, LLM agents, tools, retrieval, "
            "state graphs, or this assistant's capabilities in that context. false if clearly unrelated "
            "(e.g., recipes, general travel planning, random chat).\n"
            "- meta: true if the user is asking ABOUT the assistant itself (what it can do, how it helps, "
            "who/what it is). Otherwise false.\n\n"
            "Respond ONLY with a single JSON object, no extra text, in the form:\n"
            '{ "in_scope": true/false, "meta": true/false, "reason": "short reason if in_scope is false or unclear" }\n\n'
            f"Question: {q!r}"
        )
        resp = model.generate_content(prompt)
        text = getattr(resp, "text", "") or ""
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            data = json.loads(match.group(0))
            in_scope = bool(data.get("in_scope", True))
            is_meta = bool(data.get("meta", False))
            reason_val = data.get("reason")
            reason = str(reason_val) if isinstance(reason_val, str) and reason_val else None
            return in_scope, is_meta, reason
    except Exception:
        return True, False, None

    return True, False, None


def is_query_in_scope(question: str) -> tuple[bool, str | None]:
    """Backward-compatible helper kept for external callers/tests."""
    allow_any = os.getenv("ALLOW_ANY_QUERY", "0") == "1"
    if allow_any:
        return True, None
    in_scope, _is_meta, reason = _classify_query_llm(question)
    return in_scope, reason


def guard_query(state: AgentState) -> AgentState:
    question = state.get("question", "")
    allow_any = os.getenv("ALLOW_ANY_QUERY", "0") == "1"

    if allow_any:
        return {**state, "guard": "ok", "is_meta": False}  # type: ignore

    in_scope, is_meta, reason = _classify_query_llm(question)
    new_state: AgentState = {**state, "is_meta": is_meta}  # type: ignore
    if in_scope:
        return {**new_state, "guard": "ok"}  # type: ignore

    message = (
        "This assistant has a very niche hobby: it only talks about LangGraph and LangChain. "
        "It won't plan your trip to the Alps, debate your life choices, or chitchat about random stuff. "
        "Please ask something about LangGraph or LangChain instead.\n"
    )
    if reason:
        message += f" ({reason})\n"
    return {**new_state, "guard": "reject", "answer": message, "sources": []}  # type: ignore


def reject_request(state: AgentState) -> AgentState:
    return state


def passthrough(state: AgentState) -> AgentState:
    return state


def retrieve_offline(state: AgentState) -> AgentState:
    question = state["question"]
    is_meta = bool(state.get("is_meta"))
    if is_meta:
        return {
            **state,
            "context_chunks": [],
            "used_mode": "offline",
        }
    data_dir = os.getenv("DATA_DIR") or os.path.join(os.path.dirname(__file__), "..", "data")
    abs_data = os.path.abspath(data_dir)
    vret = VectorFaissRetriever(data_dir=Path(abs_data))
    offline_chunks = vret.retrieve(question, k=12)
    context_chunks: list[dict[str, str]] = offline_chunks
    used_mode: Literal["offline", "online", "offline+online"] = "offline"
    mode = cast(Mode, state.get("mode", "auto"))
    if mode == "auto" and len(offline_chunks) < 3:
        tavily_key = get_env_str("TAVILY_API_KEY")
        online = OnlineRetriever(tavily_api_key=tavily_key)
        online_chunks = online.retrieve(
            question,
            k=8,
            fetch_pages=True,
            thorough=True,
            langgraph_only=True,
            prefer_docs=True,
            max_chars=9000,
        )
        seen2 = set(ch.get("source") for ch in context_chunks if ch.get("source"))
        for ch in online_chunks:
            src = ch.get("source")
            if src and src not in seen2:
                context_chunks.append(ch)
                seen2.add(src)
        used_mode = "offline+online"
    return {
        **state,
        "context_chunks": context_chunks,
        "used_mode": used_mode,
    }


def retrieve_online(state: AgentState) -> AgentState:
    question = state["question"]
    is_meta = bool(state.get("is_meta"))
    if is_meta:
        return {
            **state,
            "context_chunks": [],
            "used_mode": "online",
        }
    tavily_key = get_env_str("TAVILY_API_KEY")
    online = OnlineRetriever(tavily_api_key=tavily_key)
    chunks = online.retrieve(
        question,
        k=10,
        fetch_pages=True,
        thorough=True,
        langgraph_only=True,
        prefer_docs=True,
        max_chars=9000,
    )
    return {**state, "context_chunks": chunks, "used_mode": "online"}


def _format_prompt(question: str, chunks: list[dict[str, str]]) -> str:
    header = (
        "You are a helpful assistant for LangGraph and LangChain developers. "
        "Use ONLY the provided context when possible. If uncertain, say so clearly.\n"
        "Respond in plain text suitable for a CLI (no markdown; avoid **, *, `, #, [] or links).\n\n"
    )
    ctx_lines = []
    for i, ch in enumerate(chunks):
        src = ch.get("source", f"chunk-{i}")
        text = ch.get("text", "").strip()
        title = ch.get("title")
        desc = ch.get("desc")
        if not text:
            continue
        if title or desc:
            header_line = f"[Source: {src}]"
            if title:
                header_line += f" [Title: {title}]"
            if desc:
                header_line += f" [Summary: {desc}]"
            ctx_lines.append(f"{header_line}\n{text}\n")
        else:
            ctx_lines.append(f"[Source: {src}]\n{text}\n")
    context_block = "\n".join(ctx_lines) if ctx_lines else "(no context available)"
    instructions = (
        "Task:\n"
        + f"- Question: {question}\n"
        + "- Provide a concise, accurate answer with references to sources, if available.\n"
    )
    return f"{header}Context:\n{context_block}\n\n{instructions}"


def generate_answer(state: AgentState) -> AgentState:  # noqa: C901
    api_key = require_env_str("GOOGLE_API_KEY")
    model_name = get_env_str("GEMINI_MODEL", "gemini-1.5-flash")
    genai.configure(api_key=api_key)
    question = state["question"]
    is_meta = bool(state.get("is_meta"))
    chunks = state.get("context_chunks", [])

    if is_meta:
        prompt = (
            "You are LangGraphHelper, a CLI assistant for LangGraph and LangChain developers.\n"
            "The user is asking what you can do and how you can help them.\n"
            "Ignore any example personas, brands, or company-specific roles that may appear in documentation.\n"
            "Respond in plain text suitable for a CLI (no markdown; avoid **, *, `, #, [] or links).\n\n"
            "Describe, in a few short sentences, how you help users:\n"
            "- Answer questions about LangGraph concepts (StateGraph, nodes, edges, persistence, etc.).\n"
            "- Explain and walk through LangGraph and LangChain examples from the docs.\n"
            "- Help design and debug LangGraph-based agents and workflows.\n"
            "- Suggest patterns and best practices for state management, tools, and retrieval.\n"
        )
    else:
        prompt = _format_prompt(question, chunks)
    stream_to_stdout = bool(state.get("stream", False))
    try:
        model = genai.GenerativeModel(model_name)
        if stream_to_stdout:
            response = model.generate_content(prompt, stream=True)
            assembled_parts: list[str] = []
            delay_s = 0.002
            try:
                delay_env = os.getenv("STREAM_DELAY_S")
                if delay_env is not None:
                    delay_s = float(delay_env)
            except Exception:
                delay_s = 0.002
            for part in response:
                piece = getattr(part, "text", "") or ""
                if piece:
                    for ch in piece:
                        print(ch, end="", flush=True)
                        if delay_s > 0:
                            time.sleep(delay_s)
                    assembled_parts.append(piece)
            with suppress(Exception):
                response.resolve()
            if assembled_parts:
                answer = "".join(assembled_parts)
            else:
                fallback = model.generate_content(prompt)
                answer = getattr(fallback, "text", "") or ""
                if answer:
                    print(answer, end="", flush=True)
        else:
            response = model.generate_content(prompt)
            answer = getattr(response, "text", "") or ""
    except Exception as e:
        msg = str(e) or ""
        lower = msg.lower()
        if "resourceexhausted" in msg or "quota" in lower or "rate limit" in lower:
            answer = (
                "Gemini API quota or rate limit was reached. "
                "Please wait a bit or switch to a model / project with available quota."
            )
            chunks = []
        else:
            answer = (
                "Gemini generation failed unexpectedly. "
                "Check your GEMINI_MODEL / GOOGLE_API_KEY settings and try again."
            )

    def _sanitize_cli(text: str) -> str:
        if not text:
            return text
        t = text
        t = re.sub(r"```.*?```", "", t, flags=re.DOTALL)
        t = t.replace("`", "")
        t = re.sub(r"\*\*(.*?)\*\*", r"\1", t)
        t = re.sub(r"__(.*?)__", r"\1", t)
        t = re.sub(r"\*(.*?)\*", r"\1", t)
        t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", t)
        t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.MULTILINE)
        t = re.sub(r"\n{3,}", "\n\n", t)
        return t.strip()

    answer = _sanitize_cli(answer)
    sources = []
    for ch in chunks:
        src = ch.get("source")
        if src:
            sources.append(src)
    seen = set()
    uniq_sources = []
    for s in sources:
        if s not in seen:
            uniq_sources.append(s)
            seen.add(s)
    return {**state, "answer": answer.strip(), "sources": uniq_sources}
