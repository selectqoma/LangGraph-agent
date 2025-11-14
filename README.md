## LangGraph Helper Agent (Python, LangGraph + Gemini)

An AI helper agent that answers practical questions about LangGraph and LangChain. It supports two operating modes:

- **offline**: answers from locally prepared documentation via a FAISS vector index (no web access used for retrieval)
- **online**: answers using live web results (Tavily or DuckDuckGo)
- **auto**: prefers offline docs; if they’re insufficient, enriches with online results

The agent uses **Google Gemini** for generation and **LangGraph** for orchestration.


### Features
- **Dual modes**: offline (local vector index) and online (web), switchable via flag or env
- **Data preparation script** fetches docs and selected LangGraph pages, then builds a FAISS vector index
- **Gemini** model integration via Google AI Studio key (free tier)
- **Tavily** search with **DuckDuckGo** fallback (both free options)
- **Interactive REPL** with a “Thinking…” spinner and streamed answer output
- **Portable**: managed with `uv`, using `pyproject.toml` only


## Setup

Prereqs:
- Python 3.10+
- `uv` (Python package manager). Install instructions: `pipx install uv` or see `https://docs.astral.sh/uv/`

Install dependencies with uv:

```bash
uv sync
```


## Prepare Offline Data

Offline mode uses locally stored docs and a vector index:

- Downloads `llms.txt` feeds:
  - `https://langchain-ai.github.io/langgraph/llms.txt`
  - `https://langchain-ai.github.io/langgraph/llms-full.txt`
  - `https://python.langchain.com/llms.txt`
- Fetches additional LangGraph pages discovered via the official sitemap
- Builds a FAISS vector index over chunked content

Run the data preparation script:

```bash
uv run scripts/prepare_data.py
```

This creates:
- `data/docs/*.txt`: downloaded text sources
- `data/index/faiss.index` and `data/index/faiss_meta.pkl`: vector index and metadata
- `data/manifest.json`: manifest of prepared data

You can change the target directory by passing a path:

```bash
uv run scripts/prepare_data.py /absolute/path/to/data
```


## Configure Environment

Supports `.env` via `python-dotenv`. Create a `.env` in the repo root if you prefer:

```
GOOGLE_API_KEY=your_key
TAVILY_API_KEY=optional_key
DATA_DIR=/absolute/path/to/data
AGENT_MODE=auto
GEMINI_MODEL=gemini-1.5-flash
ALLOW_ANY_QUERY=0
STREAM_DELAY_S=0.002
LOCAL_EMBEDDINGS_ONLY=1
```

Required:
- `GOOGLE_API_KEY`: from Google AI Studio (`https://aistudio.google.com/app/apikey`)

Optional:
- `TAVILY_API_KEY`: for Tavily web search (`https://app.tavily.com/`), used in online mode (DuckDuckGo is used if missing)
- `AGENT_MODE`: default mode (`offline`, `online`, or `auto`)
- `DATA_DIR`: absolute path to data directory (defaults to `<repo>/data`)
- `GEMINI_MODEL`: default `gemini-1.5-flash`
- `ALLOW_ANY_QUERY`: set to `1` to disable the LangGraph/LangChain-only guardrail and allow any question
- `STREAM_DELAY_S`: optional float delay in seconds between streamed characters in the REPL (default `0.002`; set to `0` to print instantly)
- `LOCAL_EMBEDDINGS_ONLY`: `1` (default) to load embedding models from local files only; set to `0` to allow downloads from the Hugging Face Hub

Example:

```bash
export GOOGLE_API_KEY="YOUR_KEY"
export TAVILY_API_KEY="optional"
export DATA_DIR="$(pwd)/data"
export AGENT_MODE="auto"
```


## Run Examples

Offline retrieval only:

```bash
uv run python main.py --mode offline "How do I add persistence to a LangGraph agent?"
```

Online retrieval only:

```bash
uv run python main.py --mode online "Show me how to implement human-in-the-loop with LangGraph"
```

Auto (prefer offline, enrich with online if needed):

```bash
uv run python main.py --mode auto "How do I add persistence to a LangGraph agent?"
```

You can also set the mode via env and omit the flag:

```bash
export AGENT_MODE=online
uv run python main.py "What are the latest LangGraph features?"
```
If your account doesn’t have access to your chosen Gemini model, set `GEMINI_MODEL` to a supported one like `gemini-1.5-flash` or `gemini-1.5-pro`.

### Interactive REPL with Streaming
Start an interactive session that streams model output in the terminal:

```bash
uv run python main.py --interactive --mode auto
```

You’ll see a banner, then a `You:` prompt. For each question:

- A `Thinking…` line animates while retrieval + generation run.
- Once ready, `LangGraphHelper:` appears and the answer is printed character-by-character for a live streaming feel.
- If your question is out of scope (not about LangGraph/LangChain), the assistant returns a humorous guardrail message explaining that it only talks about LangGraph/LangChain.

Press Ctrl+C to exit.

## Architecture Overview

- `agent/state.py`: Typed state for LangGraph (`mode`, `question`, `context_chunks`, `answer`, `sources`, `used_mode`)
- `agent/nodes.py`:
  - `guard_query`: uses a small Gemini classifier to decide if the question is in-scope and whether it is “meta” (about the assistant itself)
  - `router`: chooses offline vs online retrieval based on `mode`
  - `retrieve_offline`: vector retrieval from local FAISS index; in `auto`, enriches with online results if offline is too thin
  - `retrieve_online`: live search results (Tavily or DuckDuckGo, with optional page fetch and embedding-based re-ranking)
  - `generate_answer`: formats a prompt with retrieved context (or a fixed self-description for meta questions) and calls Gemini; also sanitizes markdown for CLI display and handles quota errors gracefully
- `agent/graph.py`: constructs a `StateGraph`:
  - `START -> guard -> route -> (offline_retrieve | online_retrieve) -> generate -> END`
  - If the guard rejects the question, it takes `START -> guard -> reject -> END`
- `retrievers/vector_faiss.py`: loads and queries the FAISS index
- `retrievers/online_search.py`: Tavily search with DuckDuckGo fallback; optional page fetch and embedding-based re-ranking
- `utils/env.py`: loads `.env` and provides helpers
- `scripts/prepare_data.py`: fetches docs and pages, chunks them, and builds the FAISS index and manifest
- `main.py`: CLI entrypoint with mode switching, interactive REPL, spinner, and environment config


## Operating Modes

### Offline
- Retrieval uses only the local FAISS vector index created by `scripts/prepare_data.py`
- No web requests are made for retrieval
- Note: Gemini itself is an online API—this agent assumes LLM access over the network. If you require fully offline generation, you can plug in a local LLM by adapting `generate_answer` (documented below).

### Online
- Retrieval uses Tavily (if `TAVILY_API_KEY` present) or DuckDuckGo
- Optionally fetches page content (kept short to avoid overly long contexts)

### Auto
- Runs offline retrieval first; if too few hits, enriches with online retrieval


### Switching Modes
- CLI flag: `--mode offline|online|auto` (see examples below)
- Environment: set `AGENT_MODE=offline|online|auto` and omit the flag
- Data directory: set `DATA_DIR=/absolute/path` to point at a different prepared dataset

### Scope Guardrail

By default, the helper only answers LangGraph/LangChain developer questions:

- In-scope: LangGraph, LangChain, LLM agents, tools, retrieval, graphs, state management, etc.
- Out-of-scope: general chit-chat, unrelated coding questions, travel planning, etc.

Out-of-scope questions receive a friendly message explaining this limitation. To disable this guardrail (for testing or free-form conversations), set:

```bash
export ALLOW_ANY_QUERY=1
```

## Data Freshness Strategy

### Offline
- Sources are the official `llms.txt` feeds plus selected pages from the LangGraph sitemap. To refresh:
  1) Re-run: `uv run scripts/prepare_data.py` (optionally with `DATA_DIR` path)
  2) This downloads the latest docs/pages and rebuilds the FAISS index

### Online
- Uses live search:
  - Tavily (if `TAVILY_API_KEY` is set) for higher-quality, API-first results
  - Falls back to DuckDuckGo if no Tavily key is present
- Optionally fetches page content and re-ranks with embeddings for relevance
- Free tiers are used by default for easy setup
  - Embeddings model: `sentence-transformers/all-MiniLM-L6-v2`


## Swapping the LLM (Optional)

The default `generate_answer` uses Gemini (`google-generativeai`). If you need a fully offline experience, implement a local LLM call in `agent/nodes.py::generate_answer` while preserving the same input/output fields. Keep the prompt formatting and state shape identical.


## Example Questions
- "How do I add persistence to a LangGraph agent?"
- "Show me how to implement human-in-the-loop with LangGraph"
- "How do I handle errors and retries in LangGraph nodes?"
- "What are best practices for state management in LangGraph?"


## Versions and Compatibility
- Python: 3.10+ (project targets 3.11 in tooling)
- Key libraries (see `pyproject.toml` for authoritative ranges):
  - langgraph: >=0.2.34,<0.3.0
  - langchain-core: >=0.3.11,<0.4.0
  - google-generativeai: >=0.8.3,<0.9.0
  - tavily-python: >=0.5.0,<0.6.0
  - duckduckgo-search: >=6.2.11,<7.0.0
  - requests: >=2.32.0,<3.0.0
  - beautifulsoup4: >=4.12.0,<5.0.0
  - faiss-cpu: >=1.8.0,<1.9.0
  - sentence-transformers: >=3.0.0,<4.0.0
  - python-dotenv: >=1.0.0,<2.0.0
  - Tooling: ruff >=0.6,<0.7; mypy >=1.11,<2


## Troubleshooting
- Missing `GOOGLE_API_KEY`: set it from Google AI Studio
- No `data/index/faiss.index`: run the data preparation script
- No Tavily key: online mode will fall back to DuckDuckGo automatically
- Hitting Gemini rate limits or quota: the REPL will show a message about Gemini quota/rate limit instead of crashing; either wait and retry or switch to a model/project with more quota
