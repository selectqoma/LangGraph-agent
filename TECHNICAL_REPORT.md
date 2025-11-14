## LangGraph Helper Agent – Technical Report

This document explains **what this project implements**, **how it is structured**, and **why key design choices were made**. It is intended for engineers who may extend, operate, or re-use this agent.


### 1. Problem Statement and Goals

- **Goal**: Provide a command-line assistant that answers **practical questions about LangGraph and LangChain**, with strong retrieval and clear scope guardrails.
- **Key requirements**:
  - **High-quality answers about LangGraph/LangChain** based on official documentation and related pages.
  - **Offline-friendly retrieval** via a local vector index, so that most questions can be answered without live web search.
  - **Online retrieval option** to stay up to date and cover gaps in the local corpus.
  - **Simple CLI UX** (single command / interactive REPL) and **portable dependency management** using `uv`.
  - **Strict topical guardrail** so the agent only engages on LangGraph/LangChain topics unless explicitly relaxed.


### 2. High-Level Architecture

At a high level, the system is a **retrieval-augmented LangGraph agent**:

- **Orchestration**: `langgraph.StateGraph` drives the control flow:
  - `agent/graph.py` builds a graph where nodes are Python functions in `agent/nodes.py`.
  - The graph enforces a clear pipeline: **guard → routing → retrieval → generation**.
- **State model**: `agent/state.py` defines `AgentState` (a `TypedDict`) capturing:
  - `mode`: `"offline" | "online" | "auto"`
  - `question`, `context_chunks`, `answer`, `sources`, `used_mode`, `stream`
- **Retrievers**:
  - `retrievers/vector_faiss.py`: offline vector retrieval over a FAISS index built from docs and LangGraph pages.
  - `retrievers/online_search.py`: online retrieval using Tavily and DuckDuckGo, with optional page fetch and embedding-based re-ranking.
- **Data preparation**:
  - `scripts/prepare_data.py`: downloads docs, fetches LangGraph pages, chunks them, and builds the FAISS index and manifest.
- **LLM and guardrails**:
  - `agent/nodes.py`: uses **Google Gemini** both for a lightweight **classifier** (`_classify_query_llm`) and for final **answer generation** (`generate_answer`).
  - **Scope guardrail** ensures the agent stays focused on LangGraph/LangChain topics.
- **CLI**:
  - `main.py`: command-line interface with:
    - `--mode` selection (`offline`, `online`, `auto`),
    - `--interactive` REPL,
    - streaming output and a spinner.

**Why LangGraph?**
- The project is about LangGraph itself; using `langgraph` for orchestration demonstrates its **StateGraph patterns** directly.
- A state graph makes the pipeline **explicit, testable, and easy to extend** (e.g., adding new nodes for tools or logging).


### 3. Orchestration Design (StateGraph)

`agent/graph.py` builds a `StateGraph[AgentState]` with the following structure:

- **Nodes**:
  - `guard` → `guard_query`
  - `route` → `passthrough`
  - `offline_retrieve` → `retrieve_offline`
  - `online_retrieve` → `retrieve_online`
  - `generate` → `generate_answer`
  - `reject` → `reject_request`
- **Edges**:
  - `START → guard`
  - Conditional from `guard`:
    - `"ok"` → `route`
    - `"reject"` → `reject`
  - Conditional from `route` via `router`:
    - `"offline"` → `offline_retrieve`
    - `"online"` → `online_retrieve`
  - `offline_retrieve → generate`
  - `online_retrieve → generate`
  - `generate → END`
  - `reject → END`

**Rationale**:
- Having **guardrail logic as the first node** ensures out-of-scope questions are filtered early, saving compute and avoiding unnecessary web/API calls.
- The **router** separates mode selection logic from retrieval implementation, making it easy to plug in additional retrieval strategies or tools.
- Both offline and online branches converge at a single **generation node**, which centralizes prompt formatting, error handling, and post-processing.


### 4. Retrieval Strategy and Data Pipeline

#### 4.1 Offline Retrieval (Vector + FAISS)

Component: `retrievers/vector_faiss.py`, index built by `scripts/prepare_data.py`.

- **What it does**:
  - Uses `sentence-transformers/all-MiniLM-L6-v2` to embed document chunks.
  - Stores embeddings in a **FAISS inner product index** (`faiss.index`) with metadata in `faiss_meta.pkl`.
  - At query time, encodes the question and returns the top-`k` documents (`text`, `source`, `title`, `desc`).

- **Why FAISS + sentence-transformers?**
  - **FAISS** is a widely-used, efficient library for vector similarity search, suitable for CLI tools and medium-sized corpora.
  - `all-MiniLM-L6-v2` is a **small, fast, and well-known embedding model** that strikes a good balance between quality and performance.
  - The choice of **inner product with normalized embeddings** (`IndexFlatIP`) is standard for cosine-like similarity.
  - `LOCAL_EMBEDDINGS_ONLY` defaulting to `1` favors **offline reproducibility** while allowing opt-in downloads when needed.

- **Why an offline index at all?**
  - Many LangGraph/LangChain questions are about **core concepts and stable APIs** that change infrequently.
  - An offline index:
    - reduces reliance on third-party search APIs,
    - improves latency and robustness,
    - ensures users can still work even with limited network connectivity.


#### 4.2 Online Retrieval (Tavily + DuckDuckGo)

Component: `retrievers/online_search.py`.

- **What it does**:
  - Uses **Tavily** (if `TAVILY_API_KEY` is set) as the primary web search API.
  - Falls back to **DuckDuckGo** via `duckduckgo_search` if Tavily is unavailable.
  - Can **fetch page content** and **re-rank results** with embeddings:
    - Embeds results using the same `SentenceTransformer` model.
    - Performs similarity-based re-ranking against the query.
  - Filters and prioritizes results:
    - Favors **preferred domains** like `langchain-ai.github.io` and `python.langchain.com`.
    - Optionally restricts to LangGraph-related URLs/content.

- **Why Tavily + DuckDuckGo?**
  - Tavily is **API-first** with good relevance on developer documentation.
  - DuckDuckGo provides a **free and simple fallback** when no Tavily key is present.
  - Combining both ensures **resilience** and keeps setup simple: the agent works out of the box even without a Tavily key.

- **Why embedding-based re-ranking?**
  - Search engines can return noisy snippets; embedding-based re-ranking:
    - improves **semantic relevance** to the user’s question,
    - allows **page content fetch** to be used effectively without overwhelming the context window.


#### 4.3 Data Preparation Pipeline

Component: `scripts/prepare_data.py`.

- **Steps**:
  1. **Create directories**: ensures `data/docs` and `data/index` exist.
  2. **Download docs**:
     - `llms.txt` feeds from:
       - `https://langchain-ai.github.io/langgraph/llms.txt`
       - `https://langchain-ai.github.io/langgraph/llms-full.txt`
       - `https://python.langchain.com/llms.txt`
     - Clean raw HTML/markdown-like content into normalized text.
  3. **Fetch LangGraph pages**:
     - Parse URLs from `llms.txt`.
     - Restrict to `langchain-ai.github.io` LangGraph pages.
     - Additionally, fetch URLs from the official LangGraph sitemap and filter to content pages.
  4. **Chunk pages and docs**:
     - Split into **section-aware chunks** with titles and short descriptions.
     - Use sliding windows with overlap for robust retrieval.
  5. **Build FAISS index**:
     - Encode combined documents (docs + pages).
     - Persist FAISS index and metadata in `data/index`.
  6. **Write manifest**:
     - `data/manifest.json` records doc paths, index locations, and counts.

- **Why this pipeline?**
  - By combining:
    - **official feeds** (`llms.txt`),
    - **linked LangGraph pages**, and
    - **sitemap-discovered pages**,
    the index covers both **high-level guides** and **deep topic pages**.
  - Section-aware chunking with short descriptions improves:
    - answer quality,
    - interpretability of sources,
    - future ability to surface explanations with section titles.


### 5. LLM Integration and Prompting

Component: `agent/nodes.py`.

#### 5.1 Query Classification (Guardrail)

- `_classify_query_llm(question)` uses a lightweight Gemini call to classify:
  - `in_scope`: whether the question is about LangGraph/LangChain/LLM agents/retrieval.
  - `meta`: whether the user is asking about the assistant itself.
- `guard_query`:
  - If `ALLOW_ANY_QUERY=1`, **bypasses** guardrails.
  - Otherwise, uses the classifier to either:
    - set `guard="ok"` and propagate `is_meta`, or
    - set `guard="reject"`, with a **friendly out-of-scope message** and empty `sources`.

**Why an LLM-based guardrail?**
- Rule-based keyword filters are brittle and easy to bypass.
- A small, low-cost Gemini classifier provides:
  - better accuracy for distinguishing **developer questions** vs. general chat,
  - the ability to detect **meta questions** and switch prompting strategies accordingly.


#### 5.2 Answer Generation

- `_format_prompt(question, chunks)`:
  - Prepares a **CLI-friendly prompt**:
    - Instructs the model to **avoid markdown** and produce plain text.
    - Includes each chunk with `[Source: ...]` plus optional title/summary.
    - Emphasizes **using only provided context when possible** and being explicit when unsure.
- `generate_answer`:
  - For meta questions:
    - Uses a fixed self-description prompt describing capabilities of "LangGraphHelper".
  - For normal questions:
    - Uses the context-augmented prompt.
  - Supports both:
    - **streaming** (`stream=True`) to the CLI with character-by-character printing.
    - **non-streaming**: one-shot `generate_content`.
  - Handles Gemini failures:
    - **Quota/rate-limit errors** → user-facing message rather than crash.
    - Other failures → generic but actionable error message.
  - Post-processes the answer with `_sanitize_cli` to:
    - strip code fences / markdown symbols,
    - normalize line breaks,
    - convert markdown links into `text (url)` format.

**Why Gemini?**
- Gemini provides:
  - **free-tier access** suitable for personal agents and small projects,
  - strong performance on code and documentation-style tasks,
  - a simple Python SDK (`google-generativeai`) that integrates easily in a CLI setting.
- Centralizing LLM usage to one function (`generate_answer`) makes it straightforward to **swap in another model** later while keeping the rest of the system unchanged.


### 6. CLI UX and Modes

Component: `main.py`.

- **Modes**:
  - `--mode offline`: only offline vector retrieval.
  - `--mode online`: only online retrieval.
  - `--mode auto`:
    - Run offline retrieval.
    - If results are sparse, **enrich with online retrieval**.
    - The final answer is generated from the combined context.
- **Interactive REPL** (`--interactive`):
  - Prints a **banner** explaining what the agent does.
  - For each question:
    - Shows a `Thinking...` spinner (threaded).
    - Invokes the graph and streams the answer.
    - Prints any URL sources at the end.
- **Non-interactive**:
  - Requires a question argument.
  - Prints mode used, answer, and sources.

**Why this CLI design?**
- The agent is intended as a **developer-side tool**:
  - REPL mode supports exploratory problem-solving.
  - One-shot CLI mode supports quick, scriptable queries.
- A spinner and streaming output make the **latency feel shorter** and provide clear feedback that the process is running.


### 7. Environment and Tooling Choices

#### 7.1 Environment Management with `uv`

- `pyproject.toml` specifies dependencies and tooling; `uv.sync` drives installation.
- **Rationale**:
  - `uv` is a **fast, modern Python package and environment manager**, well-suited for reproducible CLI tools.
  - Single-source configuration in `pyproject.toml` simplifies setup and CI/CD integration.


#### 7.2 Linting and Type Checking

- `pyproject.toml` configures:
  - `ruff` for linting and formatting.
  - `mypy` for (optional) static type checking.
- **Rationale**:
  - The codebase uses `TypedDict` (`AgentState`) and type hints extensively; adding mypy gives extra confidence.
  - Ruff enforces a **consistent style** and catches common issues early (unused imports, simple bugs, etc.).


### 8. Error Handling, Robustness, and Trade-offs

- **Graceful degradation**:
  - If Tavily is unavailable or no API key is provided, online retrieval transparently falls back to DuckDuckGo.
  - If embeddings cannot be loaded (e.g., missing model), offline retrieval returns an empty list instead of failing hard.
  - Gemini quota/limit errors are surfaced as user-facing messages.
- **Guardrail bypass**:
  - `ALLOW_ANY_QUERY=1` allows users to test or repurpose the agent beyond LangGraph/LangChain topics.
  - This keeps the default behavior safe and focused while giving power users flexibility.
- **Trade-offs**:
  - The system still depends on **remote LLM APIs** (Gemini); it is not fully offline.
  - Offline index building requires network access once (to fetch docs/pages) and can be **time-consuming**, but it is a one-time or periodic operation.
  - The retrieval logic favors **simplicity and clarity** over maximum complexity (e.g., no multi-hop reasoning or query decomposition yet).


### 9. Extensibility and Future Work

The current design was intentionally kept modular to support future extensions:

- **Additional tools/nodes**:
  - New LangGraph nodes can be added for:
    - code generation examples,
    - running small diagnostics or checks against user-provided snippets,
    - integrating other doc sources or internal knowledge bases.
- **LLM abstraction**:
  - Extracting a small LLM interface around `generate_answer` would make it easy to:
    - plug in local models (e.g., via `llama.cpp` / `vLLM`),
    - support multiple providers behind a feature flag.
- **Richer source display**:
  - The CLI currently prints only URLs as sources.
  - Future work could surface section titles or short summaries alongside URLs for better explainability.
- **Caching**:
  - Add caching for:
    - Embeddings (for frequent queries),
    - Online search results,
    - LLM responses for identical questions.


### 10. Summary of Key Design Justifications

- **LangGraph StateGraph**: makes the agent’s flow **explicit, composable, and easy to extend**; aligns with the educational goal of showcasing LangGraph itself.
- **Offline FAISS index + online search**: balances **speed, reliability, and freshness**; offline answers most questions, online covers gaps and keeps content current.
- **Gemini for both classification and answering**: reuses a single LLM provider for both guardrails and generation, simplifying configuration and leveraging its strengths on documentation-style tasks.
- **Scope guardrail**: keeps the assistant **focused and predictable**, matching its purpose as a LangGraph/LangChain helper rather than a general chatbot.
- **`uv`-based setup and typed code**: improve **reproducibility, maintainability, and readiness for production/CI environments**.
