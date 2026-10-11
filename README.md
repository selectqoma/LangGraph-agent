# LangGraph Helper Agent

A command-line assistant for LangGraph and LangChain questions. It uses Gemini to answer from a local FAISS index, web search, or both.

## Run

Requires Python 3.10+ and `uv`.

```sh
uv sync
export GOOGLE_API_KEY=your_key
export GEMINI_MODEL=your_available_gemini_model
uv run python main.py --mode online "How do I add persistence to a LangGraph agent?"
```

Set `TAVILY_API_KEY` to use Tavily search; otherwise the agent uses DuckDuckGo. Environment settings can also go in `.env`. The code defaults to `gemini-1.5-flash`; choose a model available to your account.

For local documentation retrieval, download the docs and build the index first:

```sh
uv run python scripts/prepare_data.py
uv run python main.py --interactive --mode auto
```

Modes: `offline` retrieves only local docs, `online` searches the web, and `auto` adds web results when local context is thin. **All modes call Gemini over the network.** Preparing the index also downloads documentation and an embedding model.

Use `DATA_DIR` to change the data location. If the embedding model is not cached, set `LOCAL_EMBEDDINGS_ONLY=0` to allow downloads. Re-run `prepare_data.py` to refresh the index. Questions are restricted to LangGraph/LangChain by default; `ALLOW_ANY_QUERY=1` removes that check.

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for implementation details.
