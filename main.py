from __future__ import annotations

import argparse
import os
import time
from itertools import cycle
from threading import Event, Thread
from typing import cast

from dotenv import load_dotenv

from agent.graph import build_graph
from agent.state import AgentState

load_dotenv()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph Helper Agent (Gemini + Offline/Online)")
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Your question about LangGraph/LangChain (omit when using --interactive)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=os.getenv("AGENT_MODE", "auto"),
        choices=["offline", "online", "auto"],
        help="Operating mode: offline (local docs), online (web), auto (prefer offline, enrich online if needed)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.getenv("DATA_DIR", os.path.join(os.path.dirname(__file__), "data")),
        help="Path to data directory (docs, index). Defaults to ./data",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        help="Gemini model name (default: gemini-1.5-flash)",
    )
    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start an interactive REPL and stream output as it is generated",
    )
    return parser.parse_args()


def _run_spinner(message: str, stop_event: Event, delay: float = 0.25) -> None:
    dots = cycle([".", "..", "...", "...."])
    while not stop_event.is_set():
        suffix = next(dots)
        print(f"\r{message}{suffix:4}", end="", flush=True)
        time.sleep(delay)
    print("\r" + " " * (len(message) + 4) + "\r", end="", flush=True)


def main() -> None:  # noqa: C901
    args = parse_args()
    os.environ["DATA_DIR"] = args.data_dir
    os.environ["GEMINI_MODEL"] = args.model
    graph = build_graph().compile()

    if getattr(args, "interactive", False):
        print("---------------------------------------------------------------------------")
        print("|    Hi, I'm LangGraphHelper, your LangGraph/LangChain assistant.          |")
        print("|  Ask me anything about LangGraph or LangChain, or press Ctrl+C to exit.  |")
        print("---------------------------------------------------------------------------")
        print()
        while True:
            try:
                q = input("You: ").strip()
                if not q:
                    continue

                stop_event = Event()
                spinner_thread = Thread(
                    target=_run_spinner,
                    args=("Thinking", stop_event),
                    daemon=True,
                )
                spinner_thread.start()

                init_state: AgentState = {
                    "mode": args.mode,
                    "question": q,
                }
                try:
                    result: AgentState = cast(AgentState, graph.invoke(init_state))
                finally:
                    stop_event.set()
                    spinner_thread.join()

                print()

                answer = (result.get("answer") or "").strip()
                print("LangGraphHelper: ", end="", flush=True)
                if answer:
                    delay_s = 0.002
                    try:
                        delay_env = os.getenv("STREAM_DELAY_S")
                        if delay_env is not None:
                            delay_s = float(delay_env)
                    except Exception:
                        delay_s = 0.002
                    for ch in answer:
                        print(ch, end="", flush=True)
                        if delay_s > 0:
                            time.sleep(delay_s)
                    print()
                else:
                    print()

                sources = result.get("sources", [])
                url_sources = [s for s in sources if isinstance(s, str) and s.startswith("http")]
                if url_sources:
                    print("Sources:")
                    for s in url_sources:
                        print(s)
                print()
            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
        return

    if not args.question:
        raise SystemExit("Error: question is required unless --interactive is set.")

    init_state = {
        "mode": args.mode,
        "question": args.question,
    }
    result = cast(AgentState, graph.invoke(init_state))
    answer = result.get("answer", "").strip()
    sources = result.get("sources", [])
    used_mode = result.get("used_mode", args.mode)

    print(f"Mode: {used_mode}")
    print("Answer:")
    print(answer or "(no answer)")
    url_sources = [s for s in sources if isinstance(s, str) and s.startswith("http")]
    if url_sources:
        print("Sources:")
        for s in url_sources:
            print(s)


if __name__ == "__main__":
    main()
