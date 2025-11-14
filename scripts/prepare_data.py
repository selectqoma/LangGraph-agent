import json
import pickle
import re
import sys
from pathlib import Path

import faiss
import numpy as np
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


DOC_URLS = [
    ("langgraph-llms.txt", "https://langchain-ai.github.io/langgraph/llms.txt"),
    ("langgraph-llms-full.txt", "https://langchain-ai.github.io/langgraph/llms-full.txt"),
    ("langchain-llms.txt", "https://python.langchain.com/llms.txt"),
]


def ensure_dirs(base_dir: Path) -> dict[str, Path]:
    docs_dir = base_dir / "docs"
    index_dir = base_dir / "index"
    docs_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)
    return {"docs": docs_dir, "index": index_dir}


def html_to_visible_text(raw: str) -> str:
    looks_html = (
        "<html" in raw.lower() or "<head" in raw.lower() or "<body" in raw.lower() or "</" in raw
    )
    if not looks_html:
        return normalize_text(raw)
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(
        ["script", "style", "noscript", "header", "footer", "nav", "aside", "meta", "link"]
    ):
        tag.decompose()
    text = soup.get_text(separator="\n")
    return normalize_text(text)


def download_docs(docs_dir: Path) -> list[Path]:
    saved: list[Path] = []
    for filename, url in DOC_URLS:
        dest = docs_dir / filename
        print(f"Downloading {url} -> {dest}")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        cleaned = html_to_visible_text(resp.text)
        dest.write_text(cleaned, encoding="utf-8")
        saved.append(dest)
    return saved


def normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def simple_sentences(text: str, max_chars: int = 200) -> str:
    parts: list[str] = []
    acc = 0
    for seg in text.split("."):
        seg = seg.strip()
        if not seg:
            continue
        seg = seg + "."
        if acc + len(seg) > max_chars and parts:
            break
        parts.append(seg)
        acc += len(seg)
        if acc >= max_chars:
            break
    return " ".join(parts) if parts else text[:max_chars]


def parse_sections_markdown(text: str) -> list[dict[str, str]]:
    """Split by markdown-style headings to get section-aware chunks."""
    lines = text.splitlines()
    sections: list[dict[str, str]] = []
    current_title = "Document"
    current_content: list[str] = []
    heading_stack: list[str] = []

    def push_section():
        if current_content:
            sections.append(
                {
                    "title": " / ".join(heading_stack) or current_title,
                    "content": "\n".join(current_content),
                }
            )

    for line in lines:
        if line.startswith("#"):
            level = len(line) - len(line.lstrip("#"))
            heading = line[level:].strip()

            push_section()
            current_content = []

            if level <= len(heading_stack):
                heading_stack = heading_stack[: level - 1]
            heading_stack.append(heading)
            continue
        current_content.append(line)
    push_section()

    if not sections:
        sections = [{"title": "Document", "content": text}]
    return sections


def build_documents_from_sections(
    file_name: str, raw_text: str, max_chars: int = 1200, overlap: int = 150
) -> list[dict[str, str]]:
    docs: list[dict[str, str]] = []
    sections = parse_sections_markdown(raw_text)
    for _sec_idx, sec in enumerate(sections):
        sec_title = sec.get("title") or "Section"
        sec_text = normalize_text(sec.get("content", ""))
        if not sec_text:
            continue

        start = 0
        while start < len(sec_text):
            end = min(start + max_chars, len(sec_text))
            chunk = sec_text[start:end]
            if not chunk.strip():
                break
            desc = simple_sentences(chunk, max_chars=240)
            docs.append(
                {
                    "text": chunk,
                    "source": f"{file_name}#chunk-{len(docs)}",
                    "title": sec_title,
                    "desc": desc,
                }
            )
            if end == len(sec_text):
                break
            start = max(0, end - overlap)
    return docs


def extract_urls_from_llms(llms_text: str) -> list[str]:
    urls = re.findall(r"https?://[^\s)]+", llms_text)
    cleaned: list[str] = []
    for u in urls:
        if any(u.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg")):
            continue
        cleaned.append(u)
    return cleaned


def html_to_visible_text_page(raw: str) -> str:
    soup = BeautifulSoup(raw, "html.parser")
    for tag in soup(
        ["script", "style", "noscript", "header", "footer", "nav", "aside", "meta", "link"]
    ):
        tag.decompose()
    return normalize_text(soup.get_text(separator="\n"))


def fetch_langgraph_pages_from_llms(
    doc_paths: list[Path], timeout_s: int = 15
) -> list[dict[str, str]]:
    urls: list[str] = []
    for p in doc_paths:
        txt = p.read_text(encoding="utf-8", errors="ignore")
        urls.extend(extract_urls_from_llms(txt))

    urls = [u for u in urls if "langchain-ai.github.io" in u and "/langgraph/" in u]

    seen: set[str] = set()
    unique_urls: list[str] = []
    for u in urls:
        if u not in seen:
            unique_urls.append(u)
            seen.add(u)
    urls = unique_urls
    pages: list[dict[str, str]] = []
    for u in urls:
        try:
            r = requests.get(
                u, timeout=timeout_s, headers={"User-Agent": "LangGraphHelperAgent/0.1"}
            )
            if r.ok and r.text:
                text = html_to_visible_text_page(r.text)
                title = None
                try:
                    soup = BeautifulSoup(r.text, "html.parser")
                    if soup.title and soup.title.string:
                        title = soup.title.string.strip()
                except Exception:
                    title = None
                if text:
                    pages.append({"url": u, "title": title or "", "text": text})
        except Exception:
            continue
    return pages


def fetch_urls_from_sitemap(
    base_url: str = "https://langchain-ai.github.io/langgraph/sitemap.xml", timeout_s: int = 15
) -> list[str]:
    try:
        r = requests.get(
            base_url, timeout=timeout_s, headers={"User-Agent": "LangGraphHelperAgent/0.1"}
        )
        if not (r.ok and r.text):
            return []

        urls = re.findall(r"<loc>(.*?)</loc>", r.text)
        cleaned: list[str] = []
        for u in urls:
            if isinstance(u, str) and u.startswith("http"):
                cleaned.append(u.strip())

        seen = set()
        out = []
        for u in cleaned:
            if u not in seen:
                out.append(u)
                seen.add(u)
        return out
    except Exception:
        return []


def filter_topic_urls(urls: list[str]) -> list[str]:
    if not urls:
        return []

    selected: list[str] = []
    for u in urls:
        if any(
            ext in u for ext in (".png", ".jpg", ".jpeg", ".gif", ".svg", ".pdf", ".zip", ".xml")
        ):
            continue
        selected.append(u)
    return selected[:200]


def chunk_pages_to_documents(
    pages: list[dict[str, str]], max_chars: int = 1200, overlap: int = 150
) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for page in pages:
        url = page["url"]
        title = page.get("title", "")
        text = page.get("text", "")
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + max_chars, len(text))
            chunk = text[start:end]
            if not chunk.strip():
                break
            desc = simple_sentences(chunk, max_chars=240)
            documents.append(
                {
                    "text": chunk,
                    "source": f"{url}#chunk-{len(documents)}",
                    "title": title,
                    "desc": desc,
                }
            )
            if end == len(text):
                break
            start = max(0, end - overlap)
    return documents


def build_faiss_index_from_documents(
    documents: list[dict[str, str]],
    index_dir: Path,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
) -> dict[str, Path]:
    if not documents:
        raise RuntimeError("No documents provided for FAISS index.")
    texts: list[str] = []
    for d in documents:
        title = d.get("title") or ""
        desc = d.get("desc") or ""
        body = d.get("text") or ""
        texts.append("\n".join([title, desc, body]).strip())
    print(f"Encoding {len(texts)} chunks with {model_name} ...")
    model = SentenceTransformer(model_name)
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    faiss_path = index_dir / "faiss.index"
    faiss.write_index(index, str(faiss_path))
    meta = {
        "model_name": model_name,
        "dim": int(dim),
        "documents": documents,
    }
    meta_path = index_dir / "faiss_meta.pkl"
    with meta_path.open("wb") as f:
        pickle.dump(meta, f)
    print(f"FAISS index built ({dim}d) with {len(documents)} chunks -> {faiss_path}")
    return {"index": faiss_path, "meta": meta_path}


def main() -> None:
    base_dir_arg = sys.argv[1] if len(sys.argv) > 1 else None
    base_dir = Path(base_dir_arg) if base_dir_arg else Path(__file__).resolve().parents[1] / "data"
    paths = ensure_dirs(base_dir)
    docs_dir, index_dir = paths["docs"], paths["index"]

    print(f"Preparing data in {base_dir}")
    doc_paths = download_docs(docs_dir)

    pages = fetch_langgraph_pages_from_llms(doc_paths)

    sitemap_urls = fetch_urls_from_sitemap()
    topic_urls = filter_topic_urls(sitemap_urls)
    if topic_urls:
        try:
            aug_pages: list[dict[str, str]] = []
            for u in topic_urls:
                try:
                    r = requests.get(
                        u, timeout=15, headers={"User-Agent": "LangGraphHelperAgent/0.1"}
                    )
                    if r.ok and r.text:
                        text = html_to_visible_text_page(r.text)
                        title = None
                        try:
                            soup = BeautifulSoup(r.text, "html.parser")
                            if soup.title and soup.title.string:
                                title = soup.title.string.strip()
                        except Exception:
                            title = None
                        if text:
                            aug_pages.append({"url": u, "title": title or "", "text": text})
                except Exception:
                    continue
            pages.extend(aug_pages)
        except Exception:
            pass
    page_docs = chunk_pages_to_documents(pages)

    doc_docs: list[dict[str, str]] = []
    for p in doc_paths:
        raw = p.read_text(encoding="utf-8", errors="ignore")
        doc_docs.extend(build_documents_from_sections(p.name, raw))
    combined_docs = doc_docs + page_docs
    faiss_paths = build_faiss_index_from_documents(combined_docs, index_dir)
    manifest = {
        "docs": [str(p) for p in doc_paths],
        "faiss_index": str(faiss_paths["index"]),
        "faiss_meta": str(faiss_paths["meta"]),
        "pages_count": len(pages),
        "chunks_count": len(combined_docs),
    }
    (base_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Done.")


if __name__ == "__main__":
    main()
