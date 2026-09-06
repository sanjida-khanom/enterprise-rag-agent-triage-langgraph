"""
Ingestion: documents -> chunks -> embeddings -> vector store.

Run once (or whenever documents change):
    python ingest.py

Design decisions worth defending in an interview are marked WHY.
"""

import argparse
import shutil
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from llm_setup import get_embeddings

DATA_DIR = Path("data")
DB_DIR = Path("chroma_db")


def load_documents(data_dir: Path):
    """Load every PDF and text file in data/, tagging each with metadata.

    WHY metadata: without a source filename and page number you cannot cite,
    and without citations nobody in a regulated business will trust the output.
    Metadata also enables filtered retrieval later (e.g. only HR documents,
    only documents effective after a given date).
    """
    docs = []
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        elif path.suffix.lower() in {".txt", ".md"}:
            loaded = TextLoader(str(path), encoding="utf-8").load()
        else:
            continue

        for d in loaded:
            d.metadata["source"] = path.name
            d.metadata.setdefault("page", 0)
            # Crude department tag from filename prefix, e.g. "hr_leave.md".
            d.metadata["department"] = path.stem.split("_")[0]
        docs.append(loaded)
        print(f"  loaded {path.name} ({len(loaded)} page(s))")

    return [d for group in docs for d in group]


def chunk_documents(docs, chunk_size: int, chunk_overlap: int):
    """Split documents into retrievable units.

    WHY RecursiveCharacterTextSplitter: it tries paragraph breaks first, then
    sentences, then words. A naive fixed-width split cuts sentences in half and
    produces chunks that are individually meaningless.

    WHY overlap: a fact whose subject is in one chunk and predicate in the next
    is unretrievable. Overlap buys insurance against unlucky boundaries.

    WHY ~1000 chars: large enough to hold a complete clause of a policy,
    small enough that the embedding represents one idea rather than an average
    of five. This is not a universal constant -- it is measured against the
    golden set in eval.py, and dense tables want different treatment entirely.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    return splitter.split_documents(docs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk-size", type=int, default=1000)
    ap.add_argument("--chunk-overlap", type=int, default=150)
    ap.add_argument("--rebuild", action="store_true", help="wipe the existing index")
    args = ap.parse_args()

    if args.rebuild and DB_DIR.exists():
        shutil.rmtree(DB_DIR)
        print("Removed existing index.")

    print("Loading documents...")
    docs = load_documents(DATA_DIR)
    if not docs:
        raise SystemExit("No documents found in data/. Add PDFs or .md files.")

    print(f"\nChunking (size={args.chunk_size}, overlap={args.chunk_overlap})...")
    chunks = chunk_documents(docs, args.chunk_size, args.chunk_overlap)
    print(f"  {len(docs)} documents -> {len(chunks)} chunks")
    avg = sum(len(c.page_content) for c in chunks) / len(chunks)
    print(f"  average chunk length: {avg:.0f} characters")

    print("\nEmbedding and indexing (first run downloads the model, ~90MB)...")
    Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        persist_directory=str(DB_DIR),
    )
    print(f"\nDone. Index written to {DB_DIR}/")
    print("Next: python rag.py \"your question here\"   or   streamlit run app.py")


if __name__ == "__main__":
    main()
