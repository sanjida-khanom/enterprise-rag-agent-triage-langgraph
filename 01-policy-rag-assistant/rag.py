"""
The RAG engine: hybrid retrieval + grounded generation with citations.

Usage:
    python rag.py "How many days of annual leave do I get?"
    python rag.py "How many days of annual leave?" --mode vector
    python rag.py "amar package change korte chai" --show-chunks
"""

import argparse
import re
from dataclasses import dataclass, field
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.prompts import ChatPromptTemplate

from llm_setup import get_llm, get_embeddings, to_text

DB_DIR = Path("chroma_db")

ANSWER_PROMPT = ChatPromptTemplate.from_template(
    """You are an internal knowledge assistant for a telecom operator.

Answer the question using ONLY the numbered context passages below.

Rules:
- If the context does not contain the answer, reply exactly:
  "I don't have that information in the available documents."
  Do not guess, and do not use general knowledge to fill gaps.
- Cite the passage numbers you used, like [1] or [2][3], inline.
- If the passages disagree with each other, say so explicitly rather than
  silently picking one.
- Answer in the same language the question was asked in.
- Be concise. Two or three sentences unless the question needs more.

Context passages:
{context}

Question: {question}

Answer:"""
)


@dataclass
class RagResult:
    question: str
    answer: str
    chunks: list = field(default_factory=list)
    scores: dict = field(default_factory=dict)


class PolicyRAG:
    """Hybrid retriever + grounded generator.

    The interesting part is _reciprocal_rank_fusion. Dense vector search
    understands meaning but is weak on exact tokens -- plan codes, error codes,
    MSISDNs, clause numbers. BM25 is the opposite: exact on tokens, blind to
    paraphrase. Telecom documents are full of both, so we run each retriever
    independently and fuse the two ranked lists.
    """

    def __init__(self, mode: str = "hybrid", k: int = 4, candidates: int = 20):
        self.mode = mode
        self.k = k
        self.candidates = candidates

        if not DB_DIR.exists():
            raise SystemExit("No index found. Run: python ingest.py")

        self.store = Chroma(
            persist_directory=str(DB_DIR), embedding_function=get_embeddings()
        )
        self.llm = get_llm(temperature=0.0)

        # BM25 needs the raw corpus in memory. Fine at this scale; at millions
        # of documents you would use OpenSearch/Elasticsearch instead.
        raw = self.store.get()
        from langchain_core.documents import Document

        self.corpus = [
            Document(page_content=t, metadata=m)
            for t, m in zip(raw["documents"], raw["metadatas"])
        ]
        self.bm25 = BM25Retriever.from_documents(self.corpus)
        self.bm25.k = self.candidates

    @staticmethod
    def _key(doc) -> str:
        return f"{doc.metadata.get('source')}::{doc.metadata.get('page')}::{doc.page_content[:80]}"

    def _reciprocal_rank_fusion(self, ranked_lists, k_rrf: int = 60):
        """Combine several ranked lists into one.

        RRF scores a document as sum(1 / (k + rank)) across the lists it
        appears in. It needs no score normalisation, which matters because
        BM25 scores and cosine similarities are not on comparable scales --
        naively adding them lets whichever has the larger numeric range win.
        A document ranked decently by BOTH retrievers beats one ranked first
        by only one, which is exactly the behaviour we want.
        """
        scores, lookup = {}, {}
        for ranked in ranked_lists:
            for rank, doc in enumerate(ranked):
                key = self._key(doc)
                lookup[key] = doc
                scores[key] = scores.get(key, 0.0) + 1.0 / (k_rrf + rank + 1)
        ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        return [(lookup[key], score) for key, score in ordered]

    def retrieve(self, question: str):
        if self.mode == "vector":
            docs = self.store.similarity_search(question, k=self.k)
            return [(d, 0.0) for d in docs]
        if self.mode == "bm25":
            return [(d, 0.0) for d in self.bm25.invoke(question)[: self.k]]

        vector_hits = self.store.similarity_search(question, k=self.candidates)
        keyword_hits = self.bm25.invoke(question)
        fused = self._reciprocal_rank_fusion([vector_hits, keyword_hits])
        return fused[: self.k]

    @staticmethod
    def _format(scored) -> str:
        blocks = []
        for i, (doc, _) in enumerate(scored, start=1):
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", 0)
            blocks.append(f"[{i}] (source: {src}, page {page})\n{doc.page_content}")
        return "\n\n".join(blocks)

    def ask(self, question: str) -> RagResult:
        scored = self.retrieve(question)
        if not scored:
            return RagResult(question, "I don't have that information in the available documents.")

        context = self._format(scored)
        answer = to_text(self.llm.invoke(
            ANSWER_PROMPT.format(context=context, question=question)
        ))
        return RagResult(
            question=question,
            answer=answer,
            chunks=[d for d, _ in scored],
            scores={self._key(d): s for d, s in scored},
        )

    def cited_indices(self, answer: str):
        """Which passages did the model actually claim to use?"""
        return sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--mode", choices=["hybrid", "vector", "bm25"], default="hybrid")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--show-chunks", action="store_true")
    args = ap.parse_args()

    rag = PolicyRAG(mode=args.mode, k=args.k)
    result = rag.ask(args.question)

    print(f"\nQ: {result.question}\n")
    print(f"A: {result.answer}\n")

    if args.show_chunks:
        print("-" * 70)
        print("RETRIEVED PASSAGES")
        for i, doc in enumerate(result.chunks, start=1):
            src = doc.metadata.get("source")
            page = doc.metadata.get("page")
            preview = doc.page_content[:220].replace("\n", " ")
            print(f"\n[{i}] {src} p{page}\n    {preview}...")
        print()
        print(f"Passages cited by the model: {rag.cited_indices(result.answer)}")


if __name__ == "__main__":
    main()
