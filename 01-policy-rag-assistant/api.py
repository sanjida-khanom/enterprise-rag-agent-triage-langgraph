"""
FastAPI backend for the Policy Knowledge Assistant.

    python -m uvicorn api:app --reload
    then open http://127.0.0.1:8000

Why FastAPI instead of Streamlit:
Streamlit couples the UI to the Python process, so there is no way for another
system to consume the answers. Here the retrieval engine sits behind a JSON API,
which is how it would actually be deployed -- the same endpoint can serve the
web UI, a call-centre desktop tool, or an internal Slack bot. The UI becomes one
client among several rather than the whole product.

The endpoints deliberately return retrieval PROVENANCE, not just the answer:
which retriever found each passage, at what rank, and what it scored after
fusion. That is what lets the interface show a non-technical viewer what the
system is actually doing instead of asking them to trust a black box.
"""

import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag import PolicyRAG, ANSWER_PROMPT
from llm_setup import describe, to_text

STATIC = Path(__file__).parent / "static"
REFUSAL_MARKER = "don't have that information"


# --------------------------------------------------------------------------
# Engine with provenance tracking
# --------------------------------------------------------------------------

class TracedRAG(PolicyRAG):
    """PolicyRAG plus a record of where each passage came from.

    Subclassed rather than edited into rag.py so the command-line tool stays
    lean -- tracing costs nothing here, but it is presentation concern, not a
    retrieval concern.
    """

    def traced_retrieve(self, question: str, mode: str, k: int) -> dict:
        t0 = time.perf_counter()

        vector_hits = self.store.similarity_search(question, k=self.candidates)
        keyword_hits = self.bm25.invoke(question)

        vector_rank = {self._key(d): i for i, d in enumerate(vector_hits)}
        keyword_rank = {self._key(d): i for i, d in enumerate(keyword_hits)}

        if mode == "vector":
            chosen = [(d, 0.0) for d in vector_hits[:k]]
        elif mode == "bm25":
            chosen = [(d, 0.0) for d in keyword_hits[:k]]
        else:
            chosen = self._reciprocal_rank_fusion([vector_hits, keyword_hits])[:k]

        passages = []
        for i, (doc, score) in enumerate(chosen, start=1):
            key = self._key(doc)
            in_vec = key in vector_rank
            in_kw = key in keyword_rank
            found_by = "both" if (in_vec and in_kw) else ("semantic" if in_vec else "keyword")
            passages.append({
                "n": i,
                "source": doc.metadata.get("source", "unknown"),
                "page": doc.metadata.get("page", 0),
                "department": doc.metadata.get("department", "general"),
                "text": doc.page_content,
                "score": round(score, 5),
                "found_by": found_by,
                "vector_rank": vector_rank.get(key),
                "keyword_rank": keyword_rank.get(key),
            })

        fused_keys = set(vector_rank) | set(keyword_rank)
        return {
            "passages": passages,
            "docs": [d for d, _ in chosen],
            "retrieval_ms": round((time.perf_counter() - t0) * 1000),
            "stats": {
                "vector_candidates": len(vector_hits),
                "keyword_candidates": len(keyword_hits),
                "unique_after_fusion": len(fused_keys),
                "overlap": len(set(vector_rank) & set(keyword_rank)),
                "returned": len(passages),
            },
        }

    def answer(self, question: str, mode: str = "hybrid", k: int = 4) -> dict:
        retrieved = self.traced_retrieve(question, mode, k)

        if not retrieved["passages"]:
            return {
                "answer": "I don't have that information in the available documents.",
                "refused": True, "cited": [], "generation_ms": 0, **retrieved,
            }

        context = self._format([(d, 0.0) for d in retrieved["docs"]])

        t0 = time.perf_counter()
        raw = self.llm.invoke(ANSWER_PROMPT.format(context=context, question=question))
        answer = to_text(raw)
        generation_ms = round((time.perf_counter() - t0) * 1000)

        cited = self.cited_indices(answer)
        for p in retrieved["passages"]:
            p["cited"] = p["n"] in cited

        retrieved.pop("docs", None)
        return {
            "answer": answer,
            "refused": REFUSAL_MARKER in answer.lower(),
            "cited": cited,
            "generation_ms": generation_ms,
            "approx_tokens": (len(context) + len(answer)) // 4,
            **retrieved,
        }


# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------

app = FastAPI(title="Policy Knowledge Assistant", version="1.0")
_engine: TracedRAG | None = None


def engine() -> TracedRAG:
    global _engine
    if _engine is None:
        _engine = TracedRAG(mode="hybrid", k=4, candidates=20)
    return _engine


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500)
    mode: str = "hybrid"
    k: int = Field(default=4, ge=1, le=10)


SCENARIOS = [
    {
        "group": "Call centre",
        "question": "Can a postpaid customer change their plan in the middle of the billing cycle?",
        "business": "Agent is on a live call. Today this means searching a PDF while the customer waits.",
        "shows": "Prepaid and postpaid rules differ. A near-miss retrieval gives the wrong answer to a paying customer.",
    },
    {
        "group": "Call centre",
        "question": "What are the requirements to port my number to another operator?",
        "business": "Porting questions are churn moments. A wrong answer loses the subscriber.",
        "shows": "Answer combines three separate conditions from one passage.",
    },
    {
        "group": "Network operations",
        "question": "What does alarm code NE-5502 mean and what causes it?",
        "business": "Field engineer at a site at 2am needs the runbook entry, not a search results page.",
        "shows": "Switch to Semantic only and this degrades. Alarm codes are near-identical as text, so meaning-based search cannot separate them. This is why keyword search runs alongside.",
        "highlight": True,
    },
    {
        "group": "Network operations",
        "question": "A site outage is affecting 3000 subscribers. What severity is it and who do I escalate to?",
        "business": "Escalating late on a major incident costs hours of downtime.",
        "shows": "Two steps: classify by subscriber threshold, then look up that severity's escalation path.",
    },
    {
        "group": "Employee self-service",
        "question": "How many days of annual leave do I get?",
        "business": "The most common HR ticket in most organisations, and the least valuable use of an HR officer's time.",
        "shows": "Baseline lookup with a source citation.",
    },
    {
        "group": "Employee self-service",
        "question": "I forgot to claim an expense from 45 days ago. Can I still get reimbursed?",
        "business": "Finance answers this repeatedly. The rule has three bands and staff get it wrong.",
        "shows": "Applies a conditional rule rather than quoting it.",
    },
    {
        "group": "Bangla and Banglish",
        "question": "amar package change korte koto taka lagbe?",
        "business": "Customers write in romanised Bangla constantly. Support tooling usually cannot read it.",
        "shows": "Romanised Bangla matched against English source documents. An English-only embedding model returns confident nonsense here instead of failing loudly.",
        "highlight": True,
    },
    {
        "group": "Bangla and Banglish",
        "question": "ছুটি কত দিন পাব বছরে?",
        "business": "Same question, Bangla script, answered in Bangla.",
        "shows": "Cross-lingual retrieval, then replies in the language asked.",
    },
    {
        "group": "Safety check",
        "question": "What is the company's policy on remote work and work from home?",
        "business": "There is no such document. An assistant that invents an answer here is worse than no assistant.",
        "shows": "Correct refusal. This is the behaviour that makes the system safe to put in front of staff.",
        "highlight": True,
    },
    {
        "group": "Safety check",
        "question": "What is the penalty for late payment of a postpaid bill?",
        "business": "The hardest refusal: billing documents exist and mention due dates, so retrieval returns confident-looking but insufficient context.",
        "shows": "Refuses despite plausible neighbouring content.",
    },
]


@app.get("/api/health")
def health():
    try:
        eng = engine()
        return {
            "ready": True,
            "config": describe(),
            "documents": len({d.metadata.get("source") for d in eng.corpus}),
            "chunks": len(eng.corpus),
        }
    except SystemExit as exc:
        return {"ready": False, "error": str(exc)}
    except Exception as exc:
        return {"ready": False, "error": f"{type(exc).__name__}: {exc}"}


@app.get("/api/scenarios")
def scenarios():
    return {"scenarios": SCENARIOS}


@app.post("/api/ask")
def ask(req: AskRequest):
    if req.mode not in {"hybrid", "vector", "bm25"}:
        raise HTTPException(400, "mode must be hybrid, vector or bm25")
    try:
        result = engine().answer(req.question, req.mode, req.k)
    except Exception as exc:
        raise HTTPException(500, f"{type(exc).__name__}: {exc}")
    result["question"] = req.question
    result["mode"] = req.mode
    result["total_ms"] = result["retrieval_ms"] + result["generation_ms"]
    return result


@app.post("/api/compare")
def compare(req: AskRequest):
    """Same question through all three retrieval modes.

    The single most useful endpoint for explaining the system: it turns an
    architectural claim into a side-by-side the viewer can judge themselves.
    """
    out = []
    eng = engine()
    for mode in ("vector", "bm25", "hybrid"):
        try:
            r = eng.answer(req.question, mode, req.k)
            out.append({
                "mode": mode,
                "answer": r["answer"],
                "refused": r["refused"],
                "sources": [
                    {"source": p["source"], "found_by": p["found_by"], "n": p["n"]}
                    for p in r["passages"]
                ],
                "total_ms": r["retrieval_ms"] + r["generation_ms"],
            })
        except Exception as exc:
            out.append({"mode": mode, "error": f"{type(exc).__name__}: {exc}"})
    return {"question": req.question, "results": out}


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


@app.get("/")
def index():
    page = STATIC / "index.html"
    if not page.exists():
        raise HTTPException(404, "static/index.html is missing")
    return FileResponse(str(page))
