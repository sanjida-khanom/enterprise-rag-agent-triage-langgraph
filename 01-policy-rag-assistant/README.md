# Project 1 — Internal Policy Knowledge Assistant

A grounded question-answering system over internal enterprise documents, with hybrid retrieval, inline citations, and a measured evaluation harness.

**This is your flagship project.** If you only get one working, make it this one. RAG over internal documents is the single most common enterprise GenAI deployment, and it maps directly onto the job description's "business process improvement."

---

## What it does

Four internal documents (HR leave policy, finance expense policy, a network escalation runbook, a customer-care tariff FAQ) are chunked, embedded, and indexed. A question retrieves the most relevant passages and the model answers **only** from those passages, citing them. If the answer isn't in the documents, it refuses.

## Architecture

```
data/*.md ──> ingest.py ──────────────────────────> chroma_db/
              load → chunk → embed → persist         (vectors + metadata)

question ──> rag.py
             ├── vector search (semantic)  ─┐
             │                              ├── RRF fusion ──> top k
             └── BM25 search (keyword)     ─┘                    │
                                                                  v
                                            grounded prompt ──> LLM ──> answer + [1][2]

golden_set.json ──> eval.py ──> hit_rate · MRR · keyword recall · refusal accuracy
```

---

## Setup

```bash
cd 01-policy-rag-assistant
python -m venv venv
source venv/bin/activate           # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then edit .env and add your GOOGLE_API_KEY
```

Get a free Gemini key at `aistudio.google.com/apikey` — no card required, works from Bangladesh.

**Verify before anything else:**
```bash
python llm_setup.py
```
Expected: `LLM OK -> ready` and `Embeddings OK -> 384 dimensions`. The first run downloads the embedding model (~120MB), so **do this at home, not at the venue.**

## Run

```bash
python ingest.py --rebuild        # build the index (do this before assessment day)

python rag.py "How many days of annual leave do I get?"
python rag.py "What does alarm code NE-5502 mean?" --show-chunks
python rag.py "ছুটি কত দিন পাব বছরে?"

python eval.py --compare          # the money shot
streamlit run app.py              # the visual demo
```

---

## The 4-minute live demo script

Rehearse this until it's smooth. Order matters — each step sets up the next.

**1. Ordinary question (30s).** `streamlit run app.py`, ask *"How many days of annual leave do I get?"* Expand the Sources panel.
> "It answered 20 days, and cited passage 1. I can see the exact chunk it used. In a policy context that traceability isn't a nice-to-have — if a supervisor tells a customer the wrong fee, someone has to be able to check where that came from."

**2. The refusal (30s).** Ask *"What is the work from home policy?"*
> "There's no remote work document, so it declines rather than inventing something plausible. This was the first thing I built, and the hardest to get right — the default behaviour of these models is to be helpful, and helpful means making something up."

**3. Hybrid vs vector — the technical centrepiece (90s).** Switch the sidebar to `vector`, ask *"What does alarm code NE-5502 mean?"* Then switch to `hybrid` and ask again.
> "Pure semantic search struggles with alphanumeric identifiers — 'NE-5502' and 'NE-3310' embed to almost the same vector because they're structurally identical strings with no semantic content. BM25 matches the literal token. I run both and fuse the ranked lists with Reciprocal Rank Fusion. Telecom documents are full of exactly this: plan codes, alarm codes, MSISDNs. It's the reason I didn't stop at vector search."

*(If the difference doesn't show live, don't fake it — say "on my eval set the gap shows up on the code-lookup questions, let me show you" and run `eval.py --compare`.)*

**4. Bangla (45s).** Ask *"amar package change korte koto taka lagbe?"*
> "That's romanised Bangla against English source documents. It works because I chose a multilingual embedding model. The default in every tutorial is all-MiniLM-L6-v2, which is English-only — it doesn't error on Bangla, it just returns quietly wrong results. For Robi's actual customer base that's not a detail."

**5. Evaluation (45s).** `python eval.py --compare`
> "21 test cases, three of which are deliberately unanswerable. I separate retrieval metrics from generation metrics, because when an answer is wrong you need to know which half broke. Refusal accuracy is the one I'd report to a business owner — it's the hallucination rate."

---

## Questions you will be asked, and how to answer

### On chunking

**"Why chunk size 1000?"**
> "It's a starting point, not a principle. Large enough to hold a complete policy clause, small enough that the embedding represents one idea rather than an average of several. I validated it against the golden set rather than trusting the default. For these documents I split on markdown headings first, so chunks tend to align with actual policy sections — respecting document structure matters more than the exact character count."

**"What happens if you chunk too small? Too large?"**
> "Too small and you lose the context that makes a passage meaningful — a chunk saying 'must be approved by the divisional head' is useless without knowing what 'it' is. Too large and the embedding blurs: one vector averaging five topics matches everything weakly and nothing strongly, plus you burn context window on irrelevant text. I tested both ends — you can see it degrade."

**"How would you handle tables or scanned PDFs?"**
> "Differently, and this is a real limitation of what I've built. A table split across chunks is worse than useless because it produces confident nonsense. I'd extract tables separately and keep each one whole as a single chunk, often with an LLM-generated summary of the table attached for retrieval. For scanned documents you need OCR first, and OCR quality becomes the ceiling on the whole system — garbage in, confidently-cited garbage out."

### On retrieval

**"Explain hybrid search and why you bothered."** — See demo step 3. Then add:
> "The cost is a second index and a fusion step. I'd justify that with the eval numbers rather than on principle."

**"What is Reciprocal Rank Fusion and why not just add the scores?"**
> "Because BM25 scores and cosine similarities aren't on the same scale — one might range 0 to 30, the other 0 to 1. Add them and BM25 always wins, regardless of quality. RRF ignores the scores entirely and uses only the rank position: each document scores 1/(60+rank) summed across the lists it appears in. A document ranked reasonably by both retrievers beats one ranked first by only one. That consensus behaviour is exactly what you want from fusion."

**"Why 60?"**
> "It's the constant from the original RRF paper. It flattens the difference between rank 1 and rank 3, so agreement between retrievers matters more than any single retriever's confidence. A small k would make rank 1 dominate and the fusion would collapse back into 'whoever was most confident wins'."

**"How would you improve retrieval further?"**
> "Three things, in order of expected value: a cross-encoder reranker — retrieve 20 cheaply, rescore with a model that reads the query and passage together, keep 4. That's usually the largest single gain. Then query rewriting, so follow-up questions like 'and for postpaid?' get expanded into standalone queries. Then metadata filtering, so an HR question doesn't retrieve network runbooks at all."

### On the model and generation

**"How do you stop it hallucinating?"**
> "Layered, because no single measure is sufficient. The prompt constrains it to the context and gives an explicit refusal string. Temperature is zero. Citations make claims checkable. And I measure it — the refusal accuracy metric exists precisely so hallucination is a number I can report, not a vibe. In production I'd add a groundedness check: a second cheap call asking whether the answer is actually supported by the passages, routing failures to a human."

**"Why not fine-tune instead?"**
> "RAG for knowledge, fine-tuning for behaviour. These policies change — an expense limit gets revised and I re-index one file in seconds. Fine-tuning would need a retraining cycle, couldn't produce citations, and would bake internal policy into model weights, which is a governance problem. Fine-tuning would make sense if I needed a consistent output format or tone the base model couldn't follow, and the two aren't mutually exclusive."

**"What's your context window strategy as documents grow?"**
> "Retrieval is the strategy — that's the whole point. k stays at 4 whether there are 4 documents or 40,000; what changes is the difficulty of finding the right 4, which is why retrieval quality is where I'd invest. I'd resist the 'just put everything in a long context window' answer: it's expensive per query, slower, and accuracy degrades in the middle of long contexts."

### On production and governance

**"How would you deploy this at Robi?"**
> "Start narrow. One department, documents that are already internal-only, read-only access, a pilot group who know it's a pilot. Measure ticket deflection and, more importantly, correction rate — how often does the user have to fix the answer. FastAPI behind the company SSO, permissions enforced at the retrieval layer so a user can only retrieve documents they're already entitled to read. That last point is the one that gets missed: if access control is only in the UI, the vector store is a data leak."

**"What about customer PII?"**
> "The documents here are policy, so no PII — that's deliberate for a first deployment. Once customer data enters the picture: redact before the model call, keep embeddings local as I've done here so document text never leaves the network, prefer a self-hosted open model for anything sensitive, log every query for audit, and check data-residency obligations before choosing a provider. For a licensed telecom operator that last one is a regulatory question, not an engineering preference."

**"What would this cost to run?"**
> "Embeddings are free — they run locally, which is why I made that choice. Per query it's the retrieved context plus the answer, so roughly a few thousand tokens. On a cheap model that's a fraction of a taka per question, and caching common questions cuts it further. The honest cost driver at scale isn't inference, it's keeping the document corpus current and correct."

### The ones designed to test honesty

**"What doesn't work well?"**
Have a real answer ready. Never say "nothing."
> "Three things. Multi-document synthesis — if the answer requires combining the leave policy and the expense policy, retrieval tends to fetch four chunks from whichever document matches more strongly. Tables, as I mentioned. And my keyword-recall metric is crude; it checks for expected strings, which rewards the right words rather than the right meaning. A proper implementation would use an LLM-as-judge with a rubric, and I'd want human spot-checks on top because judges have their own biases."

**"How long did this take you and what did you find hardest?"**
Be truthful. Something like:
> "About a day and a half. The hardest part wasn't the pipeline — that's well-documented. It was getting refusal to work reliably. My first version answered the work-from-home question with a confident invented policy, because these models are trained to be helpful and helpful defaults to answering. Getting it to say 'I don't know' took explicit instruction, a fixed refusal string, and a test that measures it."

---

## Known limitations

Say these before you're asked. It reads as engineering maturity, not weakness.

- No reranking layer — the next thing I'd add
- Table handling is naive
- Keyword-based answer scoring is a proxy for correctness, not a measure of it
- BM25 index lives in memory; needs OpenSearch at real scale
- No conversation memory — each question is independent, so follow-ups like "and for postpaid?" fail
- No access control on retrieval

## Files

| File | What it is |
|---|---|
| `llm_setup.py` | Provider abstraction — swap Gemini/Groq/Ollama by config |
| `ingest.py` | Load, chunk, embed, persist |
| `rag.py` | Hybrid retrieval + RRF + grounded generation |
| `eval.py` | Metrics against the golden set |
| `golden_set.json` | 21 cases incl. 3 refusal tests, 2 Bangla |
| `app.py` | Streamlit demo |
