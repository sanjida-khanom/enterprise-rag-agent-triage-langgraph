# Project 3 — Complaint Triage & Insight Engine

Turns 30 free-text customer complaints in Bangla, English and Banglish into a validated structured dataset, then into a management brief.

**This is your Corporate Strategy project.** Projects 1 and 2 prove you can build. This one proves you can find where the money is — which is what the role is actually for.

---

## Why this one matters most for this role

Most GenAI demos are chat interfaces. Most enterprise GenAI *value* is not, because chat output is prose and no downstream system consumes prose. The valuable pattern is **unstructured input → validated structured output → existing business systems**.

Once complaints are structured, you get two wins from one extraction pass:

- **Operational** — automatic routing to the right team. Misrouting resets the resolution clock, so this is measurable in average handling time.
- **Strategic** — a dataset that answers questions nobody could previously afford to ask. Which areas generate the most critical network complaints? What share of contacts carry an explicit churn signal? That analytical layer is the strategic prize; the operational saving pays for the pipeline.

State this framing out loud in the interview. It's the difference between "I built a classifier" and "I found a process worth automating and quantified the second-order value."

---

## Architecture

```
data/complaints.csv          30 real-shaped complaints, mixed language
        │
        v
   triage.py                 LLM + Pydantic schema → validated records
        │                    (native function calling, JSON fallback)
        v
   triaged.csv               9 structured fields per complaint
        │
        v
   report.py                 pandas computes every number
        │                    LLM writes prose over those numbers only
        v
   dashboard + management brief
```

## The schema is the project

```python
category         network_coverage | billing | data_pack | customer_service | ...
severity         low | medium | high | critical
sentiment        angry | frustrated | neutral | satisfied
churn_risk       bool   ← only on explicit signal
routing_team     network_operations | billing_team | retention_team | ...
language         bangla | english | mixed
summary_en       one neutral English sentence
location_mentioned  str | None
requires_callback   bool
```

`Literal` types make invalid categories *impossible*, not merely discouraged — the model physically cannot return `"Billing Issues"` when the enum says `billing`. Field descriptions are sent to the model as part of the schema, so they function as per-field instructions.

---

## Setup and run

```bash
cd 03-complaint-triage-engine
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # add GOOGLE_API_KEY
python llm_setup.py         # verify

python triage.py --limit 5        # quick check
python triage.py --all            # full run, writes triaged.csv (~30-60s)
python report.py                  # dashboard + brief
```

Free tiers rate-limit; `--sleep 0.5` is the default pause between calls. Raise it if you see 429 errors.

**Run the full pipeline the night before and keep `triaged.csv`.** If the wifi dies you can still run `report.py` on the saved file.

---

## The 3-minute demo script

**1. Show the input first (30s).** Open `data/complaints.csv`.
> "Thirty complaints as they actually arrive — Bangla script, English, and romanised Banglish, often mixed inside one message. Currently a human reads each one and routes it."

**2. Run `triage.py --limit 5` live (45s).** Let them watch classifications stream.
> "Each one becomes a validated record. The important part isn't the classification, it's that the output is a schema, not prose — this goes straight into a routing queue or a dashboard. Pydantic enforces it, so the enum values are guaranteed valid rather than hopefully valid."

**3. Point at a churn-risk row (30s).**
> "This one names a competitor. I deliberately set churn_risk to fire only on an explicit signal — naming a competitor or threatening to port — not on anger, because angry customers are common and leaving customers are not. If you conflate them you hand retention a queue they'll stop trusting."

**4. `python report.py` (60s).**
> "Same extraction pass, second use. Pandas computes every number; the model only writes prose over figures it was handed. I never ask a language model to do arithmetic over a dataset — it produces confident, plausible, wrong totals. That division of labour is deliberate."

**5. Close on the business case (30s).**
> "The operational case is routing accuracy. The strategic case is that this is a dataset Robi doesn't currently have — complaint volume by area and severity, updated daily, which is a network-investment signal as much as a customer-care one."

---

## Questions you will be asked

### Technical

**"How do you guarantee valid JSON?"**
> "`with_structured_output` uses the provider's native function calling, so the model emits a tool call conforming to the schema rather than free text I have to parse. Pydantic validates on top. There's a fallback path that prompts for raw JSON and validates identically, for local models that don't support tool calling — the schema is enforced either way, only the mechanism changes."

**"How would you evaluate the classifier?"**
> "Have humans label a sample — 200 complaints or so — and measure per-class precision and recall against it. Aggregate accuracy hides the failure that matters: if `critical` is only 3% of volume, a model that never predicts critical still scores 97%. I'd also track inter-annotator agreement first, because if two humans only agree 70% of the time on severity, that's the realistic ceiling and it means the category definitions need work, not the model."

**"Is an LLM the right tool here? This is just classification."**
Excellent question, and the honest answer is strong:
> "For a fixed set of categories with thousands of labelled examples, a fine-tuned small model would be cheaper and faster per item. The LLM wins on three things: it works with zero labelled data on day one, it handles Bangla and Banglish without a custom pipeline, and it does open-ended extraction — the summary and location fields aren't classification at all.
>
> What I'd actually do is use the LLM to bootstrap: run it for a few weeks, have agents correct the output, and you now have a labelled dataset. Then distil the high-volume categories into a small fast classifier and keep the LLM for the long tail and the extraction fields. That's cheaper than either approach alone."

**"What about cost at Robi's volume?"**
> "Each complaint is a few hundred tokens in and maybe a hundred out, so on a cheap model it's a fraction of a taka per item — meaningful at scale but not the binding constraint. Batch the processing rather than doing it per-message in real time, use the smallest model that holds accuracy, and once you've distilled to a local classifier the marginal cost approaches zero. I'd model this against the fully-loaded cost of the agent minutes it saves before committing."

**"What if the model misclassifies something critical as low?"**
> "That's the failure mode that matters, and it's why I'd tune the threshold asymmetrically. Missing a genuine critical is far more expensive than over-flagging one, so I'd deliberately bias toward recall on the critical class and accept more false positives. I'd also never let automated routing be the only path — a keyword safety net for terms like 'fraud' or 'no service' that force a human review regardless of what the model said."

### Strategy — this is where you win the role

**"How would you pitch this to a business owner?"**
> "Not with the technology. I'd start with their number: what's the current misroute rate, and what does a misroute cost in resolution time? Then propose a two-week shadow pilot — the model classifies in parallel with humans, nobody's routing changes, and we compare. If agreement is high, we automate the confident cases and leave the rest to humans. That derisks it completely, and it gives them a number rather than a promise."

**"What could go wrong in deployment?"**
> "Three things I'd watch. Drift — a new product launch creates complaint types the schema doesn't have, and accuracy silently degrades, so I'd monitor the 'other' category as an early warning. Automation bias — once agents trust the routing they stop sanity-checking it, so errors propagate further before anyone notices. And the political one: if this is framed as headcount reduction, the agents whose corrections you need for the training data have every reason not to give them to you. Framing it as capacity, not replacement, isn't just diplomacy — the data pipeline depends on it."

**"Who owns this in production?"**
> "Contact centre operations owns the outcome, because it's their metric. A central team owns the model and the pipeline. That split matters — if the AI team owns the KPI, the business treats it as someone else's project and never adopts it."

**"What's the second-order value?"**
> "Complaint volume by area and severity is a network investment signal. If Sylhet generates disproportionate critical network complaints week after week, that's evidence for capex prioritisation that currently lives as anecdote in a call centre. Same extraction, entirely different consumer. That's the argument for doing the structuring properly rather than just building a router."

### Honest limits

**"What's weak here?"**
> "I have no ground truth. I've eyeballed the outputs and they look right, but 'looks right' isn't a metric — the first thing I'd do with real data is get a labelled sample. Thirty complaints is a demo, not a distribution. And I'd want a Bangla-speaking care agent to review the Bangla classifications specifically, because I can read them but I'm not the domain expert on whether the severity calls match how the care team actually triages."

---

## Files

| File | What it is |
|---|---|
| `triage.py` | Pydantic schema + extraction with fallback |
| `report.py` | pandas analytics + LLM management brief |
| `data/complaints.csv` | 30 mixed-language complaints |
| `llm_setup.py` | Provider abstraction (shared) |
