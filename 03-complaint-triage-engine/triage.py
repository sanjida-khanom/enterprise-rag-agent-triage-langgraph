"""
Complaint Triage Engine: unstructured customer complaints -> structured records.

    python triage.py --limit 10
    python triage.py --all --out triaged.csv

The business case in one line: a contact centre receives thousands of free-text
complaints a day in Bangla, English and Banglish. Humans read and route them.
Routing is repetitive judgement work -- exactly what an LLM does cheaply -- and
misrouting is expensive because it resets the resolution clock.

The engineering point of this project is STRUCTURED OUTPUT. A chatbot returns
prose, which no downstream system can consume. This returns a validated schema
that goes straight into a database, a dashboard, or a routing queue. Most real
enterprise GenAI value is here, not in chat interfaces.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, Field

from llm_setup import get_llm, to_text

DATA = Path(__file__).parent / "data"


class ComplaintAnalysis(BaseModel):
    """The schema every complaint is forced into.

    Pydantic is doing real work here, not decoration. The field descriptions
    are sent to the model as part of the tool schema, so they function as
    per-field instructions. The Literal types make invalid categories
    impossible rather than merely discouraged -- the model physically cannot
    return 'Billing Issues' when the enum says 'billing'.
    """

    category: Literal[
        "network_coverage", "billing", "data_pack", "customer_service",
        "sim_and_device", "value_added_services", "other",
    ] = Field(description="The single primary issue category.")

    severity: Literal["low", "medium", "high", "critical"] = Field(
        description=(
            "critical = service completely unusable or a suspected fraud/security "
            "issue. high = significant financial impact or repeated unresolved "
            "contact. medium = single clear problem. low = query or minor annoyance."
        )
    )

    sentiment: Literal["angry", "frustrated", "neutral", "satisfied"] = Field(
        description="The customer's emotional tone."
    )

    churn_risk: bool = Field(
        description=(
            "True only if the customer explicitly threatens to leave, mentions a "
            "competitor, or mentions porting their number out."
        )
    )

    routing_team: Literal[
        "network_operations", "billing_team", "retention_team",
        "technical_support", "frontline_care",
    ] = Field(description="The team best able to resolve this.")

    language: Literal["bangla", "english", "mixed"] = Field(
        description="Language the complaint was written in."
    )

    summary_en: str = Field(
        description="One neutral English sentence, under 20 words, stating the problem."
    )

    location_mentioned: Optional[str] = Field(
        default=None, description="Area or district named, if any. Null otherwise."
    )

    requires_callback: bool = Field(
        description="True if the issue cannot be resolved without contacting the customer."
    )


SYSTEM = """You classify inbound telecom customer complaints for a Bangladeshi mobile operator.

Complaints arrive in Bangla, English, or romanised Bangla (Banglish). Read all three fluently.

Be conservative:
- Reserve 'critical' for genuine service-down or fraud situations. Anger alone is not critical.
- Set churn_risk only on an explicit signal (naming a competitor, threatening to port or leave). A frustrated tone is not enough.
- If a location is not clearly named, return null rather than guessing.
Base every field only on what the text actually says."""


def build_extractor():
    llm = get_llm(temperature=0.0)
    try:
        return llm.with_structured_output(ComplaintAnalysis), "native"
    except Exception:
        return llm, "fallback"


def analyse_one(extractor, mode: str, text: str) -> Optional[ComplaintAnalysis]:
    """Extract one complaint, with a JSON-parsing fallback.

    with_structured_output uses the provider's native function-calling and is
    reliable. Some local models via Ollama don't support it, so the fallback
    prompts for raw JSON and validates with Pydantic anyway -- the schema is
    enforced either way, only the mechanism changes.
    """
    if mode == "native":
        return extractor.invoke(f"{SYSTEM}\n\nComplaint:\n{text}")

    schema = json.dumps(ComplaintAnalysis.model_json_schema()["properties"], indent=1)
    raw = to_text(extractor.invoke(
        f"{SYSTEM}\n\nReturn ONLY a JSON object matching this schema, no markdown "
        f"fences and no commentary:\n{schema}\n\nComplaint:\n{text}"
    ))
    cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```")
    return ComplaintAnalysis(**json.loads(cleaned))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=10)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="triaged.csv")
    ap.add_argument("--sleep", type=float, default=0.5, help="pause between calls; free tiers rate-limit")
    args = ap.parse_args()

    df = pd.read_csv(DATA / "complaints.csv")
    if not args.all:
        df = df.head(args.limit)

    extractor, mode = build_extractor()
    print(f"Structured output mode: {mode}")
    print(f"Processing {len(df)} complaints...\n")

    rows, failed = [], 0
    for i, rec in enumerate(df.itertuples(), start=1):
        try:
            a = analyse_one(extractor, mode, rec.complaint_text)
            rows.append({
                "complaint_id": rec.complaint_id,
                "received_at": rec.received_at,
                "channel": rec.channel,
                **a.model_dump(),
            })
            flag = "  <-- CHURN RISK" if a.churn_risk else ""
            print(f"[{i:>3}/{len(df)}] {rec.complaint_id}  {a.category:<20} "
                  f"{a.severity:<8} -> {a.routing_team}{flag}")
        except Exception as exc:
            failed += 1
            print(f"[{i:>3}/{len(df)}] {rec.complaint_id}  FAILED: {str(exc)[:60]}")
        time.sleep(args.sleep)

    if not rows:
        raise SystemExit("Nothing processed successfully. Check your API key and provider.")

    out = pd.DataFrame(rows)
    out.to_csv(args.out, index=False)
    print(f"\nWrote {len(out)} records to {args.out} ({failed} failed)")
    print("Next: python report.py")


if __name__ == "__main__":
    main()
