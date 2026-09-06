"""
Turn triaged complaints into a management brief.

    python report.py

Why this file exists: triage.py saves the contact centre time. This file is
what makes the project interesting to Corporate Strategy. The same extraction
pass that routes a ticket also produces a structured dataset that answers
questions nobody could previously ask cheaply -- which areas generate the most
critical network complaints, what proportion of contacts carry churn signals,
where volume is concentrating week over week.

The pattern to articulate in an interview: the operational win pays for the
pipeline, and the analytical win is the strategic prize on top of it.

Note the division of labour. Pandas computes every number. The LLM only writes
prose over numbers it was handed. Never ask a language model to do arithmetic
over a dataset -- it will produce plausible, confident, wrong totals.
"""

from pathlib import Path

import pandas as pd

from llm_setup import get_llm, to_text

INPUT = Path("triaged.csv")


def compute_stats(df: pd.DataFrame) -> dict:
    return {
        "total": len(df),
        "by_category": df.category.value_counts().to_dict(),
        "by_severity": df.severity.value_counts().to_dict(),
        "by_team": df.routing_team.value_counts().to_dict(),
        "by_language": df.language.value_counts().to_dict(),
        "by_channel": df.channel.value_counts().to_dict(),
        "churn_risk_count": int(df.churn_risk.sum()),
        "churn_risk_pct": round(100 * df.churn_risk.mean(), 1),
        "critical_count": int((df.severity == "critical").sum()),
        "callback_required": int(df.requires_callback.sum()),
        "top_locations": df.location_mentioned.dropna().value_counts().head(5).to_dict(),
        "angry_pct": round(100 * (df.sentiment == "angry").mean(), 1),
    }


def print_dashboard(s: dict, df: pd.DataFrame):
    print(f"\n{'=' * 66}")
    print(f"  COMPLAINT ANALYTICS  ({s['total']} complaints)")
    print(f"{'=' * 66}\n")

    def bar(label, count, total, width=26):
        filled = int(width * count / total) if total else 0
        return f"  {label:<24}{'#' * filled:<{width}} {count:>4} ({100*count/total:>4.0f}%)"

    print("  BY CATEGORY")
    for k, v in sorted(s["by_category"].items(), key=lambda x: -x[1]):
        print(bar(k, v, s["total"]))

    print("\n  BY SEVERITY")
    for k in ["critical", "high", "medium", "low"]:
        if k in s["by_severity"]:
            print(bar(k, s["by_severity"][k], s["total"]))

    print("\n  ROUTING DESTINATION")
    for k, v in sorted(s["by_team"].items(), key=lambda x: -x[1]):
        print(bar(k, v, s["total"]))

    print("\n  LANGUAGE MIX")
    for k, v in sorted(s["by_language"].items(), key=lambda x: -x[1]):
        print(bar(k, v, s["total"]))

    print(f"\n  KEY INDICATORS")
    print(f"    Churn risk signals      {s['churn_risk_count']:>4}  ({s['churn_risk_pct']}%)")
    print(f"    Critical severity       {s['critical_count']:>4}")
    print(f"    Angry sentiment         {s['angry_pct']:>4}%")
    print(f"    Callback required       {s['callback_required']:>4}")
    if s["top_locations"]:
        locs = ", ".join(f"{k} ({v})" for k, v in s["top_locations"].items())
        print(f"    Locations mentioned     {locs}")

    at_risk = df[df.churn_risk]
    if not at_risk.empty:
        print(f"\n  CHURN-RISK QUEUE (route to retention within 24h)")
        for r in at_risk.head(8).itertuples():
            print(f"    {r.complaint_id}  {r.category:<18} {r.summary_en[:44]}")


def generate_brief(stats: dict) -> str:
    llm = get_llm(temperature=0.3)
    prompt = f"""You are writing for the executive team of a Bangladeshi mobile operator.

Here are this period's complaint analytics:
{stats}

Write a brief with exactly these three sections:

FINDINGS - three observations, each grounded in a specific number above.
RISKS - the two most consequential problems these numbers point to.
ACTIONS - three specific recommendations, each naming an owner team and a
metric that would show it worked.

Rules: use only the numbers given; do not invent data. No preamble. Under 300
words. Write for a busy executive who wants the decision, not the analysis."""
    return to_text(llm.invoke(prompt))


def main():
    if not INPUT.exists():
        raise SystemExit("triaged.csv not found. Run: python triage.py --all")

    df = pd.read_csv(INPUT)
    stats = compute_stats(df)
    print_dashboard(stats, df)

    print(f"\n{'=' * 66}")
    print("  MANAGEMENT BRIEF (LLM-generated over the computed figures)")
    print(f"{'=' * 66}\n")
    print(generate_brief(stats))
    print()


if __name__ == "__main__":
    main()
