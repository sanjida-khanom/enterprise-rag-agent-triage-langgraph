"""
Tools available to the agent.

Tool design is most of agent engineering. Three rules applied here:

1. The docstring IS the prompt. The model chooses tools by reading these
   descriptions, so they state when to use the tool, not just what it does.
2. Tools fail loudly and in natural language. Returning "No subscriber found
   with that number" lets the model recover; raising an exception kills the run.
3. Write actions are separated from read actions and gated. create_ticket is
   the only tool with side effects, and it is the one a production system would
   put behind human confirmation.
"""

import json
from datetime import datetime
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

DATA = Path(__file__).parent / "data"
SUBSCRIBERS = pd.read_csv(DATA / "subscribers.csv", dtype={"msisdn": str})
PLANS = pd.read_csv(DATA / "plans.csv")
KB = json.loads((DATA / "policy_kb.json").read_text(encoding="utf-8"))

TICKETS: list[dict] = []  # in-memory stand-in for a real ticketing system


@tool
def lookup_subscriber(msisdn: str) -> str:
    """Look up a subscriber's account by their mobile number (MSISDN).

    Use this whenever the user mentions a phone number, or asks about "my
    account", "my balance", "my plan", or their eligibility for something.
    Returns account type, current plan, balance, tenure and billing cycle day.
    """
    msisdn = "".join(ch for ch in msisdn if ch.isdigit())
    row = SUBSCRIBERS[SUBSCRIBERS.msisdn.str.endswith(msisdn[-10:])]
    if row.empty:
        return f"No subscriber found with number ending {msisdn[-4:]}. Ask the user to confirm the number."
    r = row.iloc[0]
    return (
        f"MSISDN: {r.msisdn}\n"
        f"Name: {r['name']}\n"
        f"Account type: {r.account_type}\n"
        f"Current plan: {r.current_plan}\n"
        f"Balance: BDT {r.balance}\n"
        f"Tenure: {r.tenure_days} days on network\n"
        f"Billing cycle day: {r.billing_cycle_day}\n"
        f"Last plan change: {r.last_plan_change_days_ago} days ago\n"
        f"Outstanding dues: BDT {r.outstanding_dues}"
    )


@tool
def search_policy(query: str) -> str:
    """Search internal policy and business-rule documents.

    Use this for any question about rules, eligibility, fees, procedures, or
    what a customer is or is not allowed to do. Always consult this before
    telling a customer that something is permitted or prohibited -- do not rely
    on general knowledge about how telecoms usually work.
    """
    q = set(query.lower().split())
    scored = []
    for entry in KB:
        text = (entry["title"] + " " + entry["content"]).lower()
        overlap = sum(1 for w in q if len(w) > 3 and w in text)
        if overlap:
            scored.append((overlap, entry))
    if not scored:
        return "No matching policy found. Do not invent a rule; tell the user this needs human review."
    scored.sort(key=lambda x: x[0], reverse=True)
    return "\n\n".join(
        f"[{e['doc_id']}] {e['title']}\n{e['content']}" for _, e in scored[:3]
    )


@tool
def list_plans(account_type: str = "any", max_price: int = 100000) -> str:
    """List available tariff plans, optionally filtered.

    Use when the user asks what plans exist, wants a recommendation, or asks
    to compare options. account_type may be 'prepaid', 'postpaid' or 'any'.
    """
    df = PLANS
    if account_type.lower() in {"prepaid", "postpaid"}:
        df = df[df.account_type.str.lower() == account_type.lower()]
    df = df[df.monthly_price <= max_price]
    if df.empty:
        return "No plans match those criteria."
    return df.to_string(index=False)


@tool
def create_ticket(msisdn: str, category: str, summary: str) -> str:
    """Create a support ticket for an issue that cannot be resolved by policy alone.

    Use this ONLY when the user has an unresolved problem needing human action,
    such as a billing dispute, a suspected network fault, or a request that
    policy does not permit but the customer wants escalated. Do not create a
    ticket merely to answer an informational question.
    """
    ticket = {
        "id": f"TKT-{len(TICKETS) + 1001}",
        "msisdn": msisdn,
        "category": category,
        "summary": summary,
        "created": datetime.now().isoformat(timespec="seconds"),
        "status": "OPEN",
    }
    TICKETS.append(ticket)
    return f"Ticket {ticket['id']} created ({category}). Expected first response within 24 hours."


ALL_TOOLS = [lookup_subscriber, search_policy, list_plans, create_ticket]
