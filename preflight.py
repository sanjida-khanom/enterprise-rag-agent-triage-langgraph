"""
Pre-flight check. Run this the night before, from the repo root.

    python preflight.py

Checks structure and data without needing an API key, then tells you exactly
what to verify manually. The point is to find problems on Saturday night at
home, not at 11am on Sunday in front of a panel.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OK, WARN, FAIL = "  [ok]  ", "  [!!]  ", "  [XX]  "

issues = []


def check(label: str, condition: bool, hint: str = "", fatal: bool = True):
    if condition:
        print(f"{OK}{label}")
    else:
        print(f"{FAIL if fatal else WARN}{label}" + (f"  -> {hint}" if hint else ""))
        if fatal:
            issues.append(label)


print("\n" + "=" * 64)
print("  PRE-FLIGHT CHECK")
print("=" * 64)

projects = {
    "01-policy-rag-assistant": ["llm_setup.py", "ingest.py", "rag.py", "eval.py",
                                "app.py", "golden_set.json", "requirements.txt"],
    "02-support-agent-langgraph": ["llm_setup.py", "agent.py", "tools.py",
                                   "requirements.txt"],
    "03-complaint-triage-engine": ["llm_setup.py", "triage.py", "report.py",
                                   "requirements.txt"],
}

print("\nFILES")
for proj, files in projects.items():
    for f in files:
        p = ROOT / proj / f
        check(f"{proj}/{f}", p.exists())

print("\nDATA")
docs = list((ROOT / "01-policy-rag-assistant" / "data").glob("*.md"))
check(f"project 1 source documents ({len(docs)} found)", len(docs) >= 4)

for f in ["subscribers.csv", "plans.csv", "policy_kb.json"]:
    check(f"project 2 data/{f}", (ROOT / "02-support-agent-langgraph" / "data" / f).exists())

check("project 3 data/complaints.csv",
      (ROOT / "03-complaint-triage-engine" / "data" / "complaints.csv").exists())

print("\nGOLDEN SET")
try:
    golden = json.loads((ROOT / "01-policy-rag-assistant" / "golden_set.json").read_text(encoding="utf-8"))
    refusals = sum(1 for c in golden if c["expected_source"] is None)
    bangla = sum(1 for c in golden if any(ord(ch) > 2400 for ch in c["question"])
                 or "korte" in c["question"].lower())
    check(f"{len(golden)} test cases", len(golden) >= 15)
    check(f"{refusals} refusal tests", refusals >= 2)
    check(f"{bangla} Bangla / Banglish cases", bangla >= 2, fatal=False)
except Exception as exc:
    check("golden_set.json parses", False, str(exc))

print("\nBUILT ARTEFACTS  (these must exist BEFORE the assessment)")
check("project 1 chroma_db/ index built",
      (ROOT / "01-policy-rag-assistant" / "chroma_db").exists(),
      "run: cd 01-policy-rag-assistant && python ingest.py --rebuild", fatal=False)
check("project 3 triaged.csv cached",
      (ROOT / "03-complaint-triage-engine" / "triaged.csv").exists(),
      "run: cd 03-complaint-triage-engine && python triage.py --all", fatal=False)

print("\nSECURITY")
env_files = list(ROOT.rglob(".env"))
check(f"no .env committed ({len(env_files)} found on disk, must be gitignored)",
      (ROOT / ".gitignore").exists(), "create .gitignore with .env in it")

print("\n" + "=" * 64)
if issues:
    print(f"  {len(issues)} problem(s) to fix:")
    for i in issues:
        print(f"    - {i}")
else:
    print("  Structure OK.")

print("""
  STILL VERIFY MANUALLY (needs your API key):

    cd 01-policy-rag-assistant && python llm_setup.py
    python ingest.py --rebuild && python eval.py --compare
    streamlit run app.py

    cd ../02-support-agent-langgraph && python agent.py --scenario 2 --trace
    cd ../03-complaint-triage-engine && python triage.py --limit 3

  THEN test the offline path:
    set LLM_PROVIDER=ollama, turn wifi OFF, rerun the above.
""")
print("=" * 64 + "\n")

sys.exit(1 if issues else 0)
