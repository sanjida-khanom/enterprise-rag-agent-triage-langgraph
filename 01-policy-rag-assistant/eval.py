"""
Evaluation harness.

This is the file that separates a demo from an engineering project. Without it,
"I improved the chunking" is an opinion. With it, it's a number.

Run:
    python eval.py                    # evaluate hybrid mode
    python eval.py --compare          # hybrid vs vector vs bm25, side by side

Metrics, deliberately split into retrieval vs generation:

  hit_rate        Did the correct source document appear in the retrieved set?
                  Purely a retrieval metric. If this is low, no amount of
                  prompt engineering will save you -- fix chunking/search first.

  mrr             Mean Reciprocal Rank. Not just whether the right chunk was
                  found, but how high it ranked. Rank matters because models
                  attend unevenly across a long context.

  keyword_recall  Did the generated answer contain the facts we expected?
                  A cheap proxy for correctness that needs no second LLM call.

  refusal_ok      For questions deliberately NOT covered by the documents,
                  did the system correctly refuse instead of hallucinating?
                  This is the metric an enterprise actually cares about. A
                  system that is 95% accurate and confidently wrong the other
                  5% of the time is worse than useless in a call centre.
"""

import argparse
import json
import time
from pathlib import Path

from rag import PolicyRAG

GOLDEN = Path("golden_set.json")
REFUSAL_MARKER = "don't have that information"


def load_golden():
    if not GOLDEN.exists():
        raise SystemExit("golden_set.json not found.")
    return json.loads(GOLDEN.read_text(encoding="utf-8"))


def evaluate(mode: str, k: int = 4, verbose: bool = False):
    rag = PolicyRAG(mode=mode, k=k)
    cases = load_golden()

    hits, rr_total, kw_scores = 0, 0.0, []
    answerable, refusals_correct, unanswerable = 0, 0, 0
    latencies = []
    failures = []

    for case in cases:
        start = time.time()
        result = rag.ask(case["question"])
        latencies.append(time.time() - start)
        answer_lower = result.answer.lower()

        if case.get("expected_source") is None:
            # Deliberately out-of-scope question: correct behaviour is refusal.
            unanswerable += 1
            if REFUSAL_MARKER in answer_lower:
                refusals_correct += 1
            else:
                failures.append(("HALLUCINATED", case["question"], result.answer[:120]))
            continue

        answerable += 1

        # --- retrieval metrics -------------------------------------------
        sources = [c.metadata.get("source") for c in result.chunks]
        if case["expected_source"] in sources:
            hits += 1
            rr_total += 1.0 / (sources.index(case["expected_source"]) + 1)
        else:
            failures.append(("RETRIEVAL MISS", case["question"], f"got {sources}"))

        # --- generation metric -------------------------------------------
        expected = [kw.lower() for kw in case.get("expected_keywords", [])]
        if expected:
            found = sum(1 for kw in expected if kw in answer_lower)
            score = found / len(expected)
            kw_scores.append(score)
            if score < 0.5:
                failures.append(("WEAK ANSWER", case["question"], result.answer[:120]))

        if verbose:
            print(f"  Q: {case['question'][:60]}")
            print(f"  A: {result.answer[:100]}\n")

    return {
        "mode": mode,
        "cases": len(cases),
        "hit_rate": hits / answerable if answerable else 0.0,
        "mrr": rr_total / answerable if answerable else 0.0,
        "keyword_recall": sum(kw_scores) / len(kw_scores) if kw_scores else 0.0,
        "refusal_ok": refusals_correct / unanswerable if unanswerable else 1.0,
        "avg_latency_s": sum(latencies) / len(latencies),
        "failures": failures,
    }


def print_report(r):
    print(f"\n{'=' * 62}")
    print(f"  MODE: {r['mode']}   ({r['cases']} test cases)")
    print(f"{'=' * 62}")
    print(f"  Retrieval hit rate   {r['hit_rate']:>6.0%}   correct doc was retrieved")
    print(f"  MRR                  {r['mrr']:>6.2f}   how high it ranked")
    print(f"  Keyword recall       {r['keyword_recall']:>6.0%}   expected facts in answer")
    print(f"  Refusal accuracy     {r['refusal_ok']:>6.0%}   refused when it should")
    print(f"  Avg latency          {r['avg_latency_s']:>6.2f}s")

    if r["failures"]:
        print(f"\n  {len(r['failures'])} case(s) to investigate:")
        for kind, q, detail in r["failures"][:6]:
            print(f"    [{kind}] {q[:52]}")
            print(f"        -> {detail[:90]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="hybrid", choices=["hybrid", "vector", "bm25"])
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.compare:
        results = [evaluate(m, args.k) for m in ("vector", "bm25", "hybrid")]
        for r in results:
            print_report(r)
        print(f"\n{'=' * 62}\n  SUMMARY\n{'=' * 62}")
        print(f"  {'mode':<10}{'hit_rate':>10}{'mrr':>8}{'kw_recall':>12}{'refusal':>10}")
        for r in results:
            print(
                f"  {r['mode']:<10}{r['hit_rate']:>9.0%}{r['mrr']:>8.2f}"
                f"{r['keyword_recall']:>11.0%}{r['refusal_ok']:>10.0%}"
            )
        print("\n  Expect hybrid to beat vector on questions containing exact")
        print("  tokens (plan codes, clause numbers) and to roughly match it")
        print("  elsewhere. That gap is the argument for the extra complexity.\n")
    else:
        print_report(evaluate(args.mode, args.k, args.verbose))


if __name__ == "__main__":
    main()
