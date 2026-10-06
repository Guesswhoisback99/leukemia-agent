"""Run the golden set through retrieval (and the chain) and report metrics.

`uv run python -m eval.run --label baseline [--k 5] [--retrieval-only]`
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

from app.chain import build_chain, get_reranker, retrieve
from app.config import get_settings
from app.store import get_vector_store
from eval.metrics import contains_expected, first_hit_rank, hit_at_k, mrr, refusal_correct

GOLDEN_PATH = Path("eval/golden.jsonl")
RESULTS_DIR = Path("eval/results")
TYPES = ["table", "prose", "diagram", "unanswerable"]


def load_golden(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def aggregate(rows: list[dict[str, Any]], k: int) -> dict[str, Any]:
    answerable = [r for r in rows if r["type"] != "unanswerable"]
    ranks = [r["rank"] for r in answerable]
    agg: dict[str, Any] = {
        "n": len(rows),
        f"hit@{k}": hit_at_k(ranks, k) if answerable else None,
        "mrr": mrr(ranks) if answerable else None,
    }
    if rows and "answer" in rows[0]:
        agg["refusal_correct"] = sum(r["refusal_correct"] for r in rows) / len(rows)
        agg["contains_expected"] = (
            sum(r["contains_expected"] for r in answerable) / len(answerable)
            if answerable
            else None
        )
    return agg


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    return f"{value:.3f}" if isinstance(value, float) else str(value)


def print_table(aggregates: dict[str, dict[str, Any]]) -> None:
    columns = list(aggregates["overall"])
    print(f"{'group':<14}" + "".join(f"{c:>19}" for c in columns))
    for group, agg in aggregates.items():
        print(f"{group:<14}" + "".join(f"{_fmt(agg[c]):>19}" for c in columns))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--retrieval-only", action="store_true")
    parser.add_argument("--golden", type=Path, default=GOLDEN_PATH)
    args = parser.parse_args()

    settings = get_settings()
    k = args.k or settings.top_k
    store = get_vector_store(settings, settings.retrieval_mode)
    reranker = get_reranker(settings)
    chain = None if args.retrieval_only else build_chain(settings, store=store, reranker=reranker)

    rows = []
    for record in load_golden(args.golden):
        if chain is None:
            retrieved = retrieve(
                store, record["question"], k, reranker=reranker, fetch_k=settings.rerank_fetch_k
            )
            result = None
        else:
            result = chain.invoke({"question": record["question"], "top_k": k})
            retrieved = result["retrieved"]
        pages = [doc.metadata["pages"] for doc, _ in retrieved]
        row = {
            **record,
            "rank": None
            if record["type"] == "unanswerable"
            else first_hit_rank(pages, record["expected_pages"]),
            "retrieved_pages": pages,
            "scores": [score for _, score in retrieved],
        }
        if result is not None:
            row["answer"] = result["answer"]
            row["refusal_correct"] = refusal_correct(result["answer"], record["type"])
            row["contains_expected"] = contains_expected(
                result["answer"], record["expected_answer"]
            )
        rows.append(row)
        print(f"{record['id']} {record['type']:<12} rank={row['rank']}", flush=True)

    by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_type[row["type"]].append(row)
    aggregates = {"overall": aggregate(rows, k)}
    aggregates |= {t: aggregate(by_type[t], k) for t in TYPES if by_type[t]}
    print()
    print_table(aggregates)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{datetime.now().astimezone():%Y%m%d-%H%M%S}-{args.label}.json"
    payload = {
        "label": args.label,
        "k": k,
        "retrieval_only": args.retrieval_only,
        "settings": settings.model_dump(),
        "aggregates": aggregates,
        "rows": rows,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
