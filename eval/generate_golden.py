"""Draft golden questions from stored chunks with the LLM.

Run on the host: `uv run python -m eval.generate_golden`. The output is a draft for human review.
"""

import json
import random
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.chain import get_llm
from app.config import get_settings
from app.ingest import CHUNKS_PATH

DRAFT_PATH = Path("eval/golden.draft.jsonl")
SEED = 7
N_TABLE = 12
N_PROSE = 10

PROMPT = """Poniżej znajduje się fragment protokołu leczenia PALG ALL7.

Ułóż jedno pytanie po polsku, na które ten fragment jednoznacznie odpowiada, \
oraz krótką oczekiwaną odpowiedź (dawka, dzień, nazwa leku itp.; maksymalnie 15 słów, \
bez długich cytatów). Pytanie powinno brzmieć jak pytanie lekarza i nie odwoływać się \
do "tego fragmentu".

Fragment:
{chunk}"""


class GoldenDraft(BaseModel):
    question: str = Field(description="Pytanie po polsku, na które fragment odpowiada")
    expected_answer: str = Field(description="Krótka odpowiedź, maksymalnie 15 słów")


def spread_sample(items: list[Any], n: int, rng: random.Random) -> list[Any]:
    """Pick up to `n` items, one at random from each of `n` contiguous slices of `items`."""
    if n >= len(items):
        return list(items)
    bounds = [round(i * len(items) / n) for i in range(n + 1)]
    return [rng.choice(items[bounds[i] : bounds[i + 1]]) for i in range(n)]


def load_chunks(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return sorted(rows, key=lambda r: r["metadata"]["chunk_index"])


def main() -> None:
    settings = get_settings()
    chunks = load_chunks(CHUNKS_PATH)
    rng = random.Random(SEED)
    tables = spread_sample([c for c in chunks if c["metadata"]["has_table"]], N_TABLE, rng)
    prose = spread_sample([c for c in chunks if not c["metadata"]["has_table"]], N_PROSE, rng)
    sampled = sorted(tables + prose, key=lambda c: c["metadata"]["chunk_index"])

    llm = get_llm(settings).with_structured_output(GoldenDraft)
    DRAFT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with DRAFT_PATH.open("w", encoding="utf-8") as f:
        for i, chunk in enumerate(sampled, start=1):
            meta = chunk["metadata"]
            draft: GoldenDraft = llm.invoke(PROMPT.format(chunk=chunk["page_content"]))
            record = {
                "id": f"q{i:02d}",
                "question": draft.question,
                "expected_answer": draft.expected_answer,
                "expected_pages": meta["pages"],
                "type": "table" if meta["has_table"] else "prose",
                "chunk_index": meta["chunk_index"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{record['id']} s.{meta['pages']} {record['type']}: {draft.question}")

    print(
        f"\nWrote {len(sampled)} drafts to {DRAFT_PATH}. Review them by hand, add 3 `diagram` "
        "(pages 5, 22, 68) and 5 `unanswerable` questions, then save as eval/golden.jsonl."
    )


if __name__ == "__main__":
    main()
