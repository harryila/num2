from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .types import QAItem


def normalize_answer(answer: str) -> str:
    return " ".join(answer.strip().lower().split())


def load_closed_book_jsonl(path: str | Path) -> list[QAItem]:
    """Load closed-book QA items from jsonl with keys item_id/prompt/target.

    Extra keys are ignored.
    """

    items: list[QAItem] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            items.append(
                QAItem(
                    item_id=str(row["item_id"]),
                    prompt=str(row["prompt"]),
                    target=normalize_answer(str(row["target"])),
                    domain_tag=str(row.get("domain_tag", "squad")),
                    difficulty=row.get("difficulty"),
                )
            )
    return items


def write_closed_book_jsonl(path: str | Path, items: Iterable[QAItem]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for item in items:
            row = {
                "item_id": item.item_id,
                "prompt": item.prompt,
                "target": item.target,
                "domain_tag": item.domain_tag,
            }
            if item.difficulty is not None:
                row["difficulty"] = item.difficulty
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_sample_dataset(n: int = 200) -> list[QAItem]:
    """Small deterministic offline dataset for smoke testing the pipeline."""

    items: list[QAItem] = []
    for i in range(n):
        items.append(
            QAItem(
                item_id=f"sample-{i}",
                prompt=f"What is fact number {i}?",
                target=normalize_answer(f"fact-{i}"),
                difficulty=((i % 10) / 10.0),
            )
        )
    return items
