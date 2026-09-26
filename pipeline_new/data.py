"""Clue loading: one clue typed on the command line, or 26k from a split.

Reads the committed `*.jsonl.gz` splits directly with gzip + json rather than
through `datasets`. Two reasons: no Arrow cache to blow the $HOME quota, and
sharding/limiting stays a cheap streaming operation instead of materializing
half a million rows to select 10,000 of them.
"""

from __future__ import annotations

import gzip
import json
import random
import re
from pathlib import Path
from typing import Iterator

from . import paths
from .adapters import ClueRecord

SPLITS = ("train", "val", "test")


def split_path(split: str) -> Path:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    gz = paths.DATASET_DIR / f"{split}.jsonl.gz"
    plain = paths.DATASET_DIR / f"{split}.jsonl"
    if gz.exists():
        return gz
    if plain.exists():
        return plain
    raise FileNotFoundError(
        f"no {split} split at {gz} (or {plain}). The gzipped splits are committed "
        "to the repo; if this is a fresh clone, check out "
        "united-cryptonite-wordplay-dataset/."
    )


def _open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def iter_rows(path: Path) -> Iterator[dict]:
    with _open_text(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def row_to_record(row: dict, index: int = 0) -> ClueRecord:
    """Map a dataset row onto the pipeline's per-clue state.

    `clue` already has the enumeration appended inline (Cryptonite style), which
    is what the generator sees. `enumeration` is kept separately because the
    length filter needs it parsed, and the README notes that recovering it from
    the clue suffix is a nuisance worth avoiding.
    """
    return ClueRecord(
        id=str(row.get("id") or f"row-{index:06d}"),
        clue=str(row["clue"]).strip(),
        enumeration=(str(row["enumeration"]).strip() if row.get("enumeration") else None),
        gold_answer=(str(row["answer"]).strip() if row.get("answer") else None),
        gold_reason=(str(row["wordplay"]).strip() if row.get("wordplay") else None),
    )


def load_split(
    split: str,
    limit: int | None = None,
    sample: str = "random",
    seed: int = 42,
    shard: tuple[int, int] | None = None,
    require_gold: bool = True,
    require_reason: bool = False,
) -> list[ClueRecord]:
    """Load clues from a committed split.

    sample="random" (default) draws a reproducible random subset rather than the
    first N rows. This matters: the splits are ordered by source and date, so
    `head -1000` is one publisher from one period, and an accuracy measured on
    it does not generalize to the split it claims to represent.

    Sharding is applied AFTER sampling, so `--limit 10000` split across 4 shards
    covers exactly the same 10,000 clues a single unsharded job would.
    """
    rows = list(iter_rows(split_path(split)))
    records = [row_to_record(r, i) for i, r in enumerate(rows)]

    if require_gold:
        records = [r for r in records if r.gold_answer]
    if require_reason:
        # The test split has zero wordplay annotations by construction, so this
        # would silently return nothing there. Say so instead.
        records = [r for r in records if r.gold_reason]
        if not records:
            raise ValueError(
                f"--require-reason found no annotated clues in split {split!r}. "
                "The test split has no wordplay annotations at all (see the "
                "dataset README); use --split val for reason-quality work."
            )

    if limit is not None and limit < len(records):
        if sample == "random":
            random.Random(seed).shuffle(records)
        records = records[:limit]

    if shard is not None:
        index, count = shard
        if not 0 <= index < count:
            raise ValueError(f"bad shard {index}/{count}: need 0 <= index < count")
        records = [r for i, r in enumerate(records) if i % count == index]

    return records


def load_clues_file(path: str | Path) -> list[ClueRecord]:
    """Load clues from your own file: .jsonl (dataset schema), or .txt.

    A .txt file is one clue per line, enumeration inferred from the trailing
    "(6)" if present -- the quickest way to run the pipeline over clues from
    today's newspaper.
    """
    path = Path(paths.expand(str(path)))
    if not path.exists():
        raise FileNotFoundError(f"no clues file at {path}")

    if path.suffix in (".jsonl", ".gz"):
        return [row_to_record(row, i) for i, row in enumerate(iter_rows(path))]

    records = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line:
            records.append(single_clue(line, index=i))
    return records


def single_clue(
    clue: str,
    enumeration: str | None = None,
    gold_answer: str | None = None,
    gold_reason: str | None = None,
    index: int = 0,
) -> ClueRecord:
    """One ad-hoc clue. Enumeration is read off the clue text when not given."""
    clue = clue.strip()
    if enumeration is None:
        m = re.search(r"\(\s*[\d,\-\s]+\s*\)\s*$", clue)
        enumeration = m.group(0).strip() if m else None
    return ClueRecord(
        id=f"adhoc-{index:03d}",
        clue=clue,
        enumeration=enumeration,
        gold_answer=(gold_answer.strip().lower() if gold_answer else None),
        gold_reason=gold_reason,
    )
