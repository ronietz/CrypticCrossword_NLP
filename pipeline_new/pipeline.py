"""The pipeline: clue -> k (answer, reason) candidates -> scored -> best.

Three stages, each observable:

    1 GENERATE   Gemma proposes k (answer, reason) candidates -- direct, or
                 answer -> wordplay -- then dedup and cheap filters
    2 SCORE      cross-encoder scores every surviving (clue, answer, reason)
    3 SELECT     argmax over the final score

Stage 1's `gen_rank == 0` candidate is preserved as the BASELINE -- the greedy
answer the plain generator would have given. Every run therefore
measures the pipeline against its own baseline on exactly the same clues, which
is the only comparison that supports the claim "better than a simple generator".

Work happens in chunks and each finished clue is appended to `records.jsonl`
immediately. That is what makes a 10k run survivable on the `studentkillable`
partition: a preempted job resumes from the last completed clue instead of
starting over.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from . import paths
from .adapters import (
    Candidate,
    ClueRecord,
    build_generator,
    build_scorer,
    describe_models,
    matches_enumeration,
    normalize_answer,
    pick_device,
)


# --------------------------------------------------------------------------
# Cheap, model-free candidate filters
# --------------------------------------------------------------------------
@dataclass
class FilterStats:
    total: int = 0
    dropped_empty: int = 0
    dropped_length: int = 0
    dropped_enumeration: int = 0
    rescued_clues: int = 0  # clues where every candidate was dropped, so we kept them all

    def as_dict(self) -> dict:
        return {
            "candidates_seen": self.total,
            "dropped_empty_answer": self.dropped_empty,
            "dropped_bad_length": self.dropped_length,
            "dropped_enumeration_mismatch": self.dropped_enumeration,
            "clues_with_all_candidates_dropped": self.rescued_clues,
        }


def apply_filters(record: ClueRecord, cfg: dict, stats: FilterStats) -> None:
    """Mark candidates as dropped in place. Nothing is deleted.

    Dropped candidates stay in the record so `records.jsonl` remains a complete
    audit trail -- you can re-derive metrics under a different filter setting
    from a finished run without regenerating anything.

    An answer whose letter count does not fit the enumeration cannot be right:
    it will not physically fit the grid. That makes this filter free accuracy,
    and it is the one lever here that does not depend on either model being good.
    """
    use_enumeration = bool(cfg.get("enumeration", True))
    min_chars = int(cfg.get("min_answer_chars", 2))
    max_chars = int(cfg.get("max_answer_chars", 40))

    for cand in record.candidates:
        stats.total += 1
        letters = normalize_answer(cand.answer)

        if not letters:
            cand.dropped = "empty_answer"
            stats.dropped_empty += 1
        elif not (min_chars <= len(letters) <= max_chars):
            cand.dropped = f"length_{len(letters)}"
            stats.dropped_length += 1
        elif use_enumeration and not matches_enumeration(cand.answer, record.enumeration):
            cand.dropped = f"enumeration_{record.enumeration}"
            stats.dropped_enumeration += 1

    # A clue with nothing left would have no prediction at all, which is worse
    # than a wrong one: it silently removes the clue from the comparison and
    # inflates whichever accuracy is computed over "clues with a prediction".
    # Always answer something.
    if record.candidates and not any(c.alive for c in record.candidates):
        stats.rescued_clues += 1
        record.rescued = True
        for cand in record.candidates:
            cand.dropped = None


# --------------------------------------------------------------------------
# Stage 3 -- selection
# --------------------------------------------------------------------------
def select_best(record: ClueRecord, cfg: dict) -> Candidate | None:
    """Set `final_score` on every live candidate and return the argmax.

    Two modes (`selection.combine`):

      scorer     (default) final = the DeBERTa scorer's P(correct). The pipeline
                 as designed.
      generator  final = 1 for the best-ranked surviving candidate, 0 for the
                 rest -- the generator's own order, i.e. the baseline after the
                 filters. Running in this mode is the ablation that proves the
                 scorer is what is doing the work.

    Ties break toward the lower generator rank, so a tie can never manufacture a
    win over the baseline out of nondeterminism.
    """
    live = [c for c in record.candidates if c.alive]
    if not live:
        return None

    mode = cfg.get("combine", "scorer")
    if mode == "scorer":
        for cand in live:
            cand.final_score = 0.0 if cand.score is None else float(cand.score)
    elif mode == "generator":
        top = min(c.gen_rank for c in live)
        for cand in live:
            cand.final_score = 1.0 if cand.gen_rank == top else 0.0
    else:
        raise ValueError(f"unknown selection.combine {mode!r} (expected 'scorer' or 'generator')")

    return max(live, key=lambda c: (c.final_score, -c.gen_rank))


def baseline_candidate(record: ClueRecord) -> Candidate | None:
    """What a plain generator would have returned: its own rank-0 decode.

    Read off the unfiltered candidate list on purpose. The baseline must be the
    honest "simple LLM generator" answer, not a version already improved by the
    pipeline's enumeration filter -- crediting the baseline with our filter would
    understate the pipeline, and crediting the pipeline with it while denying it
    to the baseline would overstate it. It belongs to the pipeline, so it is
    reported separately (see metrics' `filtered_baseline`).
    """
    if not record.candidates:
        return None
    return min(record.candidates, key=lambda c: c.gen_rank)


# --------------------------------------------------------------------------
# Stage reporting
# --------------------------------------------------------------------------
def _truncate(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 1] + "…"


def print_stage_report(
    record: ClueRecord,
    chosen: Candidate | None,
    baseline: Candidate | None,
    stream=sys.stdout,
) -> None:
    """Human-readable trace of all three stages for one clue."""
    w = stream.write
    gold = record.gold_answer
    correct = (
        chosen is not None
        and gold is not None
        and normalize_answer(chosen.answer) == normalize_answer(gold)
    )

    w("\n" + "═" * 100 + "\n")
    w(f"CLUE  [{record.id}]  {record.clue}\n")
    if record.enumeration:
        w(f"      enumeration {record.enumeration}\n")
    if gold:
        w(f"GOLD  {gold}\n")
    if record.gold_reason:
        w(f"      {_truncate(record.gold_reason, 88)}\n")

    live = [c for c in record.candidates if c.alive]
    dropped = [c for c in record.candidates if not c.alive]

    w("\n── stage 1  GENERATE " + "─" * 79 + "\n")
    if record.rescued:
        w(f"   {len(record.candidates)} unique candidates, but EVERY one failed the filters")
        w("\n   -> all reinstated so the clue still gets an answer (rescue); expect a wrong pick\n")
    else:
        w(f"   {len(record.candidates)} unique candidates, {len(live)} survived filters")
        if dropped:
            w(f", {len(dropped)} dropped")
        w("\n")
    if baseline is not None:
        w(f"   generator top-1 (the baseline): {baseline.answer!r}\n")

    w("\n── stage 2  SCORE " + "─" * 82 + "\n")
    w(f"   {'rank':>4}  {'answer':<22} {'score':>7} {'gen':>4} {'final':>7}  reason\n")
    ordered = sorted(
        live, key=lambda c: (-(c.final_score if c.final_score is not None else -1), c.gen_rank)
    )
    for rank, cand in enumerate(ordered, start=1):
        marker = "*" if cand is chosen else " "
        hit = "✓" if gold and normalize_answer(cand.answer) == normalize_answer(gold) else " "
        score = "    n/a" if cand.score is None else f"{cand.score:7.4f}"
        final = "    n/a" if cand.final_score is None else f"{cand.final_score:7.4f}"
        w(
            f" {marker}{hit}{rank:>3}  {_truncate(cand.answer, 22):<22} "
            f"{score} {cand.gen_rank:>4} {final}  {_truncate(cand.reason, 40)}\n"
        )
    for cand in dropped:
        w(f"   ---  {_truncate(cand.answer, 22):<22} {'dropped':>7} {cand.dropped:>17}\n")

    w("\n── stage 3  SELECT " + "─" * 81 + "\n")
    if chosen is None:
        w("   no candidate survived -- no prediction\n")
    else:
        verdict = ""
        if gold:
            verdict = "  ✓ CORRECT" if correct else "  ✗ wrong"
        w(f"   ANSWER  {chosen.answer}{verdict}\n")
        w(f"   REASON  {_truncate(chosen.reason, 88) or '(none produced)'}\n")
        w(f"   score   {chosen.final_score:.4f}   (was generator rank {chosen.gen_rank})\n")
        if baseline is not None and gold:
            base_ok = normalize_answer(baseline.answer) == normalize_answer(gold)
            if correct and not base_ok:
                w(f"   → RERANK WIN: baseline said {baseline.answer!r}, pipeline fixed it\n")
            elif base_ok and not correct:
                w(f"   → RERANK LOSS: baseline had it right ({baseline.answer!r}), we broke it\n")
    stream.flush()


# --------------------------------------------------------------------------
# The run loop
# --------------------------------------------------------------------------
@dataclass
class RunOutcome:
    records: list[ClueRecord]
    run_dir: Path
    filter_stats: FilterStats
    generator_stats: dict
    elapsed_seconds: float
    resumed: int
    # True when the scorer resolved to a base model with a randomly initialized
    # head, i.e. the fine-tuned checkpoint was not in place. Recorded rather than
    # only warned about: a warning scrolls out of a Slurm log, and a run whose
    # scores were noise must not be mistakable for a real result later.
    scorer_head_untrained: bool = False


def _load_done(records_path: Path) -> dict[str, ClueRecord]:
    """Read back already-finished clues from a partial run."""
    done: dict[str, ClueRecord] = {}
    if not records_path.exists():
        return done
    with open(records_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = ClueRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                # A job killed mid-write leaves one truncated final line. Drop
                # it and redo that clue; anything else would corrupt the run.
                continue
            done[record.id] = record
    return done


def run(
    records: list[ClueRecord],
    cfg: dict,
    run_dir: Path,
    stage: str = "all",
    print_first: int = 3,
    resume: bool = True,
    candidates_from: Path | None = None,
    progress_every: int = 50,
    stream=sys.stdout,
) -> RunOutcome:
    """Execute the pipeline over `records`, writing artifacts into `run_dir`.

    stage="generate"  stop after stage 1. Writes candidates, no scores.
    stage="score"     require candidates from a previous run; skip the generator
                      entirely. This is how you evaluate a new scorer against an
                      existing 10k generation without paying for generation
                      again -- the expensive half of the pipeline by far.
    stage="all"       both, in one pass with both models resident.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    records_path = run_dir / "records.jsonl"

    device = pick_device(cfg.get("device", "auto"))
    started = time.time()

    # ---- reuse a previous generation, if asked -----------------------------
    cached: dict[str, ClueRecord] = {}
    if candidates_from is not None:
        source = Path(candidates_from)
        if source.is_dir():
            source = source / "records.jsonl"
        cached = _load_done(source)
        stream.write(f"Reusing {len(cached)} generated clue(s) from {source}\n")
        missing = [r.id for r in records if r.id not in cached]
        if missing:
            stream.write(
                f"WARNING: {len(missing)} requested clue(s) are absent from that run "
                f"(first: {missing[:3]}); they will be regenerated.\n"
            )

    # ---- resume a partial run of THIS config ------------------------------
    done = _load_done(records_path) if resume else {}
    pending = [r for r in records if r.id not in done]
    resumed = len(records) - len(pending)
    if resumed:
        stream.write(f"Resuming: {resumed} clue(s) already complete in {records_path}\n")

    need_generator = stage in ("all", "generate") and any(r.id not in cached for r in pending)
    if stage == "score" and not cached:
        raise ValueError("stage='score' needs --candidates-from pointing at a finished run")

    # Which model runs at which stage, straight from the config, before any load.
    stream.write("Models:\n")
    for line in describe_models({
        **({"generator": cfg["generator"]} if need_generator else {}),
        **({"scorer": cfg["scorer"]} if stage in ("all", "score") else {}),
    }):
        stream.write(f"  {line}\n")
    if not need_generator:
        stream.write("  Generator: not loaded (candidates reused)\n")
    stream.flush()

    generator = None
    if need_generator:
        stream.write("Loading generator …\n")
        stream.flush()
        generator = build_generator(cfg["generator"], device)
        stream.write(f"  generator: {generator.description}\n")

    scorer = None
    if stage in ("all", "score"):
        stream.write("Loading scorer …\n")
        stream.flush()
        scorer = build_scorer(cfg["scorer"], device)
        stream.write(f"  scorer:    {scorer.description}\n")

    stream.write(f"  device:    {device}\n\n")
    stream.flush()

    filter_cfg = cfg.get("filters", {})
    selection_cfg = cfg.get("selection", {})
    filter_stats = FilterStats()
    chunk_size = int(cfg.get("chunk_size", 32))
    printed = 0
    completed = list(done.values())

    with open(records_path, "a", encoding="utf-8") as sink:
        for start in range(0, len(pending), chunk_size):
            chunk = pending[start : start + chunk_size]

            # stage 1
            to_generate = []
            for record in chunk:
                if record.id in cached:
                    record.candidates = [
                        Candidate.from_dict(c.to_dict()) for c in cached[record.id].candidates
                    ]
                    for cand in record.candidates:
                        # Re-derive these; they belong to the scorer/selector,
                        # not to the generation being reused.
                        cand.score = None
                        cand.final_score = None
                        cand.dropped = None
                else:
                    to_generate.append(record)
            if to_generate and generator is not None:
                generator.generate(to_generate)

            for record in chunk:
                apply_filters(record, filter_cfg, filter_stats)

            # stage 2
            if scorer is not None:
                scorer.score(chunk)

            # stage 3
            for record in chunk:
                chosen = select_best(record, selection_cfg) if scorer is not None else None
                if printed < print_first:
                    print_stage_report(record, chosen, baseline_candidate(record), stream)
                    printed += 1
                sink.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            sink.flush()  # a preempted job must not lose the chunk it finished
            completed.extend(chunk)

            seen = start + len(chunk)
            if progress_every and (seen % progress_every < chunk_size or seen == len(pending)):
                rate = seen / max(time.time() - started, 1e-6)
                remaining = (len(pending) - seen) / max(rate, 1e-9)
                stream.write(
                    f"[{seen}/{len(pending)}] {rate:.2f} clue/s, "
                    f"~{remaining / 60:.1f} min left\n"
                )
                stream.flush()

    generator_stats = dict(generator.stats) if generator is not None else {}
    if generator_stats.get("decoded") and not generator_stats.get("reason_parsed"):
        stream.write(
            "\nWARNING: no decode yielded a reason (no `parse.patterns` match; for "
            "gemma_two_stage, no wordplay decode contained `wordplay:`), so the scorer "
            "is judging (clue, answer) only.\n"
            "         If this checkpoint is supposed to emit reasoning, run with --probe "
            "to see its actual output format, then fix generator.parse.patterns.\n\n"
        )

    # Restore the caller's ordering; resumed records were appended out of order.
    by_id = {r.id: r for r in completed}
    ordered = [by_id[r.id] for r in records if r.id in by_id]

    return RunOutcome(
        scorer_head_untrained=bool(getattr(scorer, "head_looks_untrained", False)),
        records=ordered,
        run_dir=run_dir,
        filter_stats=filter_stats,
        generator_stats=generator_stats,
        elapsed_seconds=time.time() - started,
        resumed=resumed,
    )
