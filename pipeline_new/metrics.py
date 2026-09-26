"""Did the pipeline actually beat the plain generator?

Four numbers, and they have to be read together:

  baseline   the generator's own top-1. What "a simple LLM generator" scores.
  pipeline   the reranked pick. What we are claiming is better.
  oracle@k   was the right answer anywhere in the k candidates? This is the HARD
             CEILING on reranking -- a scorer cannot pick an answer that was
             never proposed. If oracle@k is barely above baseline, the generator
             is the bottleneck and no amount of scorer work will help.
  random@k   pick uniformly from the candidates. The floor. A "reranker" that
             lands here has learned nothing; it is being credited for the
             generator's candidate list.

Plus a significance test, because at n=1000 a two-point accuracy gap is well
inside sampling noise and reporting it as an improvement would be wrong.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .adapters import ClueRecord, normalize_answer
from .pipeline import baseline_candidate, select_best


def is_correct(answer: str | None, gold: str | None) -> bool:
    if not answer or not gold:
        return False
    return normalize_answer(answer) == normalize_answer(gold)


# --------------------------------------------------------------------------
# Statistics (no scipy -- it is not in the cluster venv, and neither of these
# needs it)
# --------------------------------------------------------------------------
def mcnemar_exact(only_pipeline: int, only_baseline: int) -> float:
    """Two-sided exact binomial p-value for a paired win/loss count.

    The right test here: both systems answer the SAME clues, so the clues they
    both get right or both get wrong carry no information about which is better.
    Only the disagreements do. An unpaired test on two accuracy numbers would
    throw that pairing away and be far too conservative.
    """
    n = only_pipeline + only_baseline
    if n == 0:
        return 1.0
    smaller = min(only_pipeline, only_baseline)
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) / (2.0**n)
    return min(1.0, 2.0 * tail)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for an accuracy. Wilson rather than normal: it stays inside [0,1]
    and behaves sanely at the small counts a 100-clue debug run produces."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Rank-based AUC (Mann-Whitney U), ties counted as half.

    Measured over every scored candidate: "given a correct and an incorrect
    candidate, how often does the scorer rank the correct one higher?" This is
    the scorer's quality in isolation, independent of how good the candidates
    were -- which is what tells you whether to work on the scorer or the
    generator next.
    """
    pairs = sorted(zip(scores, labels))
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # Average ranks over tied score groups.
    ranks = [0.0] * len(pairs)
    i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for t in range(i, j + 1):
            ranks[t] = avg
        i = j + 1

    rank_sum_pos = sum(r for r, (_, label) in zip(ranks, pairs) if label == 1)
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


# --------------------------------------------------------------------------
# The metric bundle
# --------------------------------------------------------------------------
def compute_metrics(records: list[ClueRecord], selection_cfg: dict | None = None) -> dict:
    selection_cfg = selection_cfg or {}
    scored = [r for r in records if r.gold_answer]

    n = len(scored)
    if n == 0:
        return {
            "n_clues": len(records),
            "n_with_gold": 0,
            "note": "no gold answers available -- accuracy cannot be computed",
        }

    baseline_hits = 0
    filtered_baseline_hits = 0
    pipeline_hits = 0
    oracle_hits = 0
    oracle_unfiltered_hits = 0
    random_expectation = 0.0
    no_prediction = 0

    only_pipeline = 0  # pipeline right, baseline wrong -> the reranker's wins
    only_baseline = 0  # baseline right, pipeline wrong -> the reranker's losses

    recall_hits: dict[int, int] = {}
    max_k = 0

    cand_scores: list[float] = []
    cand_labels: list[int] = []
    chosen_scores_right: list[float] = []
    chosen_scores_wrong: list[float] = []
    total_candidates = 0
    total_live = 0

    for record in scored:
        gold = record.gold_answer
        live = [c for c in record.candidates if c.alive]
        total_candidates += len(record.candidates)
        total_live += len(live)

        base = baseline_candidate(record)
        base_ok = base is not None and is_correct(base.answer, gold)
        baseline_hits += base_ok

        # The generator's top pick among candidates that survived the filters:
        # isolates how much of any gain is the free enumeration filter rather
        # than the scorer. Without this the two are indistinguishable.
        filtered_base = min(live, key=lambda c: c.gen_rank) if live else None
        filtered_baseline_hits += filtered_base is not None and is_correct(
            filtered_base.answer, gold
        )

        chosen = select_best(record, selection_cfg)
        if chosen is None:
            no_prediction += 1
        pipeline_ok = chosen is not None and is_correct(chosen.answer, gold)
        pipeline_hits += pipeline_ok

        if pipeline_ok and not base_ok:
            only_pipeline += 1
        elif base_ok and not pipeline_ok:
            only_baseline += 1

        oracle_hits += any(is_correct(c.answer, gold) for c in live)
        oracle_unfiltered_hits += any(is_correct(c.answer, gold) for c in record.candidates)

        if live:
            n_right = sum(is_correct(c.answer, gold) for c in live)
            random_expectation += n_right / len(live)

        # recall@k over the GENERATOR's ordering: how deep do you have to look?
        by_rank = sorted(record.candidates, key=lambda c: c.gen_rank)
        max_k = max(max_k, len(by_rank))
        found_at = next(
            (i for i, c in enumerate(by_rank, start=1) if is_correct(c.answer, gold)), None
        )
        if found_at is not None:
            for k in range(found_at, len(by_rank) + 1):
                recall_hits[k] = recall_hits.get(k, 0) + 1

        for cand in live:
            if cand.score is not None:
                cand_scores.append(cand.score)
                cand_labels.append(int(is_correct(cand.answer, gold)))
        if chosen is not None and chosen.final_score is not None:
            (chosen_scores_right if pipeline_ok else chosen_scores_wrong).append(
                chosen.final_score
            )

    # recall@k is monotone; carry the last observed value forward so the curve
    # does not dip at k values beyond a short candidate list.
    recall_curve = {}
    running = 0
    for k in range(1, max_k + 1):
        running = max(running, recall_hits.get(k, 0))
        recall_curve[k] = running / n

    def mean(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    metrics = {
        "n_clues": len(records),
        "n_with_gold": n,
        "candidates": {
            "mean_generated": total_candidates / n,
            "mean_after_filters": total_live / n,
            "clues_with_no_prediction": no_prediction,
        },
        "accuracy": {
            "baseline_generator_top1": baseline_hits / n,
            "baseline_after_filters": filtered_baseline_hits / n,
            "pipeline": pipeline_hits / n,
            "random_from_candidates": random_expectation / n,
            "oracle_at_k": oracle_hits / n,
            "oracle_at_k_unfiltered": oracle_unfiltered_hits / n,
        },
        "counts": {
            "baseline_correct": baseline_hits,
            "pipeline_correct": pipeline_hits,
            "oracle_correct": oracle_hits,
        },
        "confidence_intervals_95": {
            "baseline": wilson_interval(baseline_hits, n),
            "pipeline": wilson_interval(pipeline_hits, n),
        },
        "improvement": {
            "delta_accuracy": (pipeline_hits - baseline_hits) / n,
            "rerank_wins": only_pipeline,
            "rerank_losses": only_baseline,
            "mcnemar_p_value": mcnemar_exact(only_pipeline, only_baseline),
            # Of the mistakes the baseline made that WERE fixable (the right
            # answer was in the candidate list), what fraction did we fix?
            # A cleaner read on the scorer than raw delta, which is capped by
            # how often the generator was already right.
            "headroom_captured": (
                (pipeline_hits - baseline_hits) / (oracle_hits - baseline_hits)
                if oracle_hits > baseline_hits
                else float("nan")
            ),
        },
        "scorer": {
            "candidate_auc": roc_auc(cand_scores, cand_labels),
            "mean_score_correct_candidates": mean(
                [s for s, y in zip(cand_scores, cand_labels) if y == 1]
            ),
            "mean_score_incorrect_candidates": mean(
                [s for s, y in zip(cand_scores, cand_labels) if y == 0]
            ),
            "mean_final_score_when_right": mean(chosen_scores_right),
            "mean_final_score_when_wrong": mean(chosen_scores_wrong),
        },
        "recall_at_k": recall_curve,
    }
    return metrics


def format_metrics(metrics: dict) -> str:
    """The summary block a Slurm log should end with."""
    if not metrics.get("n_with_gold"):
        return (
            f"\n{metrics['n_clues']} clue(s) processed. No gold answers, "
            "so no accuracy to report.\n"
        )

    acc = metrics["accuracy"]
    imp = metrics["improvement"]
    sc = metrics["scorer"]
    lo_b, hi_b = metrics["confidence_intervals_95"]["baseline"]
    lo_p, hi_p = metrics["confidence_intervals_95"]["pipeline"]

    lines = [
        "",
        "═" * 78,
        f"RESULTS over {metrics['n_with_gold']} clue(s) with a gold answer",
        "═" * 78,
        f"  random from candidates (floor)   {acc['random_from_candidates']:7.2%}",
        f"  BASELINE  generator top-1        {acc['baseline_generator_top1']:7.2%}"
        f"   [{lo_b:.2%}, {hi_b:.2%}]",
        f"            + enumeration filter   {acc['baseline_after_filters']:7.2%}",
        f"  PIPELINE  scorer argmax          {acc['pipeline']:7.2%}"
        f"   [{lo_p:.2%}, {hi_p:.2%}]",
        f"  oracle@k  (ceiling)              {acc['oracle_at_k']:7.2%}",
        "",
        f"  delta vs baseline                {imp['delta_accuracy']:+7.2%}",
        f"  rerank wins / losses             {imp['rerank_wins']} / {imp['rerank_losses']}",
        f"  McNemar exact p                  {imp['mcnemar_p_value']:.4g}"
        f"   {'(significant at .05)' if imp['mcnemar_p_value'] < 0.05 else '(NOT significant)'}",
        f"  headroom captured                {imp['headroom_captured']:7.2%}",
        "",
        f"  scorer AUC over candidates       {sc['candidate_auc']:.4f}",
        f"  mean score  correct / incorrect  {sc['mean_score_correct_candidates']:.4f}"
        f" / {sc['mean_score_incorrect_candidates']:.4f}",
        f"  mean candidates per clue         {metrics['candidates']['mean_generated']:.1f}"
        f" ({metrics['candidates']['mean_after_filters']:.1f} after filters)",
        "═" * 78,
    ]

    # Say what the numbers mean, so a run that looks like a win but is not gets
    # called out in the log rather than in a report a week later.
    gap = acc["oracle_at_k"] - acc["baseline_generator_top1"]
    if gap < 0.01:
        lines.append(
            "  DIAGNOSIS: oracle@k is barely above baseline -- the k candidates almost never\n"
            "             contain a better answer, so reranking has nothing to work with.\n"
            "             Work on the GENERATOR first: raise num_candidates or temperature\n"
            "             (generator.* for gemma_direct, generator.answer.* for\n"
            "             gemma_two_stage). Scorer work cannot help here."
        )
    elif imp["delta_accuracy"] <= 0:
        lines.append(
            "  DIAGNOSIS: the scorer is not beating the generator's own ranking despite\n"
            f"             {gap:.1%} of headroom. Retrain the scorer on generated hard\n"
            "             negatives -- candidates from this generator for the same clue."
        )
    elif imp["mcnemar_p_value"] >= 0.05:
        lines.append(
            "  DIAGNOSIS: the gain is in the right direction but not statistically\n"
            "             significant. Run more clues before claiming it."
        )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Artifacts
# --------------------------------------------------------------------------
def write_predictions_csv(records: list[ClueRecord], path: Path, selection_cfg: dict) -> None:
    import csv

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "id",
                "clue",
                "enumeration",
                "gold_answer",
                "pipeline_answer",
                "pipeline_reason",
                "pipeline_score",
                "baseline_answer",
                "pipeline_correct",
                "baseline_correct",
                "oracle_correct",
                "n_candidates",
                "n_after_filters",
            ]
        )
        for record in records:
            chosen = select_best(record, selection_cfg)
            base = baseline_candidate(record)
            live = [c for c in record.candidates if c.alive]
            writer.writerow(
                [
                    record.id,
                    record.clue,
                    record.enumeration or "",
                    record.gold_answer or "",
                    chosen.answer if chosen else "",
                    chosen.reason if chosen else "",
                    f"{chosen.final_score:.6f}" if chosen and chosen.final_score is not None else "",
                    base.answer if base else "",
                    int(is_correct(chosen.answer if chosen else None, record.gold_answer)),
                    int(is_correct(base.answer if base else None, record.gold_answer)),
                    int(any(is_correct(c.answer, record.gold_answer) for c in live)),
                    len(record.candidates),
                    len(live),
                ]
            )


def write_plots(records: list[ClueRecord], metrics: dict, fig_dir: Path) -> list[Path]:
    """Three figures. Returns the paths written (empty if matplotlib is absent)."""
    try:
        import matplotlib

        matplotlib.use("Agg")  # no display on a compute node
        import matplotlib.pyplot as plt
    except ImportError:
        return []

    if not metrics.get("n_with_gold"):
        return []

    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    acc = metrics["accuracy"]

    # 1. the headline comparison, with the floor and ceiling that make it readable
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["random\n(floor)", "baseline\ngenerator top-1", "+ enum\nfilter", "PIPELINE\nreranked", "oracle@k\n(ceiling)"]
    values = [
        acc["random_from_candidates"],
        acc["baseline_generator_top1"],
        acc["baseline_after_filters"],
        acc["pipeline"],
        acc["oracle_at_k"],
    ]
    colors = ["#bbbbbb", "#4878a8", "#6f9fc8", "#c1443c", "#7f9f6f"]
    bars = ax.bar(labels, values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1%}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_ylabel("exact-match accuracy")
    ax.set_ylim(0, max(values) * 1.25 + 1e-9)
    ax.set_title(f"Pipeline vs plain generator ({metrics['n_with_gold']} clues)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    path = fig_dir / "accuracy_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    written.append(path)

    # 2. recall@k -- how many candidates are worth generating
    curve = metrics.get("recall_at_k") or {}
    if curve:
        ks = sorted(int(k) for k in curve)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(ks, [curve[str(k)] if str(k) in curve else curve[k] for k in ks], marker="o")
        ax.axhline(
            acc["baseline_generator_top1"],
            color="#4878a8",
            linestyle="--",
            label="baseline top-1",
        )
        ax.axhline(acc["pipeline"], color="#c1443c", linestyle="--", label="pipeline")
        ax.set_xlabel("k (candidates inspected, generator order)")
        ax.set_ylabel("fraction of clues whose gold answer is in the top k")
        ax.set_title("Reranking headroom: oracle recall@k")
        ax.legend()
        ax.grid(linestyle=":", alpha=0.5)
        fig.tight_layout()
        path = fig_dir / "recall_at_k.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    # 3. is the scorer separating right from wrong at all?
    right = [
        c.score
        for r in records
        if r.gold_answer
        for c in r.candidates
        if c.alive and c.score is not None and is_correct(c.answer, r.gold_answer)
    ]
    wrong = [
        c.score
        for r in records
        if r.gold_answer
        for c in r.candidates
        if c.alive and c.score is not None and not is_correct(c.answer, r.gold_answer)
    ]
    if right and wrong:
        fig, ax = plt.subplots(figsize=(8, 5))
        bins = 30
        ax.hist(wrong, bins=bins, alpha=0.6, color="#c1443c", density=True, label=f"incorrect (n={len(wrong)})")
        ax.hist(right, bins=bins, alpha=0.6, color="#4878a8", density=True, label=f"correct (n={len(right)})")
        ax.set_xlabel("scorer P(correct)")
        ax.set_ylabel("density")
        auc = metrics["scorer"]["candidate_auc"]
        ax.set_title(f"Scorer separability over candidates (AUC = {auc:.3f})")
        ax.legend()
        ax.grid(linestyle=":", alpha=0.5)
        fig.tight_layout()
        path = fig_dir / "scorer_separability.png"
        fig.savefig(path, dpi=200)
        plt.close(fig)
        written.append(path)

    return written


def write_metrics_json(metrics: dict, path: Path) -> None:
    def clean(obj):
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
            return None  # JSON has no NaN; null is honest and parses everywhere
        return obj

    path.write_text(json.dumps(clean(metrics), indent=2), encoding="utf-8")
