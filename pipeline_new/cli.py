"""Command-line front end. One entry point for one clue and for ten thousand.

Two generator modes, picked by --config (default: gemma_two_stage):
  gemma_two_stage          answer adapter -> wordplay adapter
  gemma_reasoning_answer   one adapter emits reasoning + answer (gemma_direct)

    # one clue, full stage-by-stage trace
    python code/pipeline/run_pipeline.py --clue "attack general at end of month (6)"

    # see what the Gemma adapters actually emit (both stages for two-stage)
    python code/pipeline/run_pipeline.py --config gemma_reasoning_answer --probe 5 --split val

    # the real evaluation
    python code/pipeline/run_pipeline.py --config gemma_two_stage --split test --limit 10000

    # re-score an earlier generation with a different scorer, no regeneration
    python code/pipeline/run_pipeline.py --split test --limit 10000 \
        --stage score --candidates-from runs/20260911-1200-gemma_two_stage-test \
        --set scorer.model=<other-scorer-repo>

    # is the scorer wired up correctly at all? (needs no generator)
    python code/pipeline/run_pipeline.py --verify-scorer 200
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from . import config_io, data, metrics as metrics_mod, paths, pipeline
from .adapters import ClueRecord

DEFAULT_CONFIG = "gemma_two_stage"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="run_pipeline",
        description="Cryptic crossword pipeline: generate k candidates, score them, pick the best.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    g = p.add_argument_group("models")
    g.add_argument("--config", default=DEFAULT_CONFIG, help="config name or path")
    g.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="dotted config override, e.g. --set generator.num_candidates=20 (repeatable)",
    )

    g = p.add_argument_group("input (pick one)")
    g.add_argument("--clue", action="append", default=[], help="an ad-hoc clue (repeatable)")
    g.add_argument("--enumeration", help="enumeration for --clue, e.g. '(4,2)'; else read off the clue")
    g.add_argument("--gold", help="known answer for a single --clue, to check correctness")
    g.add_argument("--split", choices=data.SPLITS, help="evaluate on a dataset split")
    g.add_argument("--clues-file", help="your own .jsonl (dataset schema) or .txt (one clue/line)")

    g = p.add_argument_group("sampling")
    g.add_argument("--limit", type=int, help="use at most N clues from the split")
    g.add_argument(
        "--sample",
        choices=("random", "head"),
        default="random",
        help="random (default) draws a reproducible subset; head takes the first N, "
        "which is one publisher from one period and not representative",
    )
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--shard", metavar="I/N", help="process shard I of N, for parallel jobs")
    g.add_argument(
        "--require-reason",
        action="store_true",
        help="keep only clues with a human wordplay annotation (val only; test has none)",
    )

    g = p.add_argument_group("execution")
    g.add_argument("--stage", choices=("all", "generate", "score"), default="all")
    g.add_argument("--candidates-from", help="reuse candidates from a previous run dir")
    g.add_argument("--run-dir", help="explicit output directory (default: runs/<timestamp>-<config>)")
    g.add_argument("--tag", help="suffix for the auto-generated run directory name")
    g.add_argument("--no-resume", action="store_true", help="ignore an existing partial run")
    g.add_argument(
        "--print-first",
        type=int,
        help="stage-by-stage trace for the first K clues (default: all for --clue, else 3)",
    )
    g.add_argument("--progress-every", type=int, default=50)

    g = p.add_argument_group("analysis")
    g.add_argument("--probe", type=int, metavar="N", help="dump raw decodes for N clues and exit")
    g.add_argument(
        "--verify-scorer",
        type=int,
        nargs="?",
        const=200,
        metavar="N",
        help="check the scorer on N annotated val clues and exit; needs no generator "
        "and no --split. Run this first after swapping in a new scorer",
    )
    g.add_argument("--no-plots", action="store_true")

    return p


def parse_shard(spec: str | None) -> tuple[int, int] | None:
    if not spec:
        return None
    try:
        index, count = spec.split("/")
        return int(index), int(count)
    except ValueError as exc:
        raise SystemExit(f"--shard wants I/N (e.g. 0/4), got {spec!r}") from exc


def resolve_records(args: argparse.Namespace) -> list[ClueRecord]:
    sources = sum(bool(x) for x in (args.clue, args.split, args.clues_file))
    if sources == 0:
        raise SystemExit("nothing to do: pass --clue, --split or --clues-file")
    if sources > 1:
        raise SystemExit("--clue, --split and --clues-file are mutually exclusive")

    if args.clue:
        if len(args.clue) > 1 and (args.enumeration or args.gold):
            raise SystemExit("--enumeration/--gold apply to a single --clue only")
        return [
            data.single_clue(
                text,
                enumeration=args.enumeration,
                gold_answer=args.gold,
                index=i,
            )
            for i, text in enumerate(args.clue)
        ]

    if args.clues_file:
        records = data.load_clues_file(args.clues_file)
    else:
        records = data.load_split(
            args.split,
            limit=args.limit,
            sample=args.sample,
            seed=args.seed,
            shard=parse_shard(args.shard),
            require_reason=args.require_reason,
        )

    if args.clues_file and args.limit:
        records = records[: args.limit]
    return records


def make_run_dir(args: argparse.Namespace, cfg: dict) -> Path:
    if args.run_dir:
        return Path(paths.expand(args.run_dir))
    parts = [datetime.now().strftime("%Y%m%d-%H%M%S"), str(cfg.get("name", "run"))]
    if args.split:
        parts.append(args.split)
    if args.tag:
        parts.append(args.tag)
    shard = parse_shard(args.shard)
    if shard:
        parts.append(f"shard{shard[0]}of{shard[1]}")
    return paths.runs_root() / "-".join(parts)


def environment_snapshot() -> dict:
    """What produced these numbers. Cheap now, invaluable when a run disagrees
    with a later one and nobody remembers which transformers version was live."""
    snap = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "storage_root": str(paths.storage_root()),
        "on_cluster": paths.on_cluster(),
        "argv": sys.argv,
    }
    try:
        import torch
        import transformers

        snap["torch"] = torch.__version__
        snap["transformers"] = transformers.__version__
        snap["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            snap["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        snap["git_commit"] = subprocess.run(
            ["git", "-C", str(paths.REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        pass
    return snap


def run_probe(records: list[ClueRecord], cfg: dict, count: int) -> None:
    """Print the exact prompts and raw generator output, then how each decode parses.

    The first thing to run against an unfamiliar checkpoint. `parse.patterns`
    cannot be written without seeing this, and a wrong pattern does not error --
    for gemma_direct every unmatched decode silently gets an empty answer and is
    dropped. Every decode is parsed, so the match count covers all k of them.
    """
    from .adapters import GemmaTwoStageGenerator, build_generator, describe_models

    subset = records[: max(1, count)]
    print("\n".join(describe_models({"generator": cfg["generator"]})))
    generator = build_generator(cfg["generator"])
    print(f"Probing {generator.description}\n")

    # gemma_direct's single decode carries answer + reason and is parsed by
    # `parse.patterns`; gemma_two_stage has no such step -- its reason is the
    # wordplay stage's text verbatim.
    two_stage = isinstance(generator, GemmaTwoStageGenerator)

    for record, blocks in generator.probe(subset):
        print("═" * 100)
        print(f"CLUE   {record.clue}")
        if record.gold_answer:
            print(f"GOLD   {record.gold_answer}")
        for label, prompt, decodes in blocks:
            print(f"[{label}] PROMPT {prompt!r}")
            print(f"[{label}] RAW DECODES:")
            for i, text in enumerate(decodes):
                print(f"  [{i:2d}] {text!r}")
        if not two_stage:
            for i, text in enumerate(blocks[0][2]):
                parsed_answer, parsed_reason = generator.parse_output(text, record.enumeration)
                print(f"PARSED [{i}] answer={parsed_answer!r} reason={parsed_reason!r}")
        print()

    stats = generator.stats
    print("─" * 100)
    if two_stage:
        print(f"{stats['reason_parsed']}/{stats['decoded']} wordplay decodes contain a `wordplay:` field.")
        return
    print(
        f"{stats['reason_parsed']}/{stats['decoded']} decodes matched a "
        f"`parse.patterns` entry and yielded a reason."
    )
    if not stats["reason_parsed"]:
        print(
            "None matched, so every candidate would get an empty answer and be dropped.\n"
            "Copy the shape you see above into generator.parse.patterns as a regex with\n"
            "named groups (?P<answer>...) and (?P<reason>...)."
        )


def verify_scorer(cfg: dict, count: int = 200, seed: int = 0) -> dict:
    """Is the scorer loaded, wired to the right template, and separating anything?

    Run this FIRST after swapping in a new scorer. It needs no generator, so it is
    fast and cannot be confounded by generation quality.

    Nothing in this repo records the input template a delivered checkpoint was
    fine-tuned with, and a mismatched template does not error -- it produces
    confident nonsense. This is the check that catches that, plus a wrong
    positive-class index, plus a corrupt or base-model checkpoint.

    Two negative types are scored, and the difference between them is the point:

      training-style   (clue_i, answer_j, reason_j) -- a different clue's answer
                       AND reasoning. This is exactly how the fine-tuning notebook
                       built its negatives, so this number is comparable to the
                       "pairwise ranking accuracy" that the training run reported.
                       If THIS is near chance, something is wrong with the setup.

      rerank-style     (clue_i, answer_i, reason_j) -- the CORRECT answer with
                       someone else's reasoning. Harder, and much closer to what
                       reranking actually asks. The scorer never saw this during
                       training, so a large gap between the two columns is the
                       out-of-distribution problem quantified -- and predicts a
                       disappointing reranking delta before you spend a GPU-hour
                       discovering it.
    """
    import random as _random

    from .adapters import Candidate, build_scorer
    from .metrics import roc_auc

    records = data.load_split("val", limit=count, seed=seed, require_reason=True)
    scorer = build_scorer(cfg["scorer"])

    print(f"Scorer:   {scorer.description}")
    print(f"Template: {cfg['scorer']['template']!r}")
    print(f"Clues:    {len(records)} annotated val clues\n")

    rng = _random.Random(seed)
    probes: list[ClueRecord] = []
    for i, rec in enumerate(records):
        j = rng.choice([k for k in range(len(records)) if k != i])
        other = records[j]
        probes.append(
            ClueRecord(
                id=rec.id,
                clue=rec.clue,
                enumeration=rec.enumeration,
                gold_answer=rec.gold_answer,
                candidates=[
                    # index 0 positive, 1 training-style negative, 2 rerank-style
                    Candidate(rec.gold_answer, rec.gold_reason, "", 0),
                    Candidate(other.gold_answer, other.gold_reason, "", 1),
                    Candidate(rec.gold_answer, other.gold_reason, "", 2),
                ],
            )
        )

    scorer.score(probes)
    pos = [p.candidates[0].score for p in probes]
    neg_train = [p.candidates[1].score for p in probes]
    neg_rerank = [p.candidates[2].score for p in probes]

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    def pairwise(a, b):
        return sum(x > y for x, y in zip(a, b)) / len(a)

    result = {
        "n_clues": len(probes),
        "mean_score_human_reasoning": mean(pos),
        "training_style_negative": {
            "mean_score": mean(neg_train),
            "pairwise_accuracy": pairwise(pos, neg_train),
            "auc": roc_auc(pos + neg_train, [1] * len(pos) + [0] * len(neg_train)),
        },
        "rerank_style_negative": {
            "mean_score": mean(neg_rerank),
            "pairwise_accuracy": pairwise(pos, neg_rerank),
            "auc": roc_auc(pos + neg_rerank, [1] * len(pos) + [0] * len(neg_rerank)),
        },
    }

    tr, rr = result["training_style_negative"], result["rerank_style_negative"]
    print("═" * 74)
    print(f"  mean score, human reasoning (positive)   {result['mean_score_human_reasoning']:.4f}")
    print()
    print(f"  {'':38s} {'training-style':>15s} {'rerank-style':>15s}")
    print(f"  {'negative construction':38s} {'other clue a+r':>15s} {'gold a, other r':>15s}")
    print(f"  {'mean score':38s} {tr['mean_score']:15.4f} {rr['mean_score']:15.4f}")
    print(f"  {'pairwise accuracy vs positive':38s} {tr['pairwise_accuracy']:15.2%} {rr['pairwise_accuracy']:15.2%}")
    print(f"  {'AUC':38s} {tr['auc']:15.4f} {rr['auc']:15.4f}")
    print("═" * 74)

    # Turn the numbers into a verdict, so a broken setup is called out here and
    # not after a 10k run.
    if tr["auc"] < 0.6:
        print(
            "  VERDICT: BROKEN. The scorer barely separates a human reasoning from an\n"
            "           unrelated clue's -- the task it was explicitly trained on. Suspect,\n"
            "           in order: a `scorer.template` that does not match training; the\n"
            "           wrong positive class (try --set scorer.positive_label=CORRECT, or\n"
            "           0 if the labels are reversed); a truncated checkpoint."
        )
    elif tr["auc"] < 0.5:
        print("  VERDICT: INVERTED. Set scorer.positive_label to the other class.")
    else:
        print(f"  VERDICT: the scorer works on its training task (AUC {tr['auc']:.3f}).")
        if rr["auc"] < 0.6:
            print(
                "           But it is near chance on rerank-style negatives -- the correct\n"
                "           answer paired with someone else's reasoning. That is the\n"
                "           distribution reranking actually operates in, so expect a small or\n"
                "           zero delta over the baseline. Mitigation: retrain the scorer\n"
                "           on generated hard negatives."
            )
        elif rr["auc"] < tr["auc"] - 0.1:
            print(
                f"           Weaker on rerank-style negatives ({rr['auc']:.3f} vs {tr['auc']:.3f}),\n"
                "           which is expected and is the headroom to watch."
            )
    print()
    return result


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    cfg = config_io.load_config(args.config)
    cfg = config_io.apply_overrides(cfg, args.overrides)

    # Handled before resolve_records: this checks the scorer in isolation and
    # supplies its own clues, so it must not require --split / --clue.
    if args.verify_scorer:
        verify_scorer(cfg, args.verify_scorer, args.seed)
        return 0

    records = resolve_records(args)
    if not records:
        raise SystemExit("no clues to process after filtering")

    if args.probe:
        run_probe(records, cfg, args.probe)
        return 0

    run_dir = make_run_dir(args, cfg)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Snapshot the RESOLVED config, after extends and --set. The config file on
    # disk is not what ran if anything was overridden.
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (run_dir / "environment.json").write_text(
        json.dumps(environment_snapshot(), indent=2), encoding="utf-8"
    )

    print_first = args.print_first
    if print_first is None:
        print_first = len(records) if args.clue else 3

    print(f"Config:   {cfg.get('name')}  ({args.config})")
    print(f"Clues:    {len(records)}")
    print(f"Run dir:  {run_dir}")
    print(f"Stage:    {args.stage}\n")

    started = time.time()
    outcome = pipeline.run(
        records,
        cfg,
        run_dir,
        stage=args.stage,
        print_first=print_first,
        resume=not args.no_resume,
        candidates_from=Path(paths.expand(args.candidates_from)) if args.candidates_from else None,
        progress_every=args.progress_every,
    )

    if args.stage == "generate":
        print(f"\nGenerated candidates for {len(outcome.records)} clue(s) -> {run_dir}/records.jsonl")
        print("Re-run with --stage score --candidates-from " + str(run_dir))
        return 0

    result = metrics_mod.compute_metrics(outcome.records, cfg.get("selection", {}))
    result["filters"] = outcome.filter_stats.as_dict()
    result["generator_parse"] = outcome.generator_stats
    result["runtime_seconds"] = round(time.time() - started, 1)
    result["config_name"] = cfg.get("name")
    result["scorer_head_untrained"] = outcome.scorer_head_untrained

    print(metrics_mod.format_metrics(result))

    if outcome.scorer_head_untrained:
        # Repeat this AFTER the numbers. The warning at model-load time is
        # thousands of log lines earlier by now, and these numbers are void.
        print(
            "!" * 78
            + "\n  RESULTS ABOVE ARE VOID: the scorer had a randomly initialized head, so\n"
            "  every score was noise. Use the fine-tuned scorer:\n"
            "  --set scorer.model=ronietz/cryptic-deberta-large-reasoning-scorer\n"
            + "!" * 78
            + "\n"
        )

    metrics_mod.write_metrics_json(result, run_dir / "metrics.json")
    metrics_mod.write_predictions_csv(
        outcome.records, run_dir / "predictions.csv", cfg.get("selection", {})
    )
    figures = [] if args.no_plots else metrics_mod.write_plots(
        outcome.records, result, run_dir / "figures"
    )

    print(f"Artifacts in {run_dir}:")
    for name in ("config.json", "environment.json", "records.jsonl", "metrics.json", "predictions.csv"):
        if (run_dir / name).exists():
            print(f"  {name}")
    for path in figures:
        print(f"  figures/{path.name}")
    if not figures and not args.no_plots:
        print("  (no figures -- matplotlib missing, or no gold answers to plot)")

    return 0
