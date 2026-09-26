"""Cryptic crossword solving pipeline: generate candidates, score them, pick one.

    from pipeline import solve
    result = solve("attack general at end of month (6)")

Stage-by-stage detail, batch evaluation and metrics live behind
`pipeline.run_pipeline`; see README.md.
"""

from __future__ import annotations

from . import paths

paths.setup_environment()  # before anything imports transformers

__all__ = ["solve", "load_config", "paths"]

_CACHED: dict[str, tuple] = {}


def load_config(name: str = "gemma_two_stage") -> dict:
    from .config_io import load_config as _load

    return _load(name)


def solve(
    clue: str,
    enumeration: str | None = None,
    config: str | dict = "gemma_two_stage",
    verbose: bool = True,
):
    """Solve one clue. Convenience wrapper for notebooks and the REPL.

    Models are cached across calls keyed on the config, so an interactive
    session pays the load cost once rather than per clue.
    """
    import sys

    from .adapters import build_generator, build_scorer, describe_models
    from .config_io import load_config as _load
    from .data import single_clue
    from .pipeline import apply_filters, baseline_candidate, print_stage_report, select_best
    from .pipeline import FilterStats

    cfg = _load(config) if isinstance(config, str) else config
    key = repr(sorted(cfg.get("generator", {}).items())) + repr(sorted(cfg.get("scorer", {}).items()))
    if key not in _CACHED:
        if verbose:
            print("\n".join(describe_models(cfg)))
        _CACHED[key] =(build_generator(cfg["generator"]), build_scorer(cfg["scorer"]))
    generator, scorer = _CACHED[key]

    record = single_clue(clue, enumeration=enumeration)
    generator.generate([record])
    apply_filters(record, cfg.get("filters", {}), FilterStats())
    scorer.score([record])
    chosen = select_best(record, cfg.get("selection", {}))

    if verbose:
        print_stage_report(record, chosen, baseline_candidate(record), sys.stdout)

    return record, chosen
