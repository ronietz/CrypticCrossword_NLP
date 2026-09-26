"""End-to-end test of everything except the two models.

torch and transformers are stubbed, and the generator/scorer are replaced with
deterministic fakes, so this needs no GPU stack and downloads nothing. A few
seconds of CPU; wall clock varies with filesystem state. It exercises the parts
most likely to be wrong and hardest to notice: filtering, selection, resume,
metrics arithmetic, the scorer verification, the artifact writers, and the
Gemma generators' prompt/parse/dedup/two-stage control flow (behind a fake
runner, so no Gemma weights are loaded).

    python code/pipeline/selftest.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tempfile
import types
from pathlib import Path

# --------------------------------------------------------------------------
# Stub torch / transformers so adapters.py imports without a GPU stack.
# --------------------------------------------------------------------------
if "torch" not in sys.modules:
    torch = types.ModuleType("torch")

    class _Device:
        def __init__(self, kind="cpu"):
            self.type = kind

        def __repr__(self):
            return f"device({self.type})"

    torch.device = _Device
    torch.cuda = types.SimpleNamespace(
        is_available=lambda: False, is_bf16_supported=lambda: False, get_device_name=lambda i: ""
    )
    torch.backends = types.SimpleNamespace(mps=types.SimpleNamespace(is_available=lambda: False))
    torch.no_grad = lambda: (lambda fn: fn)
    torch.bfloat16 = "bfloat16"
    torch.float16 = "float16"
    torch.__version__ = "stub"
    sys.modules["torch"] = torch

    transformers = types.ModuleType("transformers")

    class _Auto:
        @staticmethod
        def from_pretrained(*a, **k):
            raise AssertionError("selftest must not load a real model")

    transformers.AutoTokenizer = _Auto
    transformers.AutoModelForCausalLM = _Auto
    transformers.AutoModelForSequenceClassification = _Auto
    transformers.__version__ = "5.17.0"   # matches what setup_env.sh installs
    sys.modules["transformers"] = transformers

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline import config_io, data, metrics as metrics_mod, pipeline  # noqa: E402
from pipeline.adapters import Candidate, ClueRecord, matches_enumeration, normalize_answer  # noqa: E402

FAILURES: list[str] = []


def check(condition: bool, label: str, detail: str = "") -> None:
    if condition:
        print(f"  ok    {label}")
    else:
        print(f"  FAIL  {label}  {detail}")
        FAILURES.append(label)


# --------------------------------------------------------------------------
# Fakes
# --------------------------------------------------------------------------
class FakeGenerator:
    """Emits the gold answer at a controlled rank, plus plausible distractors.

    `gold_rank` drives the whole test: at 0 the baseline is already right (the
    reranker can only lose), at 2 the gold answer is present but not top-1 (the
    reranker can win), at None it is absent (oracle@k must equal 0).
    """

    description = "fake-generator"

    def __init__(self, gold_rank: int | None = 2, k: int = 5, wrong_length: bool = False):
        self.gold_rank = gold_rank
        self.k = k
        self.wrong_length = wrong_length
        self.stats = {"decoded": 0, "reason_parsed": 0, "unparsed": 0}

    def generate(self, records):
        for record in records:
            cands = []
            for i in range(self.k):
                if self.gold_rank is not None and i == self.gold_rank and record.gold_answer:
                    answer = record.gold_answer
                elif self.wrong_length and i == 0:
                    answer = "z" * 39  # must be killed by the enumeration filter
                else:
                    answer = f"wrong{i}"
                cands.append(
                    Candidate(
                        answer=answer,
                        reason=f"reason for {answer}",
                        raw=answer,
                        gen_rank=i,
                    )
                )
            record.candidates = cands
            self.stats["decoded"] += self.k
            self.stats["reason_parsed"] += self.k


class OracleScorer:
    """A perfect scorer: 0.99 for the gold answer, low noise otherwise."""

    description = "fake-oracle-scorer"

    def score(self, records):
        rng = random.Random(0)
        for record in records:
            for cand in record.candidates:
                if not cand.alive:
                    continue
                right = record.gold_answer and normalize_answer(cand.answer) == normalize_answer(
                    record.gold_answer
                )
                cand.score = 0.99 if right else rng.uniform(0.0, 0.4)


class AdversarialScorer:
    """A perfectly wrong scorer: always ranks the gold answer last."""

    description = "fake-adversarial-scorer"

    def score(self, records):
        for record in records:
            for cand in record.candidates:
                if not cand.alive:
                    continue
                right = record.gold_answer and normalize_answer(cand.answer) == normalize_answer(
                    record.gold_answer
                )
                cand.score = 0.01 if right else 0.9


def fake_records(n: int = 40) -> list[ClueRecord]:
    return [
        ClueRecord(
            id=f"t-{i:03d}",
            clue=f"some cryptic clue number {i} (6)",
            enumeration="(6)",
            gold_answer="charge",
            gold_reason="CH(chapter) + ARGE",
        )
        for i in range(n)
    ]


def run_pipeline_with(gen, scorer, cfg, records, tmp: Path, **kw):
    pipeline.build_generator = lambda c, d=None: gen
    pipeline.build_scorer = lambda c, d=None: scorer
    return pipeline.run(records, cfg, tmp, print_first=0, progress_every=0, stream=_Quiet(), **kw)


class _Quiet:
    def write(self, *_a):
        pass

    def flush(self):
        pass


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------
def test_helpers():
    print("\nhelpers")
    check(normalize_answer("Running Buffet.") == "runningbuffet", "normalize_answer collapses case/space/punct")
    check(normalize_answer("hard-wood") == normalize_answer("hard wood"), "hyphen == space")
    check(matches_enumeration("charge", "(6)"), "6-letter answer fits (6)")
    check(not matches_enumeration("charger", "(6)"), "7-letter answer rejected by (6)")
    check(matches_enumeration("running buffet", "(7,6)"), "multiword fits (7,6)")
    check(matches_enumeration("runningbuffet", "(7,6)"), "total-length match accepted (data artifact)")
    check(matches_enumeration("anything", None), "no enumeration => no filtering")
    check(not matches_enumeration("", "(6)"), "empty answer never fits")


def test_config():
    print("\nconfig")
    base = config_io.load_config("default")
    check("type" not in base["generator"], "default.json is a base only: no generator type")
    direct_cfg = config_io.load_config("gemma_reasoning_answer")
    two = config_io.load_config("gemma_two_stage")
    check(direct_cfg["generator"]["attn_implementation"] == "eager",
          "extends merges shared Gemma load settings from default.json")
    check(two["scorer"]["template"] == base["scorer"]["template"], "extends inherits un-overridden fields")

    over = config_io.apply_overrides(two, ["generator.answer.num_candidates=20", "selection.combine=generator"])
    check(over["generator"]["answer"]["num_candidates"] == 20, "--set parses an int into a stage block")
    check(over["selection"]["combine"] == "generator", "--set parses a string")
    check(two["generator"]["answer"]["num_candidates"] == 12, "--set does not mutate the input config")
    check(sorted(p.stem for p in (Path(__file__).parent / "config").glob("*.json"))
          == ["default", "gemma_reasoning_answer", "gemma_two_stage"],
          "only the shared base and the two Gemma configs ship")

    direct = direct_cfg["generator"]
    check(direct["type"] == "gemma_direct", "direct config selects gemma_direct")
    check(direct["adapter"] == "ronietz/cryptic-gemma2-2b-reasoning-answer", "direct adapter repo")
    check(direct["base_model"] == "google/gemma-2-2b", "direct base model is gemma-2-2b")
    tg = two["generator"]
    check(tg["type"] == "gemma_two_stage", "two-stage config selects gemma_two_stage")
    check(tg["answer"]["adapter"] == "ronietz/cryptic-gemma2-2b-answer", "answer adapter repo")
    check(tg["wordplay"]["adapter"] == "ronietz/cryptic-gemma2-2b-wordplay", "wordplay adapter repo")
    check(tg["answer"]["base_model"] == tg["wordplay"]["base_model"] == "google/gemma-2-2b",
          "both stages name gemma-2-2b as their base")
    check("num_candidates" not in tg, "two-stage k lives under generator.answer, not top level")
    check("letters" not in tg["answer"]["prompt"], "answer prompt never uses a `letters` field")
    check(direct_cfg["scorer"] == two["scorer"], "both modes use the same scorer settings")

    from pipeline import adapters
    check(sorted(adapters.GENERATORS) == ["gemma_direct", "gemma_two_stage"],
          "exactly two generator types are registered")
    try:
        adapters.build_generator({"type": "no_such_type"})
        check(False, "unknown generator type rejected")
    except ValueError:
        check(True, "unknown generator type rejected with a clear error")

    # The fine-tuned deberta-v3-large scorer: a batch size sized for large, fp32
    # to avoid perturbing near-tied candidate scores, the training template.
    sc = two["scorer"]
    check(sc["model"] == "ronietz/cryptic-deberta-large-reasoning-scorer",
          "scorer is the published fine-tuned checkpoint", sc["model"])
    check(sc["batch_size"] <= 16, "batch size reduced for a large scorer", str(sc["batch_size"]))
    check(sc["dtype"] == "float32", "scorer pinned to fp32", str(sc["dtype"]))
    check(sc["max_length"] <= 512, "max_length within the checkpoint's position embeddings")
    check(sc["template"] == "CLUE: {clue}\nANSWER: {answer}\nREASONING: {reason}",
          "scorer template matches the fine-tuning layout")
    check(sc.get("positive_label") is None, "positive class read from the checkpoint's label2id")


def test_hub_only_models():
    """Every model is a hub repo id; nothing trained can load from a local dir."""
    print("\nhub-only model ids")
    import os
    from pipeline import adapters
    from pipeline.paths import hub_repo_id

    # Every model field in every shipped config is a plain repo id.
    fields = []
    for name in ("default", "gemma_reasoning_answer", "gemma_two_stage"):
        cfg = config_io.load_config(name)
        gen = cfg["generator"]
        blocks = [gen, gen.get("answer", {}), gen.get("wordplay", {})]
        fields += [(name, k, b[k]) for b in blocks for k in ("model", "base_model", "adapter") if k in b]
        fields.append((name, "scorer.model", cfg["scorer"]["model"]))
    bad = [f for f in fields if hub_repo_id(f[2]) != f[2] or ";" in f[2] or "{" in f[2]]
    check(not bad and len(fields) >= 8, "every config model field is a bare hub repo id", str(bad))

    for spec in ("/home/morg/models/x", "./models/foo", "{storage}/models/x;google/gemma-2-2b",
                 "C:\\models\\x", "gemma-2-2b", "a/b/c", "../x/y", ""):
        try:
            hub_repo_id(spec)
            check(False, f"rejects {spec!r}")
        except ValueError:
            check(True, f"rejects {spec!r}")

    # A local directory named like the repo would be picked up by
    # from_pretrained() in place of the hub repo -- refused.
    tmp = Path(tempfile.mkdtemp())
    cwd = os.getcwd()
    try:
        (tmp / "ronietz" / "cryptic-gemma2-2b-answer").mkdir(parents=True)
        os.chdir(tmp)
        try:
            hub_repo_id("ronietz/cryptic-gemma2-2b-answer")
            check(False, "refuses a repo id shadowed by a local directory")
        except ValueError:
            check(True, "refuses a repo id shadowed by a local directory")
    finally:
        os.chdir(cwd)
        shutil.rmtree(tmp, ignore_errors=True)

    # Model summary, straight from the config.
    lines = adapters.describe_models(config_io.load_config("gemma_reasoning_answer"))
    check(lines == ["Generator mode: gemma_direct", "Base model: google/gemma-2-2b",
                    "Generator adapter: ronietz/cryptic-gemma2-2b-reasoning-answer",
                    "Scorer: ronietz/cryptic-deberta-large-reasoning-scorer"],
          "direct-mode model summary", str(lines))
    lines = adapters.describe_models(config_io.load_config("gemma_two_stage"))
    check(lines == ["Generator mode: gemma_two_stage", "Base model: google/gemma-2-2b",
                    "Answer adapter: ronietz/cryptic-gemma2-2b-answer",
                    "Wordplay adapter: ronietz/cryptic-gemma2-2b-wordplay",
                    "Scorer: ronietz/cryptic-deberta-large-reasoning-scorer"],
          "two-stage model summary", str(lines))

    # Load-once: building the two-stage generator constructs ONE runner (one base
    # copy) carrying both adapters from their hub repos; generating over many
    # chunks and candidates constructs nothing further.
    built = []

    class CountingRunner(FakeCausalRunner):
        def __init__(self, base_model, adapters, device, dtype=None, attn_implementation=None):
            built.append((base_model, dict(adapters)))
            super().__init__({"answer": _answer_replies, "wordplay": _wordplay_replies})

    real = adapters.CausalLMRunner
    adapters.CausalLMRunner = CountingRunner
    try:
        cfg = config_io.load_config("gemma_two_stage")
        gen = adapters.GemmaTwoStageGenerator(cfg["generator"], device="cpu")
        for i in range(3):
            gen.answer_runner.reset_passes()
            gen.generate([ClueRecord(id=f"c{i}", clue="x (6)", enumeration="(6)")])
        check(built == [("google/gemma-2-2b", {"answer": "ronietz/cryptic-gemma2-2b-answer",
                                                "wordplay": "ronietz/cryptic-gemma2-2b-wordplay"})],
              "two-stage loads one gemma-2-2b with both hub adapters, once", str(built))
        check(gen.answer_runner is gen.wordplay_runner, "wordplay stage reuses the loaded model")

        built.clear()
        diff = config_io.apply_overrides(cfg, ["generator.wordplay.base_model=google/gemma-2-9b"])
        adapters.GemmaTwoStageGenerator(diff["generator"], device="cpu")
        check(len(built) == 2, "different per-stage base models get one runner each", str(built))

        built.clear()
        adapters.GemmaDirectGenerator(config_io.load_config("gemma_reasoning_answer")["generator"],
                                      device="cpu")
        check(built == [("google/gemma-2-2b", {"direct": "ronietz/cryptic-gemma2-2b-reasoning-answer"})],
              "direct mode loads gemma-2-2b + the reasoning-answer adapter", str(built))
    finally:
        adapters.CausalLMRunner = real

    # pipeline.run prints the summary before loading anything.
    import io
    buf = io.StringIO()
    tmp = Path(tempfile.mkdtemp())
    try:
        pipeline.build_generator = lambda c, d=None: FakeGenerator(gold_rank=0)
        pipeline.build_scorer = lambda c, d=None: OracleScorer()
        pipeline.run(fake_records(2), config_io.load_config("gemma_two_stage"), tmp,
                     print_first=0, progress_every=0, stream=buf)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    text = buf.getvalue()
    check("Answer adapter: ronietz/cryptic-gemma2-2b-answer" in text
          and "Scorer: ronietz/cryptic-deberta-large-reasoning-scorer" in text,
          "run prints the model summary at startup")


def test_dataset():
    print("\ndataset")
    try:
        recs = data.load_split("test", limit=25, seed=1)
    except FileNotFoundError as exc:
        check(False, "test split loads", str(exc))
        return
    check(len(recs) == 25, "limit honoured", f"got {len(recs)}")
    check(all(r.gold_answer for r in recs), "every row has a gold answer")
    check(all(r.clue for r in recs), "every row has clue text")

    a = [r.id for r in data.load_split("test", limit=25, seed=1)]
    b = [r.id for r in data.load_split("test", limit=25, seed=1)]
    c = [r.id for r in data.load_split("test", limit=25, seed=2)]
    check(a == b, "sampling is reproducible for a fixed seed")
    check(a != c, "a different seed draws a different subset")

    shards = [
        {r.id for r in data.load_split("test", limit=40, seed=1, shard=(i, 4))} for i in range(4)
    ]
    union = set().union(*shards)
    check(sum(len(s) for s in shards) == 40, "shards partition without loss")
    check(len(union) == 40, "shards do not overlap")
    check(union == set(a) | {r.id for r in data.load_split("test", limit=40, seed=1)} - set(a),
          "sharded coverage equals unsharded coverage")

    adhoc = data.single_clue("attack general at end of month (6)")
    check(adhoc.enumeration == "(6)", "enumeration inferred from clue text", adhoc.enumeration or "")

    try:
        data.load_split("test", limit=5, require_reason=True)
        check(False, "require_reason on test raises")
    except ValueError:
        check(True, "require_reason on test raises a clear error (no annotations there)")


def test_filters():
    print("\nfilters")
    cfg = config_io.load_config("gemma_two_stage")
    stats = pipeline.FilterStats()
    rec = ClueRecord(id="x", clue="c (6)", enumeration="(6)", gold_answer="charge")
    rec.candidates = [
        Candidate("charge", "r", "charge", 0),
        Candidate("charger", "r", "charger", 1),
        Candidate("", "r", "", 2),
        Candidate("ch", "r", "ch", 3),
    ]
    pipeline.apply_filters(rec, cfg["filters"], stats)
    alive = [c.answer for c in rec.candidates if c.alive]
    check(alive == ["charge"], "wrong-length, empty and short answers dropped", str(alive))
    check(stats.dropped_enumeration == 2, "enumeration drops counted", str(stats.as_dict()))

    # Every candidate bad -> must rescue, or the clue vanishes from the metrics.
    stats2 = pipeline.FilterStats()
    rec2 = ClueRecord(id="y", clue="c (6)", enumeration="(6)", gold_answer="charge")
    rec2.candidates = [Candidate("toolongforsix", "r", "", 0), Candidate("ab", "r", "", 1)]
    pipeline.apply_filters(rec2, cfg["filters"], stats2)
    check(all(c.alive for c in rec2.candidates), "all-dropped clue is rescued, never left unanswered")
    check(stats2.rescued_clues == 1, "rescue is counted, not hidden")
    check(rec2.rescued is True, "rescue is flagged on the record, so the report can say so")
    check(ClueRecord.from_dict(rec2.to_dict()).rescued is True, "rescue flag survives JSONL round-trip")
    check(rec.rescued is False, "a clue with survivors is not marked rescued")


def test_selection():
    print("\nselection")
    rec = ClueRecord(id="z", clue="c", gold_answer="charge")
    rec.candidates = [
        Candidate("wrong", "r", "", 0, score=0.30),
        Candidate("charge", "r", "", 1, score=0.95),
    ]
    best = pipeline.select_best(rec, {"combine": "scorer"})
    check(best.answer == "charge", "scorer mode picks the highest score")

    best = pipeline.select_best(rec, {"combine": "generator"})
    check(best.answer == "wrong", "generator mode reproduces the baseline (ablation)")
    rec.candidates[0].dropped = "enumeration_(6)"
    best = pipeline.select_best(rec, {"combine": "generator"})
    check(best.answer == "charge", "generator mode takes the best-ranked SURVIVOR")

    try:
        pipeline.select_best(rec, {"combine": "no_such_mode"})
        check(False, "unknown combine mode rejected")
    except ValueError:
        check(True, "unknown combine mode rejected with a clear error")

    tie = ClueRecord(id="t", clue="c")
    tie.candidates = [
        Candidate("second", "r", "", 3, score=0.5),
        Candidate("first", "r", "", 1, score=0.5),
    ]
    check(
        pipeline.select_best(tie, {"combine": "scorer"}).answer == "first",
        "ties break toward the lower generator rank",
    )

    empty = ClueRecord(id="e", clue="c")
    check(pipeline.select_best(empty, {}) is None, "no candidates => no selection, no crash")


def test_metrics_arithmetic():
    print("\nmetrics")
    cfg = config_io.load_config("gemma_two_stage")
    tmp = Path(tempfile.mkdtemp())

    try:
        # Gold at rank 2 with a perfect scorer: baseline wrong, pipeline right.
        out = run_pipeline_with(FakeGenerator(gold_rank=2), OracleScorer(), cfg, fake_records(40), tmp / "a")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["accuracy"]["baseline_generator_top1"] == 0.0, "baseline is 0 when gold is never rank 0")
        check(m["accuracy"]["pipeline"] == 1.0, "perfect scorer recovers every clue")
        check(m["accuracy"]["oracle_at_k"] == 1.0, "oracle@k is 1 when gold is always present")
        check(m["improvement"]["rerank_wins"] == 40, "all 40 disagreements are wins")
        check(m["improvement"]["rerank_losses"] == 0, "no losses")
        check(m["improvement"]["mcnemar_p_value"] < 1e-6, "40-0 is significant")
        check(abs(m["improvement"]["headroom_captured"] - 1.0) < 1e-9, "all headroom captured")
        check(m["scorer"]["candidate_auc"] == 1.0, "AUC is 1 for a perfect scorer")
        check(m["recall_at_k"][1] == 0.0 and m["recall_at_k"][3] == 1.0, "recall@k steps up at k=3")

        # Adversarial scorer: strictly worse than baseline, and must be reported so.
        out = run_pipeline_with(FakeGenerator(gold_rank=0), AdversarialScorer(), cfg, fake_records(20), tmp / "b")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["accuracy"]["baseline_generator_top1"] == 1.0, "baseline right when gold is rank 0")
        check(m["accuracy"]["pipeline"] == 0.0, "adversarial scorer breaks every clue")
        check(m["improvement"]["delta_accuracy"] == -1.0, "delta is negative, not hidden")
        check(m["improvement"]["rerank_losses"] == 20, "losses counted")
        check(m["scorer"]["candidate_auc"] == 0.0, "AUC is 0 for an inverted scorer")

        # Gold absent: the ceiling itself must be 0.
        out = run_pipeline_with(FakeGenerator(gold_rank=None), OracleScorer(), cfg, fake_records(10), tmp / "c")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["accuracy"]["oracle_at_k"] == 0.0, "oracle@k is 0 when gold is never generated")
        check(m["accuracy"]["pipeline"] == 0.0, "cannot pick an answer that was never proposed")

        # No gold at all (the --clue case) must not divide by zero.
        plain = [ClueRecord(id="n1", clue="a clue (6)", enumeration="(6)")]
        out = run_pipeline_with(FakeGenerator(gold_rank=None), OracleScorer(), cfg, plain, tmp / "d")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["n_with_gold"] == 0, "no-gold run reports n_with_gold=0 instead of crashing")
        check("no gold" in metrics_mod.format_metrics(m).lower(), "no-gold summary says so")

        check(metrics_mod.mcnemar_exact(0, 0) == 1.0, "McNemar with no disagreements is p=1")
        check(abs(metrics_mod.mcnemar_exact(5, 5) - 1.0) < 1e-9, "McNemar symmetric case is p=1")
        check(metrics_mod.mcnemar_exact(10, 0) < 0.01, "10-0 is significant")
        lo, hi = metrics_mod.wilson_interval(50, 100)
        check(lo < 0.5 < hi and 0 <= lo and hi <= 1, "Wilson interval brackets the estimate")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_resume_and_artifacts():
    print("\nresume + artifacts")
    cfg = config_io.load_config("gemma_two_stage")
    tmp = Path(tempfile.mkdtemp())
    try:
        run_dir = tmp / "run"
        records = fake_records(20)

        # First pass over half the clues, then resume over all of them.
        run_pipeline_with(FakeGenerator(gold_rank=1), OracleScorer(), cfg, records[:10], run_dir)
        first_lines = (run_dir / "records.jsonl").read_text().count("\n")

        gen = FakeGenerator(gold_rank=1)
        out = run_pipeline_with(gen, OracleScorer(), cfg, fake_records(20), run_dir)
        check(first_lines == 10, "first pass wrote 10 records", str(first_lines))
        check(out.resumed == 10, "resume skipped the 10 finished clues", str(out.resumed))
        check(gen.stats["decoded"] == 10 * 5, "resume regenerated only the missing 10 clues")
        check(len(out.records) == 20, "resumed run returns all 20 records")
        check([r.id for r in out.records] == [f"t-{i:03d}" for i in range(20)],
              "resumed records come back in the caller's order")

        # A truncated final line (a job killed mid-write) must not poison resume.
        with open(run_dir / "records.jsonl", "a") as fh:
            fh.write('{"id": "t-999", "clue": "trunc')
        out2 = run_pipeline_with(FakeGenerator(gold_rank=1), OracleScorer(), cfg, fake_records(20), run_dir)
        check(len(out2.records) == 20, "truncated trailing line is ignored on resume")

        # --candidates-from: score an existing generation without regenerating.
        rescore_dir = tmp / "rescore"
        gen2 = FakeGenerator(gold_rank=1)
        out3 = run_pipeline_with(
            gen2, AdversarialScorer(), cfg, fake_records(20), rescore_dir,
            stage="score", candidates_from=run_dir,
        )
        check(gen2.stats["decoded"] == 0, "stage=score never invokes the generator")
        check(len(out3.records) == 20, "rescored every clue from the cached candidates")
        m = metrics_mod.compute_metrics(out3.records, cfg["selection"])
        check(m["accuracy"]["pipeline"] == 0.0, "rescoring applied the NEW scorer, not the cached scores")

        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        metrics_mod.write_metrics_json(m, run_dir / "metrics.json")
        reloaded = json.loads((run_dir / "metrics.json").read_text())
        check(reloaded["n_with_gold"] == 20, "metrics.json round-trips")
        check(
            "nan" not in (run_dir / "metrics.json").read_text().lower(),
            "metrics.json contains no bare NaN (would be invalid JSON)",
        )

        metrics_mod.write_predictions_csv(out.records, run_dir / "predictions.csv", cfg["selection"])
        rows = (run_dir / "predictions.csv").read_text().strip().split("\n")
        check(len(rows) == 21, "predictions.csv has a header plus one row per clue", str(len(rows)))

        figs = metrics_mod.write_plots(out.records, m, run_dir / "figures")
        check(len(figs) == 3, "three figures written", str([f.name for f in figs]))
        check(all(f.exists() and f.stat().st_size > 1000 for f in figs), "figures are non-trivial files")

        # The stage report must render without exploding on odd input.
        import io

        buf = io.StringIO()
        rec = out.records[0]
        pipeline.print_stage_report(rec, pipeline.select_best(rec, cfg["selection"]),
                                    pipeline.baseline_candidate(rec), buf)
        text = buf.getvalue()
        check("stage 1" in text and "stage 2" in text and "stage 3" in text, "report shows all three stages")
        check("RERANK WIN" in text, "report flags a rerank win")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_verify_scorer():
    print("\nverify-scorer")
    from pipeline import adapters, cli

    cfg = config_io.load_config("gemma_two_stage")

    class FakeVerifyScorer:
        """Separates training-style negatives well, rerank-style poorly.

        That asymmetry is the realistic case and the one the verdict text must
        call out, so it is what the test asserts on.
        """

        description = "fake-verify-scorer"

        def score(self, records):
            for r in records:
                r.candidates[0].score = 0.90   # human reasoning
                r.candidates[1].score = 0.10   # other clue's answer + reasoning
                r.candidates[2].score = 0.88   # gold answer, other reasoning: nearly tied

    real = adapters.build_scorer
    adapters.build_scorer = lambda c, d=None: FakeVerifyScorer()
    try:
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = cli.verify_scorer(cfg, count=30, seed=0)
        text = buf.getvalue()
    finally:
        adapters.build_scorer = real

    check(res["n_clues"] == 30, "verify-scorer used the requested clue count", str(res["n_clues"]))
    check(res["training_style_negative"]["pairwise_accuracy"] == 1.0,
          "training-style separation reported")
    check(res["training_style_negative"]["auc"] == 1.0, "training-style AUC computed")
    check(res["rerank_style_negative"]["pairwise_accuracy"] == 1.0,
          "rerank-style comparison reported separately")
    check("VERDICT" in text, "verify-scorer prints a verdict")
    check("training-style" in text and "rerank-style" in text, "both negative types shown")

    p = cli.build_parser()
    check(p.parse_args(["--verify-scorer"]).verify_scorer == 200, "--verify-scorer defaults to 200")
    check(p.parse_args(["--verify-scorer", "50"]).verify_scorer == 50, "--verify-scorer takes a count")
    check(p.parse_args(["--clue", "x"]).verify_scorer is None, "--verify-scorer off by default")


def test_cli():
    print("\nCLI")
    from pipeline import cli

    p = cli.build_parser()
    args = p.parse_args(["--clue", "x (6)"])
    check(args.clue == ["x (6)"] and args.stage == "all", "CLI parses a single clue")
    check(args.config == "gemma_two_stage", "default config is gemma_two_stage")
    check(cli.parse_shard("2/8") == (2, 8), "--shard parses")
    for bad in (["--clue", "a", "--split", "test"], []):
        try:
            cli.resolve_records(p.parse_args(bad))
            check(False, f"rejects {bad}")
        except SystemExit:
            check(True, f"rejects {bad or '(no input)'}")
    check(len(cli.resolve_records(p.parse_args(["--clue", "a (4)", "--clue", "b (5)"]))) == 2,
          "multiple --clue flags accepted")
    snap = cli.environment_snapshot()
    check("python" in snap and "timestamp" in snap, "environment snapshot populated")


class FakeCausalRunner:
    """Stands in for CausalLMRunner: scripted decodes, every call recorded.

    `replies[adapter]` is a function (prompt, pass_index, n) -> list of n texts,
    so a test controls exactly what each stage and each decode pass emits.
    """

    description = "fake-gemma"

    def __init__(self, replies):
        self.replies = replies
        self.calls = []
        self.pass_index = {}

    def generate(self, prompts, *, adapter, num_return_sequences, gen_kwargs, batch_size,
                 max_input_tokens, add_special_tokens):
        idx = self.pass_index.get(adapter, 0)
        self.pass_index[adapter] = idx + 1
        self.calls.append(dict(adapter=adapter, prompts=list(prompts), n=num_return_sequences,
                               gen_kwargs=dict(gen_kwargs), add_special_tokens=add_special_tokens))
        return [self.replies[adapter](p, idx, num_return_sequences) for p in prompts]

    def reset_passes(self):
        self.pass_index = {}


def _answer_replies(prompt, pass_index, n):
    # Pass 0 is the greedy decode (the baseline); pass 1 the samples, which
    # repeat the greedy answer in two spellings to exercise dedup.
    if pass_index == 0:
        return ["ALBATROSS"] * n
    return (["GANNET", "albatross", "Albatross.", "PELICANS", "GULL"] * 3)[:n]


def _wordplay_replies(prompt, pass_index, n):
    answer = prompt.split("answer: ")[1].split("\n")[0]
    return [f"definition: seabird ; wordplay: made-up for {answer}"] * n


def test_gemma_direct():
    print("\ngemma direct (reasoning + answer in one decode)")
    from pipeline.adapters import GemmaDirectGenerator

    cfg = config_io.load_config("gemma_reasoning_answer")
    decodes = [
        "definition: large seabird ; wordplay: ALBA + TROSS ; answer: A L B A T R O S S",
        "definition: large seabird ; wordplay: ALBA + TROSS ; answer: ALBATROSS",   # dup of 0
        "definition: ... ; wordplay: ... ; answer: ALBATROSS",                     # same answer, other reason
        "definition: bird ; wordplay: G + ANNET ; answer: G A N N E T",
        "definition: truncated before any answer",                                # malformed
        "definition: x ; wordplay: y ; answer: A L B A T R O S S E S",
    ]

    def replies(prompt, pass_index, n):
        return decodes[:1] if pass_index == 0 else decodes[1 : 1 + n]

    runner = FakeCausalRunner({"direct": replies})
    gen = GemmaDirectGenerator(cfg["generator"], runner=runner)
    rec = ClueRecord(id="d", clue="large seabird circles about (9)", enumeration="(9)",
                     gold_answer="albatross")
    gen.generate([rec])

    first, second = runner.calls
    check(first["n"] == 1 and first["gen_kwargs"]["do_sample"] is False,
          "greedy decode runs first (rank 0 = baseline)")
    check(second["n"] == 11 and second["gen_kwargs"]["do_sample"] is True,
          "then num_candidates-1 samples", str(second["n"]))
    check(first["add_special_tokens"] is True, "direct prompt keeps BOS, as in training")
    check(first["prompts"][0] == "solve the cryptic clue. clue: large seabird circles about ; "
          "enumeration: 9 ; letters: 9\nresponse: ", "direct prompt matches the training format",
          repr(first["prompts"][0]))

    c0 = rec.candidates[0]
    check(c0.answer == "albatross", "spaced-letter answer rejoined", c0.answer)
    check(c0.reason == "definition: large seabird ; wordplay: ALBA + TROSS",
          "reason keeps definition AND wordplay, drops the answer field", repr(c0.reason))
    check(c0.gen_rank == 0, "greedy decode is rank 0 (the baseline)")
    check(c0.raw == decodes[0], "raw decode preserved")
    got = gen.parse_output("definition: a bird ; wordplay: anagram of ... ; answer: ALBATROSS", "(9)")
    check(got == ("albatross", "definition: a bird ; wordplay: anagram of ..."),
          "plain 'answer: ALBATROSS' parses", str(got))
    check(gen.parse_output("definition: x ; wordplay: y ; answer: P I P E D R E A M", "(4,5)")[0]
          == "pipe dream", "word breaks restored from the enumeration")

    answers = [(c.answer, c.reason) for c in rec.candidates]
    check(len(answers) == len(set(answers)), "identical (answer, reason) decodes collapsed")
    check([c.gen_rank for c in rec.candidates] == list(range(len(rec.candidates))),
          "ranks are contiguous after dedup")
    malformed = [c for c in rec.candidates if c.raw == decodes[4]]
    check(malformed and malformed[0].answer == "", "decode with no answer gets an empty answer")

    stats = pipeline.FilterStats()
    pipeline.apply_filters(rec, cfg["filters"], stats)
    alive = sorted({c.answer for c in rec.candidates if c.alive})
    check(alive == ["albatross"], "enumeration + empty filters drop the rest", str(alive))
    check(malformed[0].dropped == "empty_answer", "malformed decode dropped as empty_answer")
    best = pipeline.select_best(rec, {"combine": "generator"})
    check(best is c0, "generator-mode selection picks the best-ranked survivor")

    # --probe with k>1 must parse and count EVERY decode, not just the first.
    probe_decodes = [
        "definition: large seabird ; wordplay: ALBA + TROSS ; answer: A L B A T R O S S",
        "definition: bird ; wordplay: G + ANNET ; answer: GANNET",
        "definition: truncated before any answer",
    ]

    def probe_replies(prompt, pass_index, n):
        return probe_decodes[:1] if pass_index == 0 else probe_decodes[1 : 1 + n]

    probe_runner = FakeCausalRunner({"direct": probe_replies})
    k3 = config_io.apply_overrides(cfg, ["generator.num_candidates=3"])
    clues = [ClueRecord(id=f"p{i}", clue="large seabird circles about (9)", enumeration="(9)")
             for i in range(2)]

    from pipeline import adapters, cli
    import contextlib, io

    def build(c, d=None):
        probe_runner.reset_passes()
        return GemmaDirectGenerator(c, runner=probe_runner)

    real = adapters.build_generator
    adapters.build_generator = build
    try:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.run_probe(clues, k3, 2)
        text = buf.getvalue()
    finally:
        adapters.build_generator = real

    # Both clues go through one batched probe, so each gets the same 3 decodes.
    parsed_lines = [line for line in text.splitlines() if line.startswith("PARSED [")]
    check(len(parsed_lines) == 6, "probe prints a PARSED line for every decode (2 clues x k=3)",
          str(len(parsed_lines)))
    check([line.split("]")[0] for line in parsed_lines[:3]] == ["PARSED [0", "PARSED [1", "PARSED [2"],
          "PARSED lines are numbered per decode")
    check("PARSED [1] answer='gannet' reason='definition: bird ; wordplay: G + ANNET'" in text,
          "non-first decodes are actually parsed")
    check("PARSED [2] answer='' reason='definition: truncated before any answer'" in text,
          "an unmatched decode is shown with its empty answer")
    check("4/6 decodes matched a `parse.patterns` entry" in text,
          "match count covers every decode, not just the first", text.strip().splitlines()[-1])


def test_gemma_two_stage():
    print("\ngemma two-stage (answer -> wordplay)")
    from pipeline import adapters, cli
    from pipeline.adapters import GemmaTwoStageGenerator, SequenceClassificationScorer

    cfg = config_io.load_config("gemma_two_stage")
    cfg = config_io.apply_overrides(cfg, ["generator.answer.num_candidates=6"])
    runner = FakeCausalRunner({"answer": _answer_replies, "wordplay": _wordplay_replies})
    gen = GemmaTwoStageGenerator(cfg["generator"], runner=runner)

    rec = ClueRecord(id="g", clue="bird in a gale, net off (6)", enumeration="(6)", gold_answer="gannet")
    gen.generate([rec])

    answer_calls = [c for c in runner.calls if c["adapter"] == "answer"]
    wordplay_calls = [c for c in runner.calls if c["adapter"] == "wordplay"]
    check(answer_calls[0]["prompts"][0]
          == "solve the cryptic clue.\nclue: bird in a gale, net off\nenumeration: 6\nanswer:",
          "answer prompt matches training exactly", repr(answer_calls[0]["prompts"][0]))
    check(all(c["add_special_tokens"] is False for c in runner.calls), "no BOS, as in training")
    check([c["n"] for c in answer_calls] == [1, 5], "answer stage: 1 greedy + k-1 samples")

    check([c.answer for c in rec.candidates] == ["albatross", "gannet", "pelicans", "gull"],
          "answers deduped after normalization, generator order kept",
          str([c.answer for c in rec.candidates]))
    check([c.gen_rank for c in rec.candidates] == [0, 1, 2, 3], "gen_rank is the answer-stage order")
    wp_prompts = [p for c in wordplay_calls for p in c["prompts"]]
    check(len(wp_prompts) == 4, "wordplay stage runs once per unique answer", str(len(wp_prompts)))
    check(all(c["n"] == 1 and c["gen_kwargs"]["do_sample"] is False for c in wordplay_calls),
          "one greedy wordplay decode per answer")
    check(wp_prompts[1] == "explain the cryptic clue.\nclue: bird in a gale, net off\n"
          "enumeration: 6\nanswer: GANNET\nreasoning:", "wordplay prompt matches training exactly",
          repr(wp_prompts[1]))
    check(all(c.reason == f"definition: seabird ; wordplay: made-up for {c.answer.upper()}"
              for c in rec.candidates), "each candidate carries ITS OWN wordplay as the reason")
    check(gen.stats["answers_generated"] == 6 and gen.stats["unique_answers"] == 4,
          "answer-stage counters", str(gen.stats))
    check(gen.stats["reason_parsed"] == 4, "wordplay decodes counted as reasons", str(gen.stats))

    try:
        bad = config_io.apply_overrides(cfg, ["generator.wordplay.strategy=sample",
                                              "generator.wordplay.num_candidates=3"])
        GemmaTwoStageGenerator(bad["generator"], runner=runner)
        check(False, "more than one wordplay decode per answer is rejected")
    except ValueError:
        check(True, "more than one wordplay decode per answer is rejected")

    # Full run: real filters, real selection and metrics, and the REAL scorer's
    # build_input -- only the forward pass is faked. The scorer must see exactly
    # clue + candidate answer + that candidate's generated reasoning.
    template = cfg["scorer"]["template"]

    class RecordingScorer(SequenceClassificationScorer):
        description = "recording-scorer"

        def __init__(self):
            self.template = template
            self.seen = []

        def score(self, records):
            for record in records:
                for cand in record.candidates:
                    if cand.alive:
                        self.seen.append(self.build_input(record, cand))
                        right = normalize_answer(cand.answer) == normalize_answer(record.gold_answer)
                        cand.score = 0.9 if right else 0.1

    tmp = Path(tempfile.mkdtemp())
    try:
        runner.reset_passes()
        gen = GemmaTwoStageGenerator(cfg["generator"], runner=runner)
        scorer = RecordingScorer()
        recs = [ClueRecord(id=f"g-{i}", clue="bird in a gale, net off (6)", enumeration="(6)",
                           gold_answer="gannet") for i in range(3)]

        # generate() is called once per chunk; the fake's pass counter must restart
        # each time, as a real decode pass does.
        real_generate = gen.generate

        def generate(records):
            runner.reset_passes()
            real_generate(records)

        gen.generate = generate
        out = run_pipeline_with(gen, scorer, cfg, recs, tmp / "two")
        r0 = out.records[0]
        check(scorer.seen[0] == "CLUE: bird in a gale, net off (6)\nANSWER: gannet\n"
              "REASONING: definition: seabird ; wordplay: made-up for GANNET",
              "DeBERTa input is clue + candidate answer + generated reasoning", repr(scorer.seen[0]))
        check(len(scorer.seen) == 3, "only enumeration survivors reach the scorer", str(len(scorer.seen)))
        check([c.answer for c in r0.candidates if c.alive] == ["gannet"],
              "enumeration filter still applies to two-stage candidates")
        base = pipeline.baseline_candidate(r0)
        check(base.answer == "albatross" and base.gen_rank == 0,
              "baseline = answer stage's top-1, before filters and reranking")
        m = metrics_mod.compute_metrics(out.records, cfg["selection"])
        check(m["accuracy"]["baseline_generator_top1"] == 0.0, "baseline accuracy from rank 0")
        check(m["accuracy"]["pipeline"] == 1.0 and m["improvement"]["rerank_wins"] == 3,
              "reranked pick scored against the baseline")
        check(m["accuracy"]["oracle_at_k"] == 1.0, "oracle@k sees the two-stage pool")

        # Scorer-only rerun over the two-stage generation: generator never runs,
        # generated reasons come back from records.jsonl intact.
        runner.calls.clear()
        rescorer = RecordingScorer()
        out2 = run_pipeline_with(gen, rescorer, cfg, [ClueRecord.from_dict(r.to_dict()) for r in recs],
                                 tmp / "rescore", stage="score", candidates_from=tmp / "two")
        check(not runner.calls, "stage=score reuses two-stage candidates without generating")
        check(rescorer.seen == scorer.seen, "cached generated reasoning reaches the new scorer unchanged")
        check(len(out2.records) == 3, "every clue rescored")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # --probe shows both stages.
    runner.reset_passes()
    real = adapters.build_generator
    adapters.build_generator = lambda c, d=None: GemmaTwoStageGenerator(c, runner=runner)
    try:
        import contextlib, io

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            cli.run_probe([rec], cfg, 1)
        text = buf.getvalue()
    finally:
        adapters.build_generator = real
    check("[answer] PROMPT" in text and "[wordplay] PROMPT" in text, "--probe prints both stages")
    check("GANNET -> definition: seabird" in text, "--probe pairs each answer with its wordplay")


def test_causal_stage():
    print("\ncausal-LM decode passes")
    from pipeline.adapters import CausalStage

    base = {"base_model": "google/gemma-2-2b", "adapter": "org/adapter", "prompt": "{clue}"}
    passes = CausalStage({**base, "num_candidates": 8}, "a").decode_passes()
    check([n for n, _ in passes] == [1, 7], "greedy_sample_union: 1 greedy + k-1 samples")
    check("temperature" not in passes[0][1], "no sampling knobs on the greedy pass")
    check([n for n, _ in CausalStage({**base, "num_candidates": 1}, "a").decode_passes()] == [1],
          "k=1 union is a single greedy decode")
    check([n for n, _ in CausalStage({**base, "strategy": "sample", "num_candidates": 5}, "a")
           .decode_passes()] == [5], "sample strategy is one pass of k")
    for bad in ({"strategy": "no_such_strategy"}, {"strategy": "greedy", "num_candidates": 3}):
        try:
            CausalStage({**base, **bad}, "a")
            check(False, f"rejects {bad}")
        except ValueError:
            check(True, f"rejects {bad}")


def main() -> int:
    print("=" * 70)
    print("pipeline selftest (torch/transformers stubbed, models faked)")
    print("=" * 70)
    for test in (
        test_helpers,
        test_config,
        test_dataset,
        test_filters,
        test_selection,
        test_metrics_arithmetic,
        test_resume_and_artifacts,
        test_verify_scorer,
        test_cli,
        test_causal_stage,
        test_gemma_direct,
        test_gemma_two_stage,
        test_hub_only_models,
    ):
        test()

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S): {FAILURES}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
