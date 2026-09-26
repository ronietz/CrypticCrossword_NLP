"""The two swappable pipeline stages, behind two narrow interfaces.

    Generator.generate(clues)   -> list[list[Candidate]]     stage 1
    Scorer.score(clue, cands)   -> fills Candidate.score     stage 2

Everything model-specific -- prompt text, decode strategy, output parsing, the
scorer's input template, which logit is the positive class -- is a config field,
never a code path. The generator is chosen by `generator.type`:

  gemma_direct     one LoRA adapter on gemma-2-2b:
                   clue -> "definition ; wordplay ; answer"
  gemma_two_stage  two LoRA adapters on one gemma-2-2b:
                   clue -> k answers, then (clue, answer) -> "definition ; wordplay"

The scorer (`scorer.type`) is the fine-tuned DeBERTa cross-encoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from typing import Protocol

from . import paths

paths.setup_environment()  # must precede the transformers import -- see paths.py

import torch  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


# --------------------------------------------------------------------------
# The unit of data that flows between the stages
# --------------------------------------------------------------------------
@dataclass
class Candidate:
    """One (answer, reason) proposal for a clue, and everything we know about it.

    `gen_rank` defines the BASELINE we have to beat: `gen_rank == 0` is the
    generator's greedy decode -- exactly what the plain model would have
    answered -- before any filtering or reranking.
    """

    answer: str
    reason: str
    raw: str
    gen_rank: int
    score: float | None = None  # scorer's P(correct); None until stage 2
    final_score: float | None = None  # what selection actually sorts on
    dropped: str | None = None  # non-None => filtered out, with the reason why

    @property
    def alive(self) -> bool:
        return self.dropped is None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Candidate":
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class ClueRecord:
    """One clue plus its candidates: the pipeline's per-example state."""

    id: str
    clue: str
    enumeration: str | None = None
    gold_answer: str | None = None
    gold_reason: str | None = None
    candidates: list[Candidate] = field(default_factory=list)
    # Set when every candidate failed the filters and they were all reinstated so
    # the clue still gets an answer. Without surfacing this, the stage report says
    # "N survived filters" for candidates that in fact all failed -- which reads
    # as a passing filter rather than a rescue.
    rescued: bool = False

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ClueRecord":
        cands = [Candidate.from_dict(c) for c in d.get("candidates", [])]
        known = {f for f in cls.__dataclass_fields__} - {"candidates"}
        return cls(candidates=cands, **{k: v for k, v in d.items() if k in known})


class Generator(Protocol):
    def generate(self, records: list[ClueRecord]) -> None:
        """Populate `record.candidates` in place for every record."""


class Scorer(Protocol):
    def score(self, records: list[ClueRecord]) -> None:
        """Set `candidate.score` in place for every surviving candidate."""


# --------------------------------------------------------------------------
# Shared helpers
# --------------------------------------------------------------------------
def pick_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _dtype_kwargs(device: torch.device, dtype: str | None) -> dict:
    """Build the from_pretrained dtype kwarg, spelled for the installed version.

    transformers >= 5 renamed `torch_dtype=` to `dtype=`. setup_env.sh installs
    >= 5, but a Colab or laptop may well have v4, and passing the wrong spelling
    is either a TypeError or -- worse in v4 -- a silently ignored kwarg that
    leaves the model in fp32 and OOMs. Inspect rather than guess.
    """
    if not dtype or dtype == "auto":
        if device.type != "cuda":
            return {}  # fp32 everywhere but CUDA; bf16 on CPU is slower, not faster
        resolved = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif dtype == "float32":
        return {}
    else:
        resolved = getattr(torch, dtype)

    import inspect

    params = inspect.signature(AutoModelForCausalLM.from_pretrained).parameters
    key = "dtype" if "dtype" in params else "torch_dtype"
    return {key: resolved}


def normalize_answer(text: str) -> str:
    """Comparison key for answers: letters and digits only, lowercased.

    Gold answers in the united dataset are lowercase and may contain spaces
    ("running buffet"). A generator may emit "RUNNING BUFFET", "running-buffet"
    or "Running Buffet." -- all four must compare equal, or the accuracy number
    is measuring formatting rather than solving.
    """
    return re.sub(r"[^a-z0-9]", "", str(text).lower())


def enumeration_lengths(enumeration: str | None) -> list[int] | None:
    """"(4,2)" -> [4, 2]. Returns None when unparseable, meaning "do not filter".

    Commas and hyphens are both word separators for length purposes: "(4-2)" is
    one hyphenated word of 4+2 letters, "(4,2)" is two words -- different
    surface forms, identical letter counts, and letter count is all we check.
    """
    if not enumeration:
        return None
    digits = re.findall(r"\d+", str(enumeration))
    if not digits:
        return None
    return [int(d) for d in digits]


def answer_lengths(answer: str) -> list[int]:
    return [len(w) for w in re.findall(r"[a-z0-9]+", str(answer).lower())]


def matches_enumeration(answer: str, enumeration: str | None) -> bool:
    """Does the answer fit the clue's letter pattern?

    A cheap, model-free filter: an answer of the wrong length is definitionally
    wrong -- it will not fit the grid. Compares the multiset-as-sequence of word
    lengths, but also accepts a total-length match, because "(4,2)" vs "(6)"
    disagreements are a known data artifact (59 rows, see the dataset README's
    enumeration_conflicts) and we would rather keep a right answer than enforce
    a word split the source data itself is inconsistent about.
    """
    expected = enumeration_lengths(enumeration)
    if expected is None:
        return True
    got = answer_lengths(answer)
    if not got:
        return False
    return got == expected or sum(got) == sum(expected)


_TRAILING_ENUMERATION = re.compile(r"\s*\(\s*[\d,\-\s]+\s*\)\s*$")


def prompt_fields(record: ClueRecord) -> dict[str, str]:
    """Every placeholder a generator `prompt` template may use.

      {clue}              the clue exactly as stored -- enumeration appended inline
      {clue_text}         the clue with its trailing "(4,5)" removed
      {enumeration}       as stored, "(4,5)"
      {enumeration_bare}  "4,5" -- what the STaR-processed Gemma training data used
      {lengths}           "4,5", digits only (hyphens become commas)
      {letter_count}      "9", the total letter count

    The Gemma adapters were trained on STaR-processed rows, which strip the
    enumeration off the clue and store it bare in its own field; a Gemma prompt
    built from {clue} would show the model "(5)" twice, which it never saw.
    """
    enumeration = record.enumeration or ""
    lengths = enumeration_lengths(record.enumeration) or []
    return {
        "clue": record.clue,
        "clue_text": _TRAILING_ENUMERATION.sub("", record.clue).strip(),
        "enumeration": enumeration,
        "enumeration_bare": enumeration.strip().strip("()").strip(),
        "lengths": ",".join(str(n) for n in lengths),
        "letter_count": str(sum(lengths)) if lengths else "",
    }


# --------------------------------------------------------------------------
# Stage 1 -- generation
# --------------------------------------------------------------------------
class _ParsedGeneratorMixin:
    """Decode -> (answer, reason) parsing and duplicate collapse.

    gemma_direct parses answer + reason out of one decode with `parse.patterns`;
    gemma_two_stage uses only the answer cleaning and the stats counters.
    """

    def _init_parser(self, parse_cfg: dict) -> None:
        self.parse_patterns = [re.compile(p) for p in parse_cfg.get("patterns", [])]
        self.strip_enumeration: bool = bool(parse_cfg.get("strip_enumeration", True))
        self.join_spaced_letters: bool = bool(parse_cfg.get("join_spaced_letters", False))
        # Diagnostic counters, surfaced by the CLI. A high unparsed count on a
        # model you believe emits reasons means the patterns are wrong -- run
        # `--probe` to see the raw decodes.
        self.stats = {"decoded": 0, "reason_parsed": 0, "unparsed": 0}

    def parse_output(self, text: str, enumeration: str | None) -> tuple[str, str]:
        """Split one raw decode into (answer, reason).

        Tries each configured regex in order, expecting named groups `answer`
        and `reason`; the first match with a non-empty answer wins. A decode no
        pattern matches is truncated or malformed: it gets answer "" (so the
        filters drop it) and keeps the whole text as its reason for the audit
        trail, rather than having rambling text scored as an "answer".
        """
        text = text.strip()
        self.stats["decoded"] += 1

        for pattern in self.parse_patterns:
            m = pattern.search(text)
            if m:
                answer = (m.groupdict().get("answer") or "").strip()
                reason = (m.groupdict().get("reason") or "").strip()
                if answer:
                    self.stats["reason_parsed"] += 1
                    return self._clean_answer(answer, enumeration), reason

        self.stats["unparsed"] += 1
        return "", text

    def _clean_answer(self, answer: str, enumeration: str | None) -> str:
        answer = answer.strip()
        if self.strip_enumeration:
            # The prompt contains the enumeration, and the model may echo it.
            # Left in place it breaks both the length filter and exact match.
            answer = _TRAILING_ENUMERATION.sub("", answer)
        answer = answer.strip().strip('."\'`,;:')
        if self.join_spaced_letters:
            answer = _join_spaced_letters(answer, enumeration)
        # Gold answers in this dataset are lowercase; normalize so the printed
        # prediction and the compared prediction are the same string.
        return answer.lower()

    def _to_candidates(self, decodes: list[str], record: ClueRecord) -> list[Candidate]:
        """Parse, then collapse duplicates while preserving generator rank.

        Sampling routinely returns the same (answer, reasoning) several times;
        without a collapse the scorer burns forward passes on repeats and the
        reported "k candidates" overstates the real diversity. The first
        occurrence keeps its place, so rank order -- the baseline -- is intact.
        """
        seen: set[tuple[str, str]] = set()
        ordered: list[Candidate] = []

        for raw in decodes:
            answer, reason = self.parse_output(raw, record.enumeration)
            key = (normalize_answer(answer), " ".join(reason.lower().split()))
            if key in seen:
                continue
            seen.add(key)
            ordered.append(Candidate(answer=answer, reason=reason, raw=raw.strip(), gen_rank=len(ordered)))

        return ordered


def _join_spaced_letters(answer: str, enumeration: str | None) -> str:
    """"P I P E D R E A M" -> "pipe dream" when the enumeration says (4,5).

    The direct Gemma adapter was trained on STaR targets, which spell the answer
    one letter per token (`format.spaced_answer`) and so lose the word breaks.
    Rejoin the letters, then restore the breaks from the enumeration when the
    letter count agrees -- gold answers carry spaces, and the scorer should see
    the answer in the same form it was trained on. Anything that is not purely
    single characters is left alone.
    """
    tokens = answer.split()
    if len(tokens) < 2 or any(len(t) != 1 for t in tokens):
        return answer
    joined = "".join(tokens)
    lengths = enumeration_lengths(enumeration)
    if lengths and len(lengths) > 1 and sum(lengths) == len(joined):
        words, pos = [], 0
        for n in lengths:
            words.append(joined[pos : pos + n])
            pos += n
        return " ".join(words)
    return joined


# (stage label, one prompt, that stage's raw decodes) -- what `--probe` prints.
ProbeBlock = tuple[str, str, list[str]]


# --------------------------------------------------------------------------
# Stage 1 -- causal LM (Gemma + LoRA) generation
# --------------------------------------------------------------------------
class CausalLMRunner:
    """One causal-LM base with one or more PEFT/LoRA adapters attached.

    All the model-touching code for the Gemma generators lives here, so the
    generator classes above it are pure prompt/parse/dedup logic and can be
    tested with a fake runner. Loaded ONCE, when the generator is built: the base
    model from its hub repo, then each adapter from its own hub repo with
    PeftModel. Stages that name the same base share one runner -- one copy of
    gemma-2-2b carrying both two-stage adapters, switched per call with
    `set_adapter` -- rather than a second copy that doubles GPU memory. Adapters
    are never merged into the base weights, and no model is ever loaded per clue
    or per candidate.

    Two details of decoder-only generation matter here:
    prompts are LEFT-padded (so every row's new tokens start at the same
    column), and the returned sequence includes the prompt, which is sliced off
    before decoding.
    """

    def __init__(
        self,
        base_model: str,
        adapters: dict[str, str],
        device: torch.device,
        dtype: str | None = "auto",
        attn_implementation: str | None = None,
    ):
        # The cluster's torch + peft combination has needed this imported before
        # PEFT touches a Gemma model; harmless where it is not needed, absent on
        # older torch.
        try:
            import torch.distributed.tensor  # noqa: F401
        except ImportError:
            pass
        from peft import PeftModel

        self.device = device
        self.base_model = paths.hub_repo_id(base_model, "base_model")
        self.adapters = {name: paths.hub_repo_id(repo, f"{name}.adapter") for name, repo in adapters.items()}

        # Each adapter repo ships the tokenizer it was trained with; use that one.
        self.tokenizers = {name: self._load_tokenizer(repo) for name, repo in self.adapters.items()}

        kwargs = _dtype_kwargs(device, dtype)
        if attn_implementation:
            kwargs["attn_implementation"] = attn_implementation
        base = AutoModelForCausalLM.from_pretrained(self.base_model, **kwargs)

        names = list(self.adapters)
        model = PeftModel.from_pretrained(base, self.adapters[names[0]], adapter_name=names[0])
        for name in names[1:]:
            model.load_adapter(self.adapters[name], adapter_name=name)
        model.to(device)
        model.eval()
        self.model = model
        self.active_adapter = names[0]

    @staticmethod
    def _load_tokenizer(repo: str):
        tokenizer = AutoTokenizer.from_pretrained(repo)
        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"  # keep the "answer:" cue if anything is cut
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @torch.no_grad()
    def generate(
        self,
        prompts: list[str],
        *,
        adapter: str,
        num_return_sequences: int,
        gen_kwargs: dict,
        batch_size: int,
        max_input_tokens: int,
        add_special_tokens: bool,
    ) -> list[list[str]]:
        """`num_return_sequences` decodes per prompt, prompt tokens removed."""
        if adapter != self.active_adapter:
            self.model.set_adapter(adapter)
            self.active_adapter = adapter
        tokenizer = self.tokenizers[adapter]

        out: list[list[str]] = []
        n = num_return_sequences
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            inputs = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_input_tokens,
                # MUST match training: the answer/wordplay adapters were trained
                # without BOS, the direct adapter with it.
                add_special_tokens=add_special_tokens,
            ).to(self.device)
            inputs.pop("token_type_ids", None)

            sequences = self.model.generate(
                **inputs,
                num_return_sequences=n,
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs,
            )
            # Left padding puts every prompt's end at the same column.
            new_tokens = sequences[:, inputs["input_ids"].shape[1] :]
            decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            # n sequences per input, flattened input-major.
            out.extend(decoded[i * n : (i + 1) * n] for i in range(len(batch)))
        return out


class CausalStage:
    """One (base model, adapter) pair plus its prompt and decode settings.

    Built from a config block holding `base_model` and `adapter` -- both hub repo
    ids, validated here so a bad config fails before any download starts.

    strategy:
      greedy_sample_union  DEFAULT. One greedy decode, then num_candidates-1
                           samples. The greedy decode comes first, so rank 0 is
                           exactly what the plain model would answer: the honest
                           baseline.
      sample               num_candidates samples. Diverse, but no well-defined
                           top-1 to use as a baseline.
      greedy               one greedy decode (num_candidates must be 1).
    """

    STRATEGIES = ("greedy_sample_union", "sample", "greedy")

    def __init__(self, cfg: dict, adapter_name: str):
        self.adapter_name = adapter_name
        self.base_model: str = paths.hub_repo_id(cfg["base_model"], f"{adapter_name}.base_model")
        self.adapter: str = paths.hub_repo_id(cfg["adapter"], f"{adapter_name}.adapter")
        self.prompt_template: str = cfg["prompt"]
        self.num_candidates: int = int(cfg.get("num_candidates", 1))
        self.strategy: str = cfg.get("strategy", "greedy_sample_union")
        self.temperature = float(cfg.get("temperature", 1.0))
        self.top_p = float(cfg.get("top_p", 0.95))
        self.max_new_tokens = int(cfg.get("max_new_tokens", 64))
        self.max_input_tokens = int(cfg.get("max_input_tokens", 256))
        self.batch_size = int(cfg.get("batch_size", 8))
        self.add_special_tokens = bool(cfg.get("add_special_tokens", True))

        if self.strategy not in self.STRATEGIES:
            raise ValueError(
                f"unknown causal-LM strategy {self.strategy!r} (expected one of {self.STRATEGIES})"
            )
        if self.strategy == "greedy" and self.num_candidates != 1:
            raise ValueError("strategy='greedy' yields one decode; set num_candidates=1")

    def decode_passes(self) -> list[tuple[int, dict]]:
        """(num_return_sequences, generate kwargs) per pass, merged in order."""
        greedy = dict(max_new_tokens=self.max_new_tokens, do_sample=False)
        sampling = dict(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            temperature=self.temperature,
            top_p=self.top_p,
        )
        if self.strategy == "greedy":
            return [(1, greedy)]
        if self.strategy == "sample":
            return [(self.num_candidates, sampling)]
        passes = [(1, greedy)]
        if self.num_candidates > 1:
            passes.append((self.num_candidates - 1, sampling))
        return passes

    def run(self, runner, prompts: list[str]) -> list[list[str]]:
        """Every pass over every prompt; per-prompt decodes in pass order."""
        collected: list[list[str]] = [[] for _ in prompts]
        for n, gen_kwargs in self.decode_passes():
            decodes = runner.generate(
                prompts,
                adapter=self.adapter_name,
                num_return_sequences=n,
                gen_kwargs=gen_kwargs,
                batch_size=self.batch_size,
                max_input_tokens=self.max_input_tokens,
                add_special_tokens=self.add_special_tokens,
            )
            for acc, texts in zip(collected, decodes):
                acc.extend(texts)
        return collected

    def build_prompt(self, record: ClueRecord, **extra: str) -> str:
        return self.prompt_template.format(**prompt_fields(record), **extra)


def _build_runners(
    cfg: dict, stages: list[CausalStage], device: torch.device | None, runner=None
) -> dict[str, CausalLMRunner]:
    """One runner per distinct base model, loaded now; stage name -> its runner.

    `runner`, when given, serves every stage -- the seam the selftest uses to
    substitute a fake without loading weights.
    """
    if runner is not None:
        return {s.adapter_name: runner for s in stages}
    by_base: dict[str, list[CausalStage]] = {}
    for stage in stages:
        by_base.setdefault(stage.base_model, []).append(stage)
    runners: dict[str, CausalLMRunner] = {}
    for base, group in by_base.items():
        shared = CausalLMRunner(
            base_model=base,
            adapters={s.adapter_name: s.adapter for s in group},
            device=device or pick_device(cfg.get("device", "auto")),
            dtype=cfg.get("dtype", "auto"),
            attn_implementation=cfg.get("attn_implementation"),
        )
        runners.update({s.adapter_name: shared for s in group})
    return runners


def describe_models(cfg: dict) -> list[str]:
    """Which model runs at which stage, read straight from the config.

    Printed at startup so a log always states what ran. Deliberately built from
    the config, not from loaded objects: the config is the source of truth, and
    this works before (or without) loading anything.
    """
    gen, scorer = cfg.get("generator"), cfg.get("scorer")
    lines = []
    if gen is not None:
        mode = gen.get("type")
        lines.append(f"Generator mode: {mode}")
        if mode == "gemma_direct":
            lines.append(f"Base model: {gen.get('base_model')}")
            lines.append(f"Generator adapter: {gen.get('adapter')}")
        elif mode == "gemma_two_stage":
            answer, wordplay = gen.get("answer", {}), gen.get("wordplay", {})
            if answer.get("base_model") == wordplay.get("base_model"):
                lines.append(f"Base model: {answer.get('base_model')}")
            else:
                lines.append(f"Answer base model: {answer.get('base_model')}")
                lines.append(f"Wordplay base model: {wordplay.get('base_model')}")
            lines.append(f"Answer adapter: {answer.get('adapter')}")
            lines.append(f"Wordplay adapter: {wordplay.get('adapter')}")
    if scorer is not None:
        lines.append(f"Scorer: {scorer.get('model')}")
    return lines


class GemmaDirectGenerator(_ParsedGeneratorMixin):
    """One adapter emits the reasoning and the answer together.

        prompt  ->  "definition: X ; wordplay: Y ; answer: A B C"

    Parsed by `parse.patterns`: the answer becomes
    Candidate.answer and everything before it -- definition AND wordplay, so the
    scorer sees all of the model's reasoning -- becomes Candidate.reason.
    """

    def __init__(self, cfg: dict, device: torch.device | None = None, runner=None):
        self.cfg = cfg
        self.stage = CausalStage(cfg, adapter_name="direct")
        self.runner = _build_runners(cfg, [self.stage], device, runner)["direct"]
        self._init_parser(cfg.get("parse", {}))
        self.description = (
            f"{self.stage.base_model} + {self.stage.adapter} "
            f"[{self.stage.strategy}, k={self.stage.num_candidates}]"
        )

    def build_prompt(self, record: ClueRecord) -> str:
        return self.stage.build_prompt(record)

    def generate(self, records: list[ClueRecord]) -> None:
        raws = self.stage.run(self.runner, [self.build_prompt(r) for r in records])
        for record, decodes in zip(records, raws):
            record.candidates = self._to_candidates(decodes, record)

    def probe(self, records: list[ClueRecord]) -> list[tuple[ClueRecord, list[ProbeBlock]]]:
        prompts = [self.build_prompt(r) for r in records]
        raws = self.stage.run(self.runner, prompts)
        return [(r, [("generator", p, d)]) for r, p, d in zip(records, prompts, raws)]


class GemmaTwoStageGenerator(_ParsedGeneratorMixin):
    """Answer adapter proposes k answers; wordplay adapter explains each one.

        answer stage    clue            -> k answers   (deduped, rank kept)
        wordplay stage  clue + answer   -> "definition: X ; wordplay: Y"

    The wordplay text becomes Candidate.reason verbatim, so the scorer judges the
    (clue, answer, reasoning) triple exactly as for the other generators.
    gen_rank is the answer stage's order, so rank 0 -- the greedy answer -- is the
    baseline, untouched by the wordplay stage or the reranker.

    Only the answer-cleaning part of the `parse` machinery applies here (the
    answer stage emits the answer alone); it is configured by `answer.parse`.
    """

    _WORDPLAY_MARKER = re.compile(r"(?i)\bwordplay\s*:")

    def __init__(self, cfg: dict, device: torch.device | None = None, runner=None):
        self.cfg = cfg
        self.answer_stage = CausalStage(cfg["answer"], adapter_name="answer")
        self.wordplay_stage = CausalStage(cfg["wordplay"], adapter_name="wordplay")
        if sum(n for n, _ in self.wordplay_stage.decode_passes()) != 1:
            raise ValueError(
                "generator.wordplay must yield exactly one decode per answer: use "
                "strategy 'greedy', or 'sample' with num_candidates=1"
            )
        runners = _build_runners(cfg, [self.answer_stage, self.wordplay_stage], device, runner)
        self.answer_runner, self.wordplay_runner = runners["answer"], runners["wordplay"]
        self._init_parser(cfg["answer"].get("parse", {}))
        self.stats.update(answers_generated=0, unique_answers=0)
        self.description = (
            f"answer: {self.answer_stage.base_model} + {self.answer_stage.adapter} "
            f"[{self.answer_stage.strategy}, k={self.answer_stage.num_candidates}]; "
            f"wordplay: {self.wordplay_stage.base_model} + {self.wordplay_stage.adapter} "
            f"[{self.wordplay_stage.strategy}]"
        )

    def build_prompt(self, record: ClueRecord) -> str:
        return self.answer_stage.build_prompt(record)

    def build_wordplay_prompt(self, record: ClueRecord, answer: str) -> str:
        # {answer} as the pipeline holds it (lowercase); {answer_upper} is the
        # uppercase surface form the STaR-processed training data used.
        return self.wordplay_stage.build_prompt(record, answer=answer, answer_upper=answer.upper())

    def _propose(self, records: list[ClueRecord]) -> list[list[Candidate]]:
        """Answer stage: k decodes per clue -> unique answers in generator order."""
        raws = self.answer_stage.run(self.answer_runner, [self.build_prompt(r) for r in records])
        proposals = []
        for record, decodes in zip(records, raws):
            seen: set[str] = set()
            cands: list[Candidate] = []
            for raw in decodes:
                self.stats["answers_generated"] += 1
                # The adapter stops at EOS; anything past a newline is run-on.
                answer = self._clean_answer(raw.strip().partition("\n")[0], record.enumeration)
                key = normalize_answer(answer)
                if key in seen:
                    continue
                seen.add(key)
                cands.append(Candidate(answer=answer, reason="", raw=raw.strip(), gen_rank=len(cands)))
            self.stats["unique_answers"] += len(cands)
            proposals.append(cands)
        return proposals

    def _explain(self, pairs: list[tuple[ClueRecord, Candidate]]) -> list[str]:
        """Wordplay stage: one decode per (clue, answer)."""
        prompts = [self.build_wordplay_prompt(r, c.answer) for r, c in pairs]
        # One batched pass over every (clue, answer) pair in the chunk.
        texts = [decodes[0].strip() for decodes in self.wordplay_stage.run(self.wordplay_runner, prompts)]
        for text in texts:
            self.stats["decoded"] += 1
            key = "reason_parsed" if self._WORDPLAY_MARKER.search(text) else "unparsed"
            self.stats[key] += 1
        return texts

    def generate(self, records: list[ClueRecord]) -> None:
        proposals = self._propose(records)
        # An empty answer is dropped by the filters anyway; explaining it would
        # only burn a decode.
        pairs = [(r, c) for r, cands in zip(records, proposals) for c in cands if normalize_answer(c.answer)]
        for (_, cand), text in zip(pairs, self._explain(pairs)):
            cand.reason = text
        for record, cands in zip(records, proposals):
            record.candidates = cands

    def probe(self, records: list[ClueRecord]) -> list[tuple[ClueRecord, list[ProbeBlock]]]:
        proposals = self._propose(records)
        pairs = [(r, c) for r, cands in zip(records, proposals) for c in cands if normalize_answer(c.answer)]
        texts = dict(zip((id(c) for _, c in pairs), self._explain(pairs)))
        out = []
        for record, cands in zip(records, proposals):
            blocks: list[ProbeBlock] = [("answer", self.build_prompt(record), [c.raw for c in cands])]
            explained = [c for c in cands if id(c) in texts]
            if explained:
                blocks.append((
                    "wordplay",
                    self.build_wordplay_prompt(record, explained[0].answer),
                    [f"{c.answer.upper()} -> {texts[id(c)]}" for c in explained],
                ))
            out.append((record, blocks))
        return out


# --------------------------------------------------------------------------
# Stage 2 -- scoring
# --------------------------------------------------------------------------
class SequenceClassificationScorer:
    """Cross-encoder verifier: P(reasoning is correct | clue, answer, reason).

    Wraps the fine-tuned deberta-v3 classifier from
    `code/DeBERTa_small_Wordplay_Reasoning_Scorer_with_finetuning.ipynb`, using
    the same `CLUE:/ANSWER:/REASONING:` template it was trained on. That
    template is a config field, and it MUST match training -- a scorer fed a
    layout it never saw produces confident nonsense, not an error.
    """

    def __init__(self, cfg: dict, device: torch.device | None = None):
        self.cfg = cfg
        self.device = device or pick_device(cfg.get("device", "auto"))
        self.model_path = paths.hub_repo_id(cfg["model"], "scorer.model")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path, **_dtype_kwargs(self.device, cfg.get("dtype", "auto"))
        )
        self.model.to(self.device)
        self.model.eval()

        self.template: str = cfg.get(
            "template", "CLUE: {clue}\nANSWER: {answer}\nREASONING: {reason}"
        )
        self.max_length: int = int(cfg.get("max_length", 256))
        self.batch_size: int = int(cfg.get("batch_size", 32))
        self.positive_index = self._resolve_positive_index(cfg.get("positive_label"))
        self._warn_if_head_untrained()

        self.description = f"{self.model_path} [positive logit {self.positive_index}]"

    def _warn_if_head_untrained(self) -> None:
        """Shout if this looks like a base model with a random classifier head.

        The single most expensive failure mode in this pipeline: pointing at
        `microsoft/deberta-v3-small` instead of the fine-tuned scorer loads fine,
        runs fine, and produces scores that are pure noise. Nothing errors, and
        the resulting "the reranker doesn't help" conclusion is about the config,
        not the method. A checkpoint fine-tuned by the notebook carries
        id2label={0: INCORRECT, 1: CORRECT}; the untouched base model carries the
        LABEL_n placeholders, which is what we detect.

        The verdict is also recorded on the instance so the run's metrics.json
        carries it -- a warning scrolls out of a Slurm log, a JSON field does not.
        """
        id2label = getattr(self.model.config, "id2label", None) or {}
        placeholder = all(
            str(name).upper().startswith("LABEL_") for name in id2label.values()
        )
        self.head_looks_untrained = bool(placeholder)
        if self.head_looks_untrained:
            import warnings

            warnings.warn(
                f"\n{'!' * 78}\n"
                f"Scorer {self.model_path!r} has placeholder labels {sorted(id2label.values())},\n"
                "which means its classification head is RANDOMLY INITIALIZED and its scores\n"
                "are noise: scorer.model names a base model, not the fine-tuned scorer.\n"
                "Use --set scorer.model=ronietz/cryptic-deberta-large-reasoning-scorer\n"
                f"before reporting any number from this run.\n{'!' * 78}",
                RuntimeWarning,
                stacklevel=3,
            )

    def _resolve_positive_index(self, configured: str | int | None) -> int:
        """Find which logit means "correct".

        Reading it from the checkpoint's own label2id rather than hardcoding 1
        is what stops a silently inverted score when a future scorer is trained
        with the labels the other way round -- a bug that looks like "the
        reranker is slightly worse than baseline" rather than like a bug.
        """
        label2id = getattr(self.model.config, "label2id", None) or {}
        if isinstance(configured, int):
            return configured
        if isinstance(configured, str):
            if configured in label2id:
                return int(label2id[configured])
            raise ValueError(
                f"positive_label {configured!r} not in checkpoint labels {sorted(label2id)}"
            )
        for name in ("CORRECT", "correct", "LABEL_1", "entailment"):
            if name in label2id:
                return int(label2id[name])
        num_labels = int(getattr(self.model.config, "num_labels", 2))
        return 1 if num_labels > 1 else 0

    def build_input(self, record: ClueRecord, cand: Candidate) -> str:
        return self.template.format(
            clue=record.clue,
            answer=cand.answer,
            reason=cand.reason,
            enumeration=record.enumeration or "",
        )

    @torch.no_grad()
    def score(self, records: list[ClueRecord]) -> None:
        # Flatten across clues before batching. Scoring per-clue would leave
        # most batches at k rows (~12), which underuses the GPU by an order of
        # magnitude at 10k-clue scale.
        flat: list[tuple[Candidate, str]] = [
            (cand, self.build_input(record, cand))
            for record in records
            for cand in record.candidates
            if cand.alive
        ]
        if not flat:
            return

        for start in range(0, len(flat), self.batch_size):
            chunk = flat[start : start + self.batch_size]
            inputs = self.tokenizer(
                [text for _, text in chunk],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            logits = self.model(**inputs).logits.float()
            if logits.shape[-1] == 1:  # a regression-head scorer
                probs = torch.sigmoid(logits[:, 0])
            else:
                probs = torch.softmax(logits, dim=-1)[:, self.positive_index]

            for (cand, _), p in zip(chunk, probs.cpu().tolist()):
                cand.score = float(p)


# --------------------------------------------------------------------------
# Registry: `generator.type` / `scorer.type` -> class
# --------------------------------------------------------------------------
GENERATORS = {
    "gemma_direct": GemmaDirectGenerator,
    "gemma_two_stage": GemmaTwoStageGenerator,
}
SCORERS = {"sequence_classification": SequenceClassificationScorer}


def build_generator(cfg: dict, device: torch.device | None = None) -> Generator:
    kind = cfg.get("type")
    if kind not in GENERATORS:
        raise ValueError(f"generator.type must be one of {sorted(GENERATORS)}, got {kind!r}")
    return GENERATORS[kind](cfg, device)


def build_scorer(cfg: dict, device: torch.device | None = None) -> Scorer:
    kind = cfg.get("type", "sequence_classification")
    if kind not in SCORERS:
        raise ValueError(f"scorer.type must be one of {sorted(SCORERS)}, got {kind!r}")
    return SCORERS[kind](cfg, device)
