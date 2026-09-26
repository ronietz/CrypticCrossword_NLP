# Candidate-generate → score → select pipeline

```
clue ──▶ [1 GENERATE]  Gemma proposes k (answer, reason) candidates
                       (direct, or answer → wordplay), dedup,
                       then drop answers that cannot fit the enumeration
     ──▶ [2 SCORE]     DeBERTa scores every (clue, answer, reason) → [0,1]
     ──▶ [3 SELECT]    argmax → the returned answer + reason
```

One entry point serves one clue and ten thousand. Every run reports the pipeline
**against its own baseline** — the generator's greedy answer — on exactly the same
clues.

```bash
# is the scorer wired up correctly? do this first, it needs no generator
python code/pipeline/run_pipeline.py --verify-scorer 200

# see what the Gemma adapters actually emit (two-stage prints both stages)
python code/pipeline/run_pipeline.py --config gemma_two_stage --probe 3 --split val

# one clue, full stage-by-stage trace
python code/pipeline/run_pipeline.py --config gemma_two_stage --clue "attack general at end of month (6)"

# the real evaluation, on a GPU node
sbatch code/pipeline/run_pipeline.sbatch --config gemma_two_stage --split test --limit 1000
sbatch code/pipeline/run_pipeline.sbatch --config gemma_reasoning_answer --split test --limit 1000
```

`--config` defaults to `gemma_two_stage`.

---

## The models

| `--config` | `generator.type` | Stage | Base model | Adapter / model loaded |
|---|---|---|---|---|
| `gemma_reasoning_answer` | `gemma_direct` | candidate generation | `google/gemma-2-2b` | `ronietz/cryptic-gemma2-2b-reasoning-answer` |
| | | scoring | – | `ronietz/cryptic-deberta-large-reasoning-scorer` |
| `gemma_two_stage` | `gemma_two_stage` | answer generation | `google/gemma-2-2b` | `ronietz/cryptic-gemma2-2b-answer` |
| | | reasoning generation | `google/gemma-2-2b` (same loaded copy) | `ronietz/cryptic-gemma2-2b-wordplay` |
| | | scoring | – | `ronietz/cryptic-deberta-large-reasoning-scorer` |

The scorer is the fine-tuned **deberta-v3-large** (435M params, 24 layers,
`id2label {0: INCORRECT, 1: CORRECT}`, fp32), fed
`CLUE: … / ANSWER: … / REASONING: …`.

**Every model is loaded from its Hugging Face repo id — never from a local
directory.** Config model fields must be bare `owner/name` ids; local paths,
`{storage}` placeholders and `local;hub` lists are rejected at startup
(`paths.hub_repo_id`), as is an id that a same-named local folder would shadow.
Downloads go to the normal HF cache under `$HF_HOME`; authentication is the
normal HF environment (`huggingface-cli login` or `HF_TOKEN`). No token lives in
the code or configs. Every run prints which model serves which stage before
loading anything:

```
Models:
  Generator mode: gemma_two_stage
  Base model: google/gemma-2-2b
  Answer adapter: ronietz/cryptic-gemma2-2b-answer
  Wordplay adapter: ronietz/cryptic-gemma2-2b-wordplay
  Scorer: ronietz/cryptic-deberta-large-reasoning-scorer
```

Both modes produce the same `Candidate(answer, reason, gen_rank)` objects, so
filtering, scoring, selection, metrics, resume and `--stage score` reruns are
identical across them.

### `gemma_direct` — reasoning and answer in one decode

One LoRA adapter, prompted exactly as in training
(`finetune_gemma2_reasoning.py`: STaR source format, tokenized with BOS):

```
solve the cryptic clue. clue: <clue> ; enumeration: 4,5 ; letters: 9
response: 
```

`letters` here is the letter *count*. One greedy decode (rank 0, the baseline)
plus `num_candidates - 1` samples. Each decode is parsed by `parse.patterns`:
the text after `answer:` becomes `Candidate.answer` (spaced letters rejoined, and
word breaks restored from the enumeration when the count agrees), and everything
before it — definition **and** wordplay — becomes `Candidate.reason`. A decode
with no `answer:` gets an empty answer and is dropped by the filters. Identical
(answer, reason) decodes are collapsed, keeping the first one's rank.

### `gemma_two_stage` — answer adapter, then wordplay adapter

```
clue ─▶ answer adapter    solve the cryptic clue. / clue: … / enumeration: 4,5 / answer:
        1 greedy + k-1 samples ─▶ normalize, dedup, keep generator order (= gen_rank)
     ─▶ wordplay adapter  explain the cryptic clue. / clue: … / enumeration: 4,5 / answer: PIPE DREAM / reasoning:
        one greedy decode per unique answer ─▶ Candidate.reason = "definition: … ; wordplay: …"
     ─▶ filters ─▶ DeBERTa ─▶ selection / metrics
```

(`/` marks a newline in the real prompt; both prompts match
`finetune_gemma2_pipeline.py` and are tokenized without BOS.) The baseline is the
answer stage's rank 0 — its greedy answer — before filters and reranking. The
`letters` field of the training data is never used: it contained the answer.
Enumeration filtering stays downstream, so every unique answer is explained,
including ones the filter then drops. Wordplay generation is one batched pass
over every (clue, answer) pair in a chunk.

Settings for each stage live under `generator.answer` and `generator.wordplay`:

```bash
--set generator.answer.num_candidates=20
--set generator.answer.temperature=0.8
--set generator.wordplay.max_new_tokens=96
```

### Loading

- Base model with `AutoModelForCausalLM`, each adapter with
  `PeftModel.from_pretrained` — never merged, never retrained. Each tokenizer is
  loaded from its adapter repo. `import torch.distributed.tensor` runs before
  PEFT loads, as the cluster requires.
- Everything loads **once**, when the pipeline builds the generator and scorer —
  never per clue or per candidate. Stages naming the same `base_model` share one
  loaded copy: two-stage keeps one gemma-2-2b carrying both adapters and switches
  between them per stage.
- Prompts are left-padded; the prompt tokens are sliced off the returned
  sequence and only the new tokens are decoded.
- `google/gemma-2-2b` is **gated**: accept its licence on huggingface.co, then
  `huggingface-cli login`. Needs `pip install peft`.
- Prompt templates may use `{clue}` (as stored, enumeration inline),
  `{clue_text}` (enumeration stripped), `{enumeration}` (`(4,5)`),
  `{enumeration_bare}` (`4,5`), `{lengths}` and `{letter_count}`; the wordplay
  prompt also gets `{answer}` / `{answer_upper}`.

### Check the scorer first

```bash
python code/pipeline/run_pipeline.py --verify-scorer 200
```

Checks the scorer in isolation, needs no generator, and catches the failure mode
that does not raise: a mismatched input template produces confident nonsense
rather than an error. It reports two AUCs; the verdict it prints says how to
read them.

There is deliberately no fallback to a bare `microsoft/deberta-v3-*`: a base
model loads fine but has a random classification head, so its scores are noise.
The scorer warns on placeholder `LABEL_n` labels, reprints a banner after the
results table, and stamps `"scorer_head_untrained": true` into `metrics.json`.

---

## Reading the results

`format_metrics` prints five accuracies, and they only mean something together:

| Number | What it is | What it tells you |
|---|---|---|
| `random_from_candidates` | pick uniformly from the k candidates | the **floor**. A "reranker" that lands here has learned nothing and is being credited for the generator's list |
| `baseline_generator_top1` | the generator's own rank-0 (greedy) decode | what **"a simple LLM generator"** scores. The thing to beat |
| `baseline_after_filters` | generator top-1 among candidates that survived the enumeration filter | isolates how much of any gain is the **free length filter** rather than the scorer |
| `pipeline` | scorer argmax | the claim |
| `oracle_at_k` | was the gold answer *anywhere* in the k candidates? | the **hard ceiling**. A scorer cannot pick an answer that was never proposed |

Plus:

- **`mcnemar_p_value`** — exact paired test over the clues where the two systems
  disagree. Both systems answer the same clues, so only disagreements carry
  information; an unpaired test on two accuracy numbers throws the pairing away.
  At n=1000 a two-point gap is well inside noise, so a delta without this is not
  a result.
- **`headroom_captured`** — of the baseline's mistakes that *were* fixable (gold
  was in the list), what fraction did the scorer fix? A cleaner read on the
  scorer than raw delta, which is capped by how often the generator was already
  right.
- **`scorer.candidate_auc`** — given a correct and an incorrect candidate, how
  often does the scorer rank the correct one higher? The scorer's quality in
  isolation. 0.5 is chance; below 0.5 means the ranking is inverted.

The summary block ends with a **diagnosis line** that names the bottleneck:
oracle@k flat against baseline → work on the generator; headroom present but
delta ≤ 0 → work on the scorer; delta positive but p ≥ 0.05 → run more clues.

### Where the headroom comes from

The scorer was fine-tuned to tell a human `wordplay` annotation apart from *an
unrelated clue's* annotation. Reranking asks a harder and different question:
separate near-miss candidates for the **same** clue. That is out of distribution,
and it is the most likely reason for a disappointing delta; the fix is training
the scorer on generated hard negatives.

---

## Files

| File | Role |
|---|---|
| `run_pipeline.py` | launcher; works by path or as `python -m pipeline.run_pipeline` |
| `cli.py` | argument parsing, run directory, artifacts, `--probe`, `--verify-scorer` |
| `pipeline.py` | the three stages, filters, selection, chunked resume, startup model summary |
| `adapters.py` | `GemmaDirectGenerator`, `GemmaTwoStageGenerator` (on a shared `CausalLMRunner`), `SequenceClassificationScorer`, and the `generator.type` / `scorer.type` registry |
| `metrics.py` | accuracies, oracle@k, McNemar, Wilson CI, AUC, three figures |
| `data.py` | split loading, reproducible sampling, sharding, ad-hoc clues |
| `config_io.py` | JSON-with-comments configs, `extends`, `--set` overrides |
| `paths.py` | cluster-vs-local data/run paths, cache redirection off `$HOME`, hub-repo-id validation |
| `selftest.py` | 173 checks over the full logic, with torch stubbed and every model faked (the Gemma generators run against a fake runner); needs no GPU and no download |
| `run_pipeline.sbatch` | Slurm wrapper following the existing repo conventions |
| `config/*.json` | `default` (shared base: load settings, scorer, filters, selection) · `gemma_reasoning_answer` · `gemma_two_stage` |

Run `python code/pipeline/selftest.py` after any change. A few seconds of CPU,
though wall-clock varies with filesystem state.

---

## Tuning and ablations

Nothing model-specific is a code path. Either edit the config or override per run:

```bash
--set scorer.model=<hub-repo-id>                   # different scorer
--set scorer.batch_size=8                          # if a card OOMs
--set generator.num_candidates=20                  # gemma_direct: more candidates
--set generator.answer.num_candidates=20           # gemma_two_stage: more answers
--set generator.answer.strategy=sample             # no greedy decode (no clean baseline)
--set selection.combine=generator                  # ablation: best-ranked survivor, no scorer
--set filters.enumeration=false                    # ablation: how much is the length filter worth?
```

Decode strategies (per stage): `greedy_sample_union` (default: 1 greedy decode,
the baseline, then k−1 samples), `sample` (k samples, no well-defined baseline),
`greedy` (one decode; what the wordplay stage uses).

A new generator family needs one class in `adapters.py` plus one `GENERATORS`
entry at the bottom of that file. Nothing else moves.

### First run: `--probe`

```bash
python code/pipeline/run_pipeline.py --config gemma_reasoning_answer --probe 5 --split val
```

Dumps raw decodes and the exact prompts. For `gemma_direct`,
`generator.parse.patterns` must match what the adapter emits, and **a wrong
pattern does not raise** — every decode would get an empty answer and be
dropped. The run reports how many decodes matched, and warns loudly if none did.

---

## Scaling to 10k

- **Streaming + resume.** Each finished clue is appended to `records.jsonl`
  immediately, and a rerun with the same `--run-dir` skips what is already there.
  `studentkillable` is preemptible and the sbatch uses `--requeue`, so a killed
  job continues rather than restarting. A truncated final line from a mid-write
  kill is detected and that one clue redone.
- **Sharding.** `--shard i/n` splits the same clue selection across parallel jobs.
  Sampling happens before sharding, so 4 shards of `--limit 10000` cover exactly
  the clues one unsharded job would.
- **Two-phase.** `--stage generate` then `--stage score --candidates-from <run>`.
  Generation is by far the expensive half; this is how you evaluate a new scorer
  on an existing 10k generation for the price of the scoring alone.
- **Cost.** Generation dominates: k Gemma decodes per clue, plus (two-stage) one
  wordplay decode per unique answer. Time a `--limit 50` run on your card and
  scale from its reported clue/s; prefer 4 shards over one long job for 10k.
- **Artifacts** per run: `config.json` (resolved, after `extends` and `--set`),
  `environment.json` (versions, GPU, git commit), `records.jsonl` (every
  candidate with scores and drop reasons — a full audit trail), `metrics.json`,
  `predictions.csv`, `figures/*.png`.

## Setup on the cluster

The pipeline needs `torch`, `transformers`, `peft`, `sentencepiece`, `protobuf`,
`numpy` and `matplotlib`. `code/deberta/without-finetuning/setup_env.sh`
installs all but `peft`; add it with `pip install peft`. Run setup on the
**login node** (compute nodes' python3.12 has no `ensurepip`, so
`python3 -m venv` fails there), `huggingface-cli login` for the gated
gemma-2-2b, `mkdir -p slurm_logs`, then submit. The sbatch preflights the
dependency list and fails in seconds with the exact `pip install` command if
anything is missing.
