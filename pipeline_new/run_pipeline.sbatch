#!/bin/bash
#SBATCH --job-name=cc-pipeline
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.err
#SBATCH --partition=studentkillable
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# Runs the candidate-generate -> score -> select pipeline on a GPU node.
# Submit from the repository root:
#
#   sbatch code/pipeline/run_pipeline.sbatch --config gemma_two_stage --split test --limit 1000
#   sbatch code/pipeline/run_pipeline.sbatch --config gemma_reasoning_answer --split test --limit 1000
#   sbatch code/pipeline/run_pipeline.sbatch --config gemma_two_stage --split test --limit 10000 --shard 0/4
#
# Every argument is forwarded verbatim to run_pipeline.py, so anything the CLI
# accepts works here. See `python code/pipeline/run_pipeline.py --help`.
#
# ---------------------------------------------------------------------------
# ONE-TIME SETUP, on the LOGIN node (the compute nodes' python3.12 has no
# ensurepip, so `python3 -m venv` fails there):
#
#   cd /home/morg/NLP_2526b/$USER
#   ./setup_env.sh                      # from code/deberta/without-finetuning/
#   mkdir -p slurm_logs                 # Slurm opens the log paths BEFORE this
#                                       # script runs, so a mkdir here is too late
#
# Every model is loaded from its Hugging Face repo id (never a local directory);
# downloads land in the normal HF cache under $HF_HOME. The models per config:
#
#   gemma_reasoning_answer  google/gemma-2-2b + ronietz/cryptic-gemma2-2b-reasoning-answer
#   gemma_two_stage         google/gemma-2-2b + ronietz/cryptic-gemma2-2b-answer
#                                             + ronietz/cryptic-gemma2-2b-wordplay
#   scorer (all configs)    ronietz/cryptic-deberta-large-reasoning-scorer
#
# The pipeline needs `pip install peft`, and google/gemma-2-2b is GATED: accept its licence on huggingface.co and run `huggingface-cli login` once
# on the login node (the token lands under $HF_HOME, which this script exports).
# Never hard-code a token. To warm the cache so a job never spends its wall clock
# downloading (this fills the HF cache only; configs still name the repo ids):
#
#   source code/deberta/without-finetuning/activate_env.sh
#   HF_HOME=$NLPDIR/.cache/huggingface python -c "from huggingface_hub import snapshot_download as d; \
#     [d(r) for r in ('google/gemma-2-2b', 'ronietz/cryptic-gemma2-2b-answer', \
#      'ronietz/cryptic-gemma2-2b-wordplay', 'ronietz/cryptic-gemma2-2b-reasoning-answer', \
#      'ronietz/cryptic-deberta-large-reasoning-scorer')]"
#
# There is deliberately NO fallback to a base deberta-v3: a base model loads fine
# but its classification head is random, so its scores would be noise.
#
# Sanity-check the scorer before spending GPU-hours on a full run -- this needs no
# generator and takes under a minute:
#
#   python code/pipeline/run_pipeline.py --verify-scorer 200
#
# ---------------------------------------------------------------------------
# Sizing --time: generation dominates -- k Gemma decodes per clue, plus (two-stage)
# one wordplay decode per unique answer. Time a `--limit 50` run on your card and
# scale from its reported clue/s before submitting 10k. `studentkillable` is
# preemptible, hence
# --requeue: the pipeline appends each finished clue to records.jsonl and resumes
# from there, so a requeued job continues rather than restarting. For 10k, prefer
# 4 shards in parallel over one long job.

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_DIR"
mkdir -p slurm_logs runs

# Optional storage location for model caches and temporary files.
# Override with STORAGE_ROOT=/path/to/storage when submitting the job.
export STORAGE_ROOT="${STORAGE_ROOT:-$PROJECT_DIR}"

export HF_HOME="${HF_HOME:-$STORAGE_ROOT/.cache/huggingface}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$STORAGE_ROOT/.cache}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$STORAGE_ROOT/.cache/pip}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$STORAGE_ROOT/.cache/matplotlib}"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_PROGRESS_BARS=1
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"

module load cuda 2>/dev/null || true

VENV="${VENV:-${VIRTUAL_ENV:-$STORAGE_ROOT/venv}}"
if [ ! -x "$VENV/bin/python" ]; then
    echo "ERROR: no usable virtualenv at $VENV" >&2
    echo "       Build it on the login node first: code/deberta/without-finetuning/setup_env.sh" >&2
    echo "       or point at an existing one:  VENV=/path/to/venv sbatch ..." >&2
    exit 2
fi
source "$VENV/bin/activate"

echo "Project:    $PROJECT_DIR"
echo "Storage:    $STORAGE_ROOT"
echo "Virtualenv: $VENV"
nvidia-smi || true

# Fail in seconds with the exact fix, rather than deep inside model loading.
python - <<'PY'
import importlib.util, sys

needed = {
    "torch": "torch",
    "transformers": "transformers",
    "numpy": "numpy",
    "matplotlib": "matplotlib",
    "sentencepiece": "sentencepiece",   # deberta-v3 ships a SentencePiece spm.model
    "google.protobuf": "protobuf",      # ...and transformers needs protobuf to parse it
    "peft": "peft",                     # the Gemma generators are LoRA adapters
}
missing = sorted({pip for mod, pip in needed.items() if importlib.util.find_spec(mod) is None})
if missing:
    sys.exit("ERROR: venv missing: " + " ".join(missing) +
             "\n       " + sys.prefix + "/bin/pip install -U " + " ".join(missing))
print("Dependencies OK.")
PY

srun python code/pipeline/run_pipeline.py "$@"
