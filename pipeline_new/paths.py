"""Path handling and environment setup for the pipeline.

Repository-relative paths are used by default. Cache and storage locations can
be overridden with environment variables such as STORAGE_ROOT and RUNS_ROOT.

Model identifiers are Hugging Face repository IDs rather than local paths.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ---------------------------------------------------------------- repo layout
# paths.py lives at <repo>/code/pipeline/paths.py, so the repo root is 2 up.
REPO_ROOT = Path(__file__).resolve().parents[2]
PIPELINE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PIPELINE_DIR / "config"
DATASET_DIR = REPO_ROOT / "united-cryptonite-wordplay-dataset"

def storage_root() -> Path:
    """Return the storage root used for caches and temporary files.

    Set STORAGE_ROOT to override the default repository-local location.
    """
    override = os.environ.get("STORAGE_ROOT")
    if override:
        return Path(override).expanduser()
    return REPO_ROOT


def on_cluster() -> bool:
    return "SLURM_JOB_ID" in os.environ

def runs_root() -> Path:
    """Parent of the per-run output directories.

    Kept in the repo tree (runs/ is gitignored) even on the cluster, because run
    artifacts are small -- JSONL, CSV, PNG -- and you want them next to the code
    that produced them. Only model weights and HF caches go to storage.
    """
    return Path(os.environ.get("RUNS_ROOT", REPO_ROOT / "runs"))


def setup_environment() -> None:
    """Redirect every cache off $HOME. Idempotent; safe to call more than once.

    Uses setdefault throughout so an sbatch script that already exported these
    (run_pipeline.sbatch does) always wins over these defaults.
    """
    root = storage_root()
    cache = root / ".cache"

    os.environ.setdefault("HF_HOME", str(cache / "huggingface"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(root / ".ipython"))
    # A dataloader worker per tokenizer thread deadlocks on fork; the warning
    # transformers prints about it is noise in a Slurm log.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

    for key in ("HF_HOME", "PIP_CACHE_DIR", "MPLCONFIGDIR"):
        Path(os.environ[key]).mkdir(parents=True, exist_ok=True)


def substitutions() -> dict[str, str]:
    """The `{placeholders}` a config file may use in any path-valued field."""
    return {
        "repo": str(REPO_ROOT),
        "storage": str(storage_root()),
        "dataset": str(DATASET_DIR),
        "user": os.environ.get("USER", "nobody"),
    }


def expand(value: str) -> str:
    """Expand `{repo}` / `{storage}` / `{dataset}` / `{user}` plus $VARS and ~."""
    return os.path.expanduser(os.path.expandvars(value.format(**substitutions())))


_HUB_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.\-]*/[\w.\-]+$")


def hub_repo_id(spec: str, field: str = "model") -> str:
    """Validate a config model field as a Hugging Face repo id, and return it.

    Every model this pipeline loads is referenced by its hub repo id -- the id
    in the config is the single source of truth for what ran. Local paths,
    `{storage}` placeholders and `local;hub` preference lists are rejected
    rather than resolved, so a stale checkpoint on disk can never stand in for
    the published one.

    One subtlety: from_pretrained() itself prefers a local directory whose
    relative path happens to equal the repo id ("ronietz/foo" under the working
    directory). That would silently load local weights under a hub name, so it
    is refused too. Authentication uses the normal HF environment/cache
    (HF_TOKEN or `huggingface-cli login`); no token is ever read from config.
    """
    spec = str(spec).strip()
    if not _HUB_REPO_ID.match(spec) or ".." in spec:
        raise ValueError(
            f"{field}={spec!r} is not a Hugging Face repo id ('owner/name'). "
            "Models are loaded from the hub only; local paths and 'local;hub' "
            "preference lists are not supported."
        )
    if os.path.isdir(spec):
        raise ValueError(
            f"{field}={spec!r}: a local directory {os.path.abspath(spec)!r} has the same "
            "name, and from_pretrained() would load it instead of the hub repo. "
            "Run from a different directory or rename that folder."
        )
    return spec
