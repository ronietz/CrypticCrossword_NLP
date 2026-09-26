"""Config loading: JSON with `//` comments, an `extends` chain, and deep merge.

Plain JSON so nothing new has to be installed on the cluster (see the package
list in `code/deberta/without-finetuning/setup_env.sh` -- no pyyaml). Full-line
`//` comments are stripped before parsing, because a config that cannot explain
why `diversity_penalty` is 0.7 is a config nobody will tune correctly. YAML is
accepted too if pyyaml happens to be installed.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path

from . import paths

_LINE_COMMENT = re.compile(r"^\s*//")


def _strip_comments(text: str) -> str:
    # Full-line only. Trailing `// ...` is deliberately NOT stripped: a naive
    # trailing-comment strip corrupts any string value containing "//", such as
    # a URL, and silently produces a wrong config rather than an error.
    return "\n".join("" if _LINE_COMMENT.match(line) else line for line in text.splitlines())


def _load_one(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix in (".yaml", ".yml"):
        import yaml  # optional; only needed if you actually write YAML configs

        return yaml.safe_load(text) or {}
    try:
        return json.loads(_strip_comments(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive dict merge; scalars and lists in `override` replace outright.

    Lists replace rather than concatenate so that overriding `parse.patterns`
    means "use exactly these", not "append to whatever the base had" -- the
    latter would leave a stale pattern shadowing the new one.
    """
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve_config_path(name_or_path: str) -> Path:
    """Accept a bare config name, a name with .json, or any path."""
    direct = Path(paths.expand(name_or_path))
    if direct.exists():
        return direct
    for candidate in (
        paths.CONFIG_DIR / name_or_path,
        paths.CONFIG_DIR / f"{name_or_path}.json",
        paths.CONFIG_DIR / f"{name_or_path}.yaml",
    ):
        if candidate.exists():
            return candidate
    available = sorted(p.name for p in paths.CONFIG_DIR.glob("*.json"))
    raise FileNotFoundError(
        f"no config {name_or_path!r}. Available in {paths.CONFIG_DIR}: {available}"
    )


def load_config(name_or_path: str, _seen: set[Path] | None = None) -> dict:
    """Load a config, following its `extends` chain from the base upward."""
    path = resolve_config_path(name_or_path)
    seen = _seen or set()
    if path in seen:
        raise ValueError(f"circular `extends` involving {path}")
    seen.add(path)

    cfg = _load_one(path)
    parent_name = cfg.pop("extends", None)
    if parent_name:
        cfg = deep_merge(load_config(parent_name, seen), cfg)

    cfg.setdefault("name", path.stem)
    return cfg


def apply_overrides(cfg: dict, overrides: list[str]) -> dict:
    """Apply `--set generator.num_candidates=20` style dotted overrides.

    Lets you sweep a hyperparameter from a shell loop without writing a config
    file per value. Values are parsed as JSON when possible so `true`, `0.7` and
    `["a","b"]` arrive with the right type, falling back to string.
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set needs key=value, got {item!r}")
        dotted, _, raw = item.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw

        node = cfg
        keys = dotted.strip().split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
            if not isinstance(node, dict):
                raise ValueError(f"--set {dotted}: {key} is not a section")
        node[keys[-1]] = value
    return cfg
