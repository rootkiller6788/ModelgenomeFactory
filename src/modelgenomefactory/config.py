"""配置：默认值 + 深合并 + 校验 + 两类摘要。

摘要分两种，**不能混**（`L2-moefactory` 的坑 70 就是混了）：

- `protocol_digest` 只跟**口径**有关（seq_len / batch / dtype / 容差 / 报告阈值），
  改这些会改变报告里的数字，但不会改变结构本身。
- `version_digest` 跟**观测到的对象**有关（zoo 索引 + 每个 config + 每个清单的摘要）。
  它变了说明「分析的输入变了」，这时候报告必须整体重跑。

`objectives_of` 返回空元组 —— 本项目**不优化任何东西**。这不是偷懒：一旦有了目标函数，
「哪个模型更好」就会混进一份只该回答「结构差在哪」的报告里。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .rng import canonical_json, join_digest, sha256_text

__all__ = [
    "DEFAULTS",
    "deep_merge",
    "load_config",
    "validate_config",
    "protocol_digest",
    "version_digest",
    "objectives_of",
    "expectations_of",
    "resolve_path",
]

DEFAULTS: dict[str, Any] = {
    "zoo": {"index": "data/zoo/zoo.json"},
    "dialects": ["hf", "vllm", "custom"],
    "cost": {"seq_len": 2048, "batch": 1, "dtype_bytes": 2},
    "acceptance": {
        "params_published_tolerance": 0.005,   # 公开参数量是四舍五入值，给 0.5% 容差
        "min_models": 12,
        "min_dialects_per_model": 2,
        "require_unknown_keys_empty": True,
        "require_crosschecks_all_pass": True,
        "require_diff_applies": True,
    },
    "expectations": {
        "llama_mutation_points": 1,
        "qwen_mutation_points": 1,
        "mistral_mutation_points": 1,
        "qwen_last_boundary_is_zero_diff": True,
        "min_full_isomorphic_classes": 1,
    },
    "report": {"headline_edits": 6, "max_matrix_rows": 32},
}


def deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = deep_merge(out[k], v)  # type: ignore[arg-type]
        else:
            out[k] = v
    return out


def resolve_path(root: Path | str, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else Path(root) / p


def load_config(path: Path | str | None = None, *, root: Path | str) -> dict[str, Any]:
    """读配置。附加 `_config_path` / `_root`，并在校验失败时**立刻报错**。"""
    root = Path(root)
    cfg = dict(DEFAULTS)
    if path is not None:
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        cfg = deep_merge(cfg, raw)
    else:
        default = root / "configs" / "default.json"
        if default.exists():
            cfg = deep_merge(cfg, json.loads(default.read_text(encoding="utf-8")))
        p = default
    cfg["_config_path"] = str(p)
    cfg["_root"] = str(root)
    validate_config(cfg)
    return cfg


def validate_config(cfg: Mapping[str, Any]) -> None:
    """配置校验。**宁可在这里炸，也不要跑到一半用一个荒唐的口径出报告。**"""
    errs: list[str] = []
    cost = cfg.get("cost") or {}
    for key in ("seq_len", "batch", "dtype_bytes"):
        v = cost.get(key)
        if not isinstance(v, int) or v <= 0:
            errs.append(f"cost.{key} 必须是正整数，得到 {v!r}")
    dialects = cfg.get("dialects") or []
    if not isinstance(dialects, list) or not dialects:
        errs.append("dialects 必须是非空列表")
    elif "hf" not in dialects:
        errs.append("dialects 必须包含 hf（真实发布清单只有这一种）")
    acc = cfg.get("acceptance") or {}
    tol = acc.get("params_published_tolerance")
    if not isinstance(tol, (int, float)) or not (0 <= tol <= 0.1):
        errs.append(f"acceptance.params_published_tolerance 应在 [0, 0.1]，得到 {tol!r}")
    if errs:
        raise ValueError("配置校验失败：\n  - " + "\n  - ".join(errs))


def protocol_digest(cfg: Mapping[str, Any]) -> str:
    """口径摘要：只吃影响**数字口径**的配置。"""
    keep = {
        "cost": cfg.get("cost"),
        "acceptance": {
            "params_published_tolerance": (cfg.get("acceptance") or {}).get("params_published_tolerance")
        },
        "report": cfg.get("report"),
    }
    return sha256_text(canonical_json(keep))


def version_digest(cfg: Mapping[str, Any], inputs: Iterable[tuple[str, Any]]) -> str:
    """版本摘要：吃**输入对象**的摘要（zoo 索引 + 每个 config + 每个清单）。"""
    parts = [sha256_text(canonical_json(cfg.get("acceptance") or {}))]
    for name, payload in inputs:
        parts.append(sha256_text(f"{name}:{canonical_json(payload)}"))
    return join_digest(parts)


def objectives_of(cfg: Mapping[str, Any]) -> tuple[()]:
    """本项目**没有**优化目标。测量型项目的这个函数必须返回空元组。"""
    return ()


def expectations_of(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return dict(cfg.get("expectations") or {})
