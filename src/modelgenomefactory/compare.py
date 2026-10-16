"""对比矩阵与同构类。

## 同构类是「指纹有什么用」的直接答案

按 `shape_fingerprint` 分组得到**骨架类**（同一个结构、不同规模），
按 `fingerprint` 分组得到**全同构类**（逐位相同的结构）。zoo 里最有信息量的两条：

- `qwen2-7b` 与 `qwen2.5-7b` 落在**同一个全同构类**里 —— 两个不同发布版本之间
  **没有任何结构差异**，进步全部来自数据与配方。
- `llama-1-7b` 与 `llama-2-7b` 落在同一个**骨架类**但不同的全同构类里 ——
  它们是同一个结构在两个规模刻度上（其实只差 eps 与最大长度）。

这不是「顺手算出来的统计」，而是「把模型从一堆代码变成可比较的结构基因」这句话
到底兑现了没有的判据。

## 矩阵的列

列分四组：身份、结构骨架（定性）、规模（定量）、代价。**骨架列与规模列分开排**，
因为「量变 / 突变」的分界线就在这两组之间 —— 矩阵本身就该把这条线画出来。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from . import cost as cost_mod
from .genome import fingerprint, shape_fingerprint
from .schema import Genome

__all__ = ["MATRIX_COLUMNS", "SKELETON_COLUMNS", "SCALE_COLUMNS", "COST_COLUMNS", "matrix_row",
           "matrix_rows", "isomorphism_classes", "class_table"]

#: 结构骨架列（定性）—— 只有这一组变了才是突变。
SKELETON_COLUMNS = (
    "attn_variant",
    "ffn_variant",
    "norm_kind",
    "position",
    "activation",
    "tie_embeddings",
    "block_group_shape",
    "has_moe",
    "has_shared_experts",
    "flags",
)

#: 规模列（定量）—— 只有这一组变了是量变。
SCALE_COLUMNS = (
    "n_layers",
    "hidden",
    "n_heads",
    "n_kv_heads",
    "head_dim",
    "intermediate",
    "vocab",
    "max_positions",
    "n_experts",
    "top_k",
)

#: 代价列。`active_params` 是 MoE 的关键列：总参数与激活参数差一个数量级。
COST_COLUMNS = (
    "params",
    "active_params",
    "flops_per_token",
    "kv_cache_bytes",
    "weight_bytes",
)

#: 身份列。
IDENTITY_COLUMNS = ("name", "family", "chain", "release", "source_model_type", "fingerprint", "shape_fingerprint")

MATRIX_COLUMNS: tuple[str, ...] = IDENTITY_COLUMNS + SKELETON_COLUMNS + SCALE_COLUMNS + COST_COLUMNS


def _attn(genome: Genome):
    return next(n for n in genome.nodes if n.kind == "attention")


def _ffn(genome: Genome):
    return next(n for n in genome.nodes if n.kind == "ffn")


def _flags(genome: Genome) -> str:
    out: set[str] = set()
    for n in genome.nodes:
        out |= set(n.flags)
    return "+".join(sorted(out)) if out else "-"


def matrix_row(genome: Genome, zoo_meta: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """一个模型一行。列的顺序即 `MATRIX_COLUMNS`。"""
    meta = dict(zoo_meta or {})
    attn, ffn = _attn(genome), _ffn(genome)
    active = cost_mod.active_params(genome)
    return {
        "name": genome.name,
        "family": meta.get("family", ""),
        "chain": meta.get("chain", 0),
        "release": meta.get("release", ""),
        "source_model_type": genome.identity_dict().get("source_model_type", ""),
        "fingerprint": fingerprint(genome)[:12],
        "shape_fingerprint": shape_fingerprint(genome)[:12],
        "attn_variant": attn.variant,
        "ffn_variant": ffn.variant,
        "norm_kind": next(n.variant for n in genome.nodes if n.kind == "norm"),
        "position": next(n.variant for n in genome.nodes if n.kind == "position"),
        "activation": genome.topo("activation", ""),
        "tie_embeddings": genome.tensor("head.weight").tied_with is not None,
        "block_group_shape": "|".join(genome.topo("block_group_shape", ())),
        "has_moe": ffn.variant == "moe",
        "has_shared_experts": any(n.kind == "shared_expert" for n in genome.nodes),
        "flags": _flags(genome),
        "n_layers": genome.topo("n_layers", 0),
        "hidden": genome.topo("hidden", 0),
        "n_heads": attn.dim("n_heads", 0),
        "n_kv_heads": attn.dim("n_kv_heads", 0),
        "head_dim": attn.dim("head_dim", 0),
        "intermediate": ffn.dim("intermediate", 0),
        "vocab": genome.topo("vocab", 0),
        "max_positions": genome.topo("max_positions", 0),
        "n_experts": next((n.dim("n_experts", 0) for n in genome.nodes if n.kind == "router"), 0),
        "top_k": next((n.dim("top_k", 0) for n in genome.nodes if n.kind == "router"), 0),
        "params": genome.cost.params,
        "active_params": active,
        "flops_per_token": genome.cost.flops_per_token,
        "kv_cache_bytes": genome.cost.kv_cache_bytes,
        "weight_bytes": genome.cost.weight_bytes,
    }


def matrix_rows(
    genomes: Iterable[Genome], zoo_meta: Mapping[str, Mapping[str, Any]] | None = None
) -> list[dict[str, Any]]:
    meta = zoo_meta or {}
    rows = [matrix_row(g, meta.get(g.name)) for g in genomes]
    rows.sort(key=lambda r: (r["family"], r["chain"]))
    return rows


def isomorphism_classes(genomes: Sequence[Genome]) -> dict[str, dict[str, list[str]]]:
    """两类同构：按骨架指纹、按全指纹。

    返回 `{"shape": {fp: [名字...]}, "full": {fp: [名字...]}}`。**只保留成员 ≥ 1 的类**，
    但报告里只展示成员 ≥ 2 的那些 —— 成员为 1 的类没有信息量。
    """
    out: dict[str, dict[str, list[str]]] = {"shape": {}, "full": {}}
    for g in genomes:
        out["shape"].setdefault(shape_fingerprint(g), []).append(g.name)
        out["full"].setdefault(fingerprint(g), []).append(g.name)
    for bucket in out.values():
        for names in bucket.values():
            names.sort()
    return out


def class_table(genomes: Sequence[Genome], *, min_size: int = 2) -> list[dict[str, Any]]:
    """同构类的表格视图（只列成员 ≥ `min_size` 的类）。"""
    classes = isomorphism_classes(genomes)
    rows: list[dict[str, Any]] = []
    for kind in ("full", "shape"):
        for fp, names in sorted(classes[kind].items()):
            if len(names) < min_size:
                continue
            rows.append({"kind": kind, "fingerprint": fp[:12], "size": len(names), "members": names})
    return rows
