"""Genome：结构基因、两级指纹、主地址空间的重建。

## 两级指纹

| 指纹 | 吃什么 | 用途 |
| --- | --- | --- |
| `shape_fingerprint` | **定性骨架**：节点路径 + 算子种类 + 变体 + 标志、张量路径 + 绑定、连接、`block_group_shape`、`activation` | 判定「是不是同构」「是不是结构突变」 |
| `fingerprint` | 骨架 + **全部规模数值**（层数 / 宽度 / 形状 / eps / theta / top-k） | 判定「是不是同一个结构」 |

于是「量变 / 突变」是可判定的，不需要拍任何阈值：

```
shape_fingerprint 相同 ⟺ 只有规模变了   （量变）
shape_fingerprint 不同 ⟺ 算子或连接变了 （突变）
```

`diff.is_mutation` 的定义就是 `left_shape_fingerprint != right_shape_fingerprint`，
而 `tests/test_diff.py` 有一条测试断言「编辑种类分类」与这个定义**永远一致** ——
两套判据必须同时成立，否则报告会说一件事、指纹说另一件事。

## 指纹里**没有**什么

`name` / `family` / `source_model_type` / `architectures` / `dialect` / `cost` 都不进指纹。
这不是省事：**同构的定义就是「改个名字还是同一个结构」**。把模型名放进指纹，
`qwen2-7b` 与 `qwen2.5-7b` 就会「不同构」—— 而它们逐位相同（见 README 附录 B）。

## 主地址空间

`Genome.primary()` 是 diff / apply 作用的唯一空间。它**不包含派生量**（`role` 由路径
唯一决定、`cost` 由整个空间决定），所以新增字段时必须同时接进 `primary()`，
否则 `apply(a, diff(a,b)) ≡ b` 会静默失效。`tests/test_genome.py` 用
`dataclasses.fields` 逐字段盯着这件事。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from . import cost as cost_mod
from .rng import canonical_json, sha256_text
from .schema import (
    SKELETON_TOPOLOGY,
    CanonicalIR,
    Cost,
    Edge,
    Genome,
    OpNode,
    RawStructure,
    TensorEntry,
    rehydrate,
)

__all__ = [
    "build",
    "skeleton",
    "shape_fingerprint",
    "fingerprint",
    "primary",
    "rebuild",
    "dump",
    "load",
    "load_dir",
    "write",
    "fingerprint_table",
]


def build(
    ir: CanonicalIR,
    raw: RawStructure,
    *,
    seq_len: int,
    batch: int,
    dtype_bytes: int,
    params_published_b: float | None = None,
) -> Genome:
    c = cost_mod.compute(
        ir,
        raw,
        seq_len=seq_len,
        batch=batch,
        dtype_bytes=dtype_bytes,
        params_published_b=params_published_b,
    )
    return Genome(
        name=ir.name,
        topology=ir.topology,
        nodes=ir.nodes,
        tensors=ir.tensors,
        edges=ir.edges,
        cost=c,
        identity=ir.identity,
        dialect=ir.dialect,
        notes=ir.notes,
    )


# ---------------------------------------------------------------- 指纹


def skeleton(g: Genome) -> dict[str, Any]:
    """定性骨架。**不含任何规模数值** —— 这就是「量变 / 突变」的分界线。"""
    return {
        "topology": {k: _jsonable(v) for k, v in g.topology if k in SKELETON_TOPOLOGY},
        "nodes": sorted((n.path, n.kind, n.variant, tuple(sorted(n.flags))) for n in g.nodes),
        "tensors": sorted((t.path, t.role, t.tied_with is not None) for t in g.tensors),
        "edges": sorted((e.src, e.dst, e.label) for e in g.edges),
    }


def shape_fingerprint(g: Genome) -> str:
    return sha256_text(canonical_json(skeleton(g)))


def fingerprint(g: Genome) -> str:
    """全指纹：骨架 + 全部规模数值。**不含 name / identity / dialect / cost。**"""
    return sha256_text(
        canonical_json(
            {
                "skeleton": skeleton(g),
                "topology": {k: _jsonable(v) for k, v in g.topology},
                "nodes": sorted(
                    (n.path, n.kind, n.variant, tuple(sorted(n.flags)), tuple(sorted(n.dims)))
                    for n in g.nodes
                ),
                "tensors": sorted((t.path, t.role, tuple(t.shape), t.tied_with) for t in g.tensors),
            }
        )
    )


def _jsonable(v: Any) -> Any:
    return list(v) if isinstance(v, tuple) else v


def fingerprint_table(genomes: Iterable[Genome]) -> list[dict[str, Any]]:
    """指纹表：模型 → (全指纹, 骨架指纹, 同构类)。"""
    out = []
    for g in genomes:
        out.append(
            {
                "name": g.name,
                "fingerprint": fingerprint(g),
                "shape_fingerprint": shape_fingerprint(g),
                "params": g.cost.params,
            }
        )
    return out


# ---------------------------------------------------------------- 主地址空间重建


def primary(g: Genome) -> dict[str, Any]:
    return g.primary()


def rebuild(g: Genome, flat: Mapping[str, Any]) -> Genome:
    """从主地址空间重建 Genome（`Diff.apply` 用）。

    代价走 `cost.derive` 重算 —— 它是地址空间的函数，必须能被复算。
    `params_formula` 与 `params_published_b` 是**外部旁证**（不来自地址空间），
    原样带过来，并在 `Cost` 的文档里写明这一点。
    """
    topology: list[tuple[str, Any]] = []
    nodes: dict[str, dict[str, Any]] = {}
    tensors: dict[str, dict[str, Any]] = {}
    edges: list[tuple[str, str, str]] = []

    for key, value in flat.items():
        if key.startswith("topology."):
            topology.append((key[len("topology.") :], value))
        elif key.startswith("nodes."):
            path, field, dim = _split_field(key[len("nodes.") :])
            slot = nodes.setdefault(path, {"dims": {}})
            if dim is None:
                slot[field] = value
            else:
                # `field` 在这里恒为 `"dims"`（它是字段名），真正的 dims 键在 `dim` 里。
                # 早先写成了 `slot["dims"][field]`，于是每个节点的 dims 都被压成一条
                # `("dims", <最后一个值>)` —— `apply(a, diff(a,b)) ≡ b` 因此静默失效，
                # 而 diff 本身看上去完全正常（26 条编辑、方向都对）。
                slot["dims"][dim] = value
        elif key.startswith("tensors."):
            path, field, dim = _split_field(key[len("tensors.") :])
            if dim is not None:
                raise ValueError(f"张量不该有 dims 字段：{key}")
            tensors.setdefault(path, {})[field] = value
        elif key.startswith("edges."):
            rest = key[len("edges.") :]
            src, _, tail = rest.partition("->")
            dst, _, label = tail.rpartition(":")
            edges.append((src, dst, label))
        else:
            raise ValueError(f"主地址空间里出现未知前缀：{key!r}")

    node_objs = tuple(
        OpNode(
            path=path,
            kind=slot["kind"],
            variant=slot["variant"],
            flags=tuple(slot.get("flags", ())),
            dims=tuple(sorted(slot["dims"].items())),
            provenance="both",
        )
        for path, slot in sorted(nodes.items())
    )
    tensor_objs = tuple(
        TensorEntry(
            path=path,
            role=_role_of(path),
            shape=tuple(slot["shape"]),
            tied_with=slot.get("tied_with"),
            provenance="both",
            source_name="",
        )
        for path, slot in sorted(tensors.items())
    )
    edge_objs = tuple(Edge(*t) for t in sorted(set(edges)))

    d = cost_mod.derive_from_flat(
        topology=topology,
        nodes=node_objs,
        tensors=tensor_objs,
        seq_len=g.cost.seq_len,
        batch=g.cost.batch,
        dtype_bytes=g.cost.dtype_bytes,
    )
    new_cost = Cost(
        params=d["params"],
        params_formula=g.cost.params_formula,
        flops_per_token=d["flops_per_token"],
        kv_cache_bytes=d["kv_cache_bytes"],
        weight_bytes=d["weight_bytes"],
        breakdown=d["breakdown"],
        params_published_b=g.cost.params_published_b,
        seq_len=g.cost.seq_len,
        batch=g.cost.batch,
        dtype_bytes=g.cost.dtype_bytes,
    )
    return Genome(
        name=g.name,
        topology=tuple(sorted(topology)),
        nodes=node_objs,
        tensors=tensor_objs,
        edges=edge_objs,
        cost=new_cost,
        identity=g.identity,
        dialect=g.dialect,
        notes=g.notes,
    )


def _split_field(rest: str) -> tuple[str, str, str | None]:
    """`blocks[*].ffn.experts[*].gate_proj.weight.shape` → (路径, 字段, dims 键)。

    路径里带 `.`，所以只能从右边切。
    """
    for field in ("kind", "variant", "flags", "role", "shape", "tied_with"):
        suffix = "." + field
        if rest.endswith(suffix):
            return rest[: -len(suffix)], field, None
    marker = ".dims."
    idx = rest.rfind(marker)
    if idx >= 0:
        return rest[:idx], "dims", rest[idx + len(marker) :]
    raise ValueError(f"认不出主地址空间键：{rest!r}")


def _role_of(path: str) -> str:
    from .dialects import role_of

    return role_of(path)


# ---------------------------------------------------------------- 落盘


def dump(g: Genome) -> dict[str, Any]:
    d = g.to_dict()
    d["fingerprint"] = fingerprint(g)
    d["shape_fingerprint"] = shape_fingerprint(g)
    return d


def load(d: Mapping[str, Any]) -> Genome:
    """从落盘的 dict 重建 Genome（**含代价**，不再重算）。

    这是 `genomes/<name>.json` 的读入口。只读命令必须能从产物重建 ——
    它们不许重跑流水线（会写盘）。
    """
    nodes = tuple(
        OpNode(
            path=n["path"],
            kind=n["kind"],
            variant=n["variant"],
            flags=tuple(n["flags"]),
            dims=tuple(sorted((k, v) for k, v in n["dims"].items())),
            provenance=n.get("provenance", "both"),
        )
        for n in d["nodes"]
    )
    tensors = tuple(
        TensorEntry(
            path=t["path"],
            role=t["role"],
            shape=tuple(t["shape"]),
            tied_with=t["tied_with"],
            provenance=t.get("provenance", "both"),
            source_name=t.get("source_name", ""),
        )
        for t in d["tensors"]
    )
    edges = tuple(Edge(e["src"], e["dst"], e["label"]) for e in d["edges"])
    c = d["cost"]
    cost = Cost(
        params=c["params"],
        params_formula=c["params_formula"],
        flops_per_token=c["flops_per_token"],
        kv_cache_bytes=c["kv_cache_bytes"],
        weight_bytes=c["weight_bytes"],
        breakdown=tuple(sorted(c["breakdown"].items())),
        params_published_b=c.get("params_published_b"),
        seq_len=c["seq_len"],
        batch=c["batch"],
        dtype_bytes=c["dtype_bytes"],
    )
    return Genome(
        name=d["name"],
        topology=tuple(
            sorted((k, rehydrate(f"topology.{k}", v)) for k, v in d["topology"].items())
        ),
        nodes=nodes,
        tensors=tensors,
        edges=edges,
        cost=cost,
        identity=tuple(sorted((k, v) for k, v in d["identity"].items())),
        dialect=d.get("dialect", "hf"),
        notes=tuple(d.get("notes", ())),
    )


def write(g: Genome, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dump(g), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_dir(dir_path: Path | str) -> dict[str, Genome]:
    dir_path = Path(dir_path)
    out: dict[str, Genome] = {}
    for p in sorted(dir_path.glob("*.json")):
        out[p.stem] = load(json.loads(p.read_text(encoding="utf-8")))
    return out


_ABSENT = object()


def same_primary(a: Genome, b: Genome) -> bool:
    """主地址空间逐位相等。`apply(a, diff(a,b)) ≡ b` 的判据就是它。"""
    return a.primary() == b.primary()


def primary_diff_keys(a: Genome, b: Genome) -> tuple[str, ...]:
    """只用于调试/测试：地址空间里不一致的键。

    必须用哨兵而不是 `.get()`：`.get()` 会把「键不存在」和「键存在但值为 None」
    看成一样，于是**恰好吞掉**「幽灵键」这一类差异 —— 曾经就是这么吞掉的。
    判据与 `same_primary` 必须一致，否则调试工具会说「没差别」而断言说「有差别」。
    """
    pa, pb = a.primary(), b.primary()
    return tuple(sorted(k for k in set(pa) | set(pb) if pa.get(k, _ABSENT) != pb.get(k, _ABSENT)))
