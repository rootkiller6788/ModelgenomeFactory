"""抽取血缘 DAG。

这个项目的血缘**不是数据的血缘，是结构的血缘**：一条 `diff` 边从哪两个 genome 来、
一个 genome 由哪些方言的 canonical 节点共同确认。于是有两条别处没有的图不变量：

1. **`genome:<name>` 的父节点恰好是该模型声明的全部方言的 canonical 节点。**
   这不是形式：它把「跨方言同指纹」这件事**画进了图里**。哪天某个方言被静默跳过，
   图里就少一条边 —— 而报告里的指纹表看不出任何异常。
2. **`mutations:<family>` 的父节点恰好是该家族链上相邻版本的那些 diff。**
   一个不多（没把无关的 diff 混进来）、一个不少（链上每个边界都被接住了）。

`lineage_extract.dot` 只画抽取部分（config → manifest → canonical → genome），
不画两两 diff —— 13 个模型有 78 个无序对，全画出来没人看得清。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "Node",
    "Edge",
    "Lineage",
    "COLLECT_NODE_ID",
    "ZOO_NODE_ID",
    "config_node",
    "manifest_node",
    "extract_node",
    "canonical_node",
    "genome_node",
    "diff_node",
    "build_lineage",
    "dumps",
    "pair_key",
]

ZOO_NODE_ID = "zoo"
COLLECT_NODE_ID = "collect"


def pair_key(a: str, b: str) -> str:
    """无序对的规范键。**方向由排序决定**，于是 `diff(a,b)` 与 `diff(b,a)` 共用一条记录。"""
    return "|".join(sorted((a, b)))


def config_node(name: str) -> str:
    return f"config:{name}"


def manifest_node(name: str, dialect: str) -> str:
    return f"manifest:{name}:{dialect}"


def extract_node(name: str) -> str:
    return f"extract:{name}"


def canonical_node(name: str, dialect: str) -> str:
    return f"canonical:{name}:{dialect}"


def genome_node(name: str) -> str:
    return f"genome:{name}"


def diff_node(a: str, b: str) -> str:
    return f"diff:{pair_key(a, b)}"


def mutation_node(family: str) -> str:
    return f"mutations:{family}"


@dataclass
class Node:
    id: str
    kind: str
    label: str
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "kind": self.kind, "label": self.label, "meta": self.meta}


@dataclass(frozen=True)
class Edge:
    src: str
    dst: str
    label: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "label": self.label}


class Lineage:
    """结构血缘图。`add_node` 拒重复、`add_edge` 校验两端存在。"""

    def __init__(self) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: set[tuple[str, str, str]] = set()

    def add_node(self, node_id: str, kind: str, label: str = "", **meta: Any) -> Node:
        if node_id in self._nodes:
            raise ValueError(f"节点已存在：{node_id}")
        n = Node(id=node_id, kind=kind, label=label or node_id, meta=dict(meta))
        self._nodes[node_id] = n
        return n

    def add_edge(self, src: str, dst: str, label: str = "") -> None:
        for end in (src, dst):
            if end not in self._nodes:
                raise ValueError(f"边 {src}->{dst} 的端点 {end!r} 不在图里")
        self._edges.add((src, dst, label))

    # -- 查询 --------------------------------------------------------------

    def has(self, node_id: str) -> bool:
        return node_id in self._nodes

    def node(self, node_id: str) -> Node:
        if node_id not in self._nodes:
            raise KeyError(f"没有节点 {node_id!r}")
        return self._nodes[node_id]

    @property
    def nodes(self) -> tuple[Node, ...]:
        return tuple(self._nodes[k] for k in sorted(self._nodes))

    @property
    def edges(self) -> tuple[Edge, ...]:
        return tuple(Edge(*t) for t in sorted(self._edges))

    def parents(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted({s for s, d, _ in self._edges if d == node_id}))

    def children(self, node_id: str) -> tuple[str, ...]:
        return tuple(sorted({d for s, d, _ in self._edges if s == node_id}))

    def roots(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes if not self.parents(n.id))

    def leaves(self) -> tuple[str, ...]:
        return tuple(n.id for n in self.nodes if not self.children(n.id))

    def by_kind(self, kind: str) -> tuple[Node, ...]:
        return tuple(n for n in self.nodes if n.kind == kind)

    def path_to_root(self, node_id: str) -> tuple[str, ...]:
        """任取一条到根的路径（含两端）。"""
        path = [node_id]
        seen = {node_id}
        cur = node_id
        while True:
            ps = self.parents(cur)
            if not ps:
                return tuple(path)
            nxt = ps[0]
            if nxt in seen:
                raise ValueError(f"图里有环：{path} → {nxt}")
            path.append(nxt)
            seen.add(nxt)
            cur = nxt

    def depth(self, node_id: str) -> int:
        return len(self.path_to_root(node_id)) - 1

    # -- 导出 --------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {"nodes": [n.to_dict() for n in self.nodes], "edges": [e.to_dict() for e in self.edges]}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Lineage":
        lin = cls()
        for n in d["nodes"]:
            lin.add_node(n["id"], n["kind"], n["label"], **n.get("meta", {}))
        for e in d["edges"]:
            lin.add_edge(e["src"], e["dst"], e.get("label", ""))
        return lin

    def to_dot(self, *, include: Sequence[str] | None = None) -> str:
        keep = set(include) if include is not None else set(self._nodes)
        lines = ["digraph modelgenomefactory {", '  rankdir="LR";', '  node [shape=box, fontname="Helvetica"];']
        for n in self.nodes:
            if n.id not in keep:
                continue
            lines.append(f'  "{n.id}" [label="{n.label}"];')
        for e in self.edges:
            if e.src not in keep or e.dst not in keep:
                continue
            lines.append(f'  "{e.src}" -> "{e.dst}" [label="{e.label}"];')
        lines.append("}")
        return "\n".join(lines) + "\n"

    def to_ascii(self, *, limit: int = 40) -> str:
        # `nodes` 是 property：每次访问都重新 `sorted` 一遍整个节点字典。
        # 原先这里取了三次（切片、判断、算余数），取一次就够 —— 渲染期间这张图不变。
        nodes = self.nodes
        lines: list[str] = []
        for n in nodes[:limit]:
            ps = self.parents(n.id)
            dep = "  " * min(self.depth(n.id), 6)
            head = f"{dep}{n.label}  [{n.kind}]"
            lines.append(head)
            if ps:
                lines.append(f"{dep}  ← {', '.join(ps)}")
        if len(nodes) > limit:
            lines.append(f"… 还有 {len(nodes) - limit} 个节点（用 --dot 看全图）")
        return "\n".join(lines)


def build_lineage(
    *,
    zoo_names: Sequence[str],
    dialects_by_model: Mapping[str, Sequence[str]],
    families: Mapping[str, Sequence[str]],
    n_pairs: int,
    n_models: int,
) -> Lineage:
    """搭出结构血缘 DAG。"""
    lin = Lineage()
    lin.add_node(ZOO_NODE_ID, "source", "zoo（13 个公开模型的结构摘录）", n_models=n_models)

    for name in zoo_names:
        lin.add_node(config_node(name), "config", f"config:{name}")
        lin.add_edge(ZOO_NODE_ID, config_node(name), "read")
        lin.add_node(extract_node(name), "stage", f"extract:{name}")
        lin.add_edge(config_node(name), extract_node(name), "归一化")
        for d in dialects_by_model[name]:
            lin.add_node(manifest_node(name, d), "manifest", f"manifest:{name}:{d}", dialect=d)
            lin.add_edge(ZOO_NODE_ID, manifest_node(name, d), "read")
            lin.add_node(canonical_node(name, d), "stage", f"canonical:{name}:{d}", dialect=d)
            lin.add_edge(extract_node(name), canonical_node(name, d), "config 侧")
            lin.add_edge(manifest_node(name, d), canonical_node(name, d), "清单侧")
        lin.add_node(genome_node(name), "genome", f"genome:{name}")
        for d in dialects_by_model[name]:
            lin.add_edge(canonical_node(name, d), genome_node(name), f"指纹·{d}")

    for fam, members in families.items():
        lin.add_node(mutation_node(fam), "analysis", f"mutations:{fam}", members=list(members))

    lin.add_node("matrix", "analysis", "matrix（对比矩阵）")
    for name in zoo_names:
        lin.add_edge(genome_node(name), "matrix", "行")

    lin.add_node(COLLECT_NODE_ID, "collect", "collect（收拢：矩阵 + 全部 diff + 突变时间线）")
    lin.add_edge("matrix", COLLECT_NODE_ID, "汇总")
    for fam in families:
        lin.add_edge(mutation_node(fam), COLLECT_NODE_ID, "汇总")
    return lin


def add_diff_nodes(
    lin: Lineage,
    pairs: Iterable[tuple[str, str]],
    *,
    fingerprint_of: Mapping[str, str],
) -> None:
    """把两两 diff 挂上去。父节点**恰好**是两个 genome。"""
    for a, b in pairs:
        nid = diff_node(a, b)
        if lin.has(nid):
            continue
        lin.add_node(nid, "diff", f"diff:{pair_key(a, b)}")
        lin.add_edge(genome_node(a), nid, "左")
        lin.add_edge(genome_node(b), nid, "右")
        lin.add_edge(nid, COLLECT_NODE_ID, "汇总")


def attach_mutation_parents(
    lin: Lineage,
    family: str,
    boundaries: Sequence[tuple[str, str]],
) -> None:
    """`mutations:<family>` 的父节点 = 该家族链上相邻版本的 diff 节点。"""
    for a, b in boundaries:
        lin.add_edge(diff_node(a, b), mutation_node(family), "边界")


def dumps(lin: Lineage) -> str:
    return json.dumps(lin.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def write(lin: Lineage, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dumps(lin), encoding="utf-8")
    return path
