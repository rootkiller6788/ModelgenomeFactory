"""家族演化时间线：把「同一家族不同版本的结构演化」自动识别成**突变点**。

验收标准第二条是「同一家族不同版本的结构演化可被自动识别为突变点」。这句话要成立，
得先回答「突变」是什么。本项目的回答是**可计算的、不含阈值的**：

> 一个版本边界是突变点 ⟺ 相邻两个版本的结构**骨架指纹**不同。

骨架指纹只吃定性事实（算子种类 / 变体 / 标志 / 张量路径 / 权重绑定 / 连接），
所以「层数 32→40」「宽度 4096→5120」「词表 32000→128256」都**不是**突变 ——
它们是同一个结构在不同规模上的实例。而「MHA→GQA」「SwiGLU→MoE」「加上滑窗」
「拆掉权重绑定」是突变。

这条判据给出的结论是可以被外部核对的（见 README 附录 B）：Llama 链 4 个边界里
只有 1 个突变点，且正好落在 Llama-2 → Llama-3 那一代；Qwen 链的最后一个边界是
**零差异**。如果哪天判据写错了，这些结论会立刻变得不合理 —— 这比「阈值调得准」可靠。
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

from .diff import headline, structural_diff
from .schema import Boundary, Diff, FamilyTimeline, Genome

__all__ = ["build_timelines", "boundary_of", "classify_diff", "timeline_table"]


def classify_diff(d: Diff) -> str:
    """`same` / `scale` / `mutation` —— 只由骨架指纹决定，不看编辑条数。"""
    if d.is_empty:
        return "same"
    return "mutation" if d.is_mutation else "scale"


def boundary_of(family: str, index: int, left: Genome, right: Genome, d: Diff) -> Boundary:
    return Boundary(
        family=family,
        left=left.name,
        right=right.name,
        index=index,
        kind=classify_diff(d),
        coordinates=d.coordinates,
        n_edits=len(d.edits),
        n_scale=len(d.scale_edits()),
        n_structural=len(d.structural_edits()),
        fingerprint_changed=d.left_fingerprint != d.right_fingerprint,
        shape_fingerprint_changed=d.is_mutation,
        headline=headline(d, limit=4, verdict=False),
    )


def build_timelines(
    members_by_family: Mapping[str, Sequence[Genome]],
) -> tuple[FamilyTimeline, ...]:
    """每个家族一条时间线。成员必须已按 `chain` 排好序（顺序是数据，不是字符串序）。"""
    out: list[FamilyTimeline] = []
    for family in sorted(members_by_family):
        members = list(members_by_family[family])
        if len(members) < 2:
            out.append(
                FamilyTimeline(
                    family=family,
                    members=tuple(m.name for m in members),
                    boundaries=(),
                    note="单成员家族：没有版本边界，因此没有突变点可判。",
                )
            )
            continue
        boundaries: list[Boundary] = []
        for i in range(len(members) - 1):
            a, b = members[i], members[i + 1]
            boundaries.append(boundary_of(family, i + 1, a, b, structural_diff(a, b)))
        out.append(
            FamilyTimeline(
                family=family,
                members=tuple(m.name for m in members),
                boundaries=tuple(boundaries),
                note="",
            )
        )
    return tuple(out)


def timeline_table(timelines: Iterable[FamilyTimeline]) -> list[dict[str, Any]]:
    """时间线的表格视图（进 `mutations.json` 与报告）。"""
    rows: list[dict[str, Any]] = []
    for tl in timelines:
        for b in tl.boundaries:
            rows.append(
                {
                    "family": tl.family,
                    "index": b.index,
                    "left": b.left,
                    "right": b.right,
                    "kind": b.kind,
                    "coordinates": list(b.coordinates),
                    "n_edits": b.n_edits,
                    "n_scale": b.n_scale,
                    "n_structural": b.n_structural,
                }
            )
    return rows
