"""结构 Diff：把两个 Genome 的差异表示成一个**可以施加回去的补丁**。

README 的验收标准是「对任意两个开源模型输出可复现的结构 Diff」。这句话要成立，
diff 必须同时满足四条**精确**性质 —— 都是代数恒等式，不是阈值：

| 性质 | 断言 | 意义 |
| --- | --- | --- |
| **完备** | `apply(a, diff(a,b)).primary() ≡ b.primary()` | 补丁不漏任何一处差异 |
| **最小** | 去掉任意一条编辑，`apply` 就不再等于 `b` | 补丁不含废话 |
| **可逆** | `diff(b,a) ≡ invert(diff(a,b))` | 差异是双向的、对称的 |
| **确定** | 同一对模型跑两次，编辑序列逐位相同 | 可复现 |

「完备」这一条是**最有价值的测试**：它盯的不是 diff 的实现，而是
`Genome.primary()` 的**覆盖完整性**。哪天给 Genome 加了新字段却忘了接进
`primary()`，`apply` 就会静默地少还原一个字段 —— 只有这条测试会响。

## 编辑种类与量变 / 突变

编辑种类分成两组，分界线是「改动是否落在定性骨架上」（见 `schema.SKELETON_TOPOLOGY`）：

- **量变**：`attr`（层数 / 宽度 / 词表 / 块组计数）、`dim`（算子与范数的规模数值）、
  `shape`（张量形状）。
- **突变**：`op`（算子种类）、`variant`（算子变体 / 定性 topology 键）、
  `flag_add` / `flag_del`、`node_add` / `node_del`、`tensor_add` / `tensor_del`、
  `tie`（权重绑定）、`edge_add` / `edge_del`（连接）。

`tests/test_diff.py` 有一条测试断言**「编辑种类说它是突变」与
「`shape_fingerprint` 变了」永远等价** —— 两套判据必须同时成立，
否则报告会说一件事、指纹说另一件事，而那种不一致不会自己报错。
"""

from __future__ import annotations

from typing import Any, Mapping

from .genome import fingerprint, rebuild, shape_fingerprint
from .schema import (
    SKELETON_TOPOLOGY,
    Diff,
    Edit,
    Genome,
)

__all__ = [
    "structural_diff",
    "apply",
    "invert",
    "is_minimal",
    "summary",
    "headline",
    "group_edits",
    "rehydrate_payload",
    "MISSING",
]

#: 互逆映射在 `schema.EDIT_INVERSE` 里，由 `Edit.inverted()` 使用。
#: 这里曾经有一份**从没被调用过**的 `_INVERSE`：表写对了，但 `invert` 走的是
#: `Edit.inverted()`，而那里只交换 `old` / `new`、不翻种类。死代码看起来像已经处理过了，
#: 所以这条「可逆性」一直没被发现 —— 保留这句注释，提醒下一个人别再加第二份。
_ADD_KINDS = frozenset({"node_add", "tensor_add", "edge_add"})
_DEL_KINDS = frozenset({"node_del", "tensor_del", "edge_del"})

#: 「显著性」排序：报告里先看算子换没换，最后才看宽度变没变。
_SIGNIFICANCE = {
    "op": 0,
    "variant": 1,
    "tie": 2,
    "edge_add": 3,
    "edge_del": 3,
    "node_add": 4,
    "node_del": 4,
    "tensor_add": 5,
    "tensor_del": 5,
    "flag_add": 6,
    "flag_del": 6,
    "shape": 7,
    "dim": 8,
    "attr": 9,
}

MISSING = object()

#: 这些编辑种类的 `None` 是一个**值**，不是「地址不存在」。
#: 目前只有 `tie`：`tensors.X.tied_with = None` 就是「没有绑定」这个事实本身。
#: 其余种类（`dim` / `attr` / `shape` / `variant` …）上的 `None` 一律读作「该地址缺席」。
_NULL_IS_A_VALUE = frozenset({"tie"})


# ---------------------------------------------------------------- 计算


def structural_diff(a: Genome, b: Genome) -> Diff:
    """两个 Genome 的结构差异。**按地址排序**，所以同输入必然同输出。

    名字是 `structural_diff` 而不是 `diff`：`diff` 同时是这个模块的名字，让函数与
    模块同名会在 `from modelgenomefactory import diff` 时**静默地**把模块遮掉（拿到的是
    函数），而 `import modelgenomefactory.diff` 又能正常用 —— 和 `extract` 那次是同一类坑。
    """
    pa, pb = a.primary(), b.primary()
    edits: list[Edit] = []

    # 先按实体分组，让「整块节点 / 整张张量 / 整条边」的增删成为一条编辑，
    # 而不是同一实体的每个字段各来一条。
    nodes_a, nodes_b = _entities(pa, "nodes."), _entities(pb, "nodes.")
    tensors_a, tensors_b = _entities(pa, "tensors."), _entities(pb, "tensors.")
    for path in sorted(set(nodes_a) | set(nodes_b)):
        edits.extend(_entity_edits("node", path, nodes_a.get(path), nodes_b.get(path)))
    for path in sorted(set(tensors_a) | set(tensors_b)):
        edits.extend(_entity_edits("tensor", path, tensors_a.get(path), tensors_b.get(path)))

    edges_a = {k: v for k, v in pa.items() if k.startswith("edges.")}
    edges_b = {k: v for k, v in pb.items() if k.startswith("edges.")}
    for addr in sorted(set(edges_a) | set(edges_b)):
        if addr in edges_a and addr in edges_b:
            continue
        if addr in edges_b:
            edits.append(Edit("edge_add", addr, None, True))
        else:
            edits.append(Edit("edge_del", addr, True, None))

    topo_a = {k: v for k, v in pa.items() if k.startswith("topology.")}
    topo_b = {k: v for k, v in pb.items() if k.startswith("topology.")}
    for addr in sorted(set(topo_a) | set(topo_b)):
        va, vb = topo_a.get(addr, MISSING), topo_b.get(addr, MISSING)
        if va == vb:
            continue
        key = addr[len("topology.") :]
        kind = "variant" if key in SKELETON_TOPOLOGY else "attr"
        edits.append(Edit(kind, addr, None if va is MISSING else va, None if vb is MISSING else vb))

    return Diff(
        left=a.name,
        right=b.name,
        edits=tuple(edits),
        left_fingerprint=fingerprint(a),
        right_fingerprint=fingerprint(b),
        left_shape_fingerprint=shape_fingerprint(a),
        right_shape_fingerprint=shape_fingerprint(b),
    )


def _entities(primary: Mapping[str, Any], prefix: str) -> dict[str, dict[str, Any]]:
    """把 `nodes.<path>.<field>` 折叠成 `{path: {field: value}}`。"""
    out: dict[str, dict[str, Any]] = {}
    for key, value in primary.items():
        if not key.startswith(prefix):
            continue
        rest = key[len(prefix) :]
        path, field, dim = _split(rest)
        slot = out.setdefault(path, {})
        if dim is None:
            slot[field] = value
        else:
            slot.setdefault("dims", {})[dim] = value
    return out


def _split(rest: str) -> tuple[str, str, str | None]:
    for field in ("kind", "variant", "flags", "shape", "tied_with"):
        suffix = "." + field
        if rest.endswith(suffix):
            return rest[: -len(suffix)], field, None
    idx = rest.rfind(".dims.")
    if idx >= 0:
        return rest[:idx], "dims", rest[idx + len(".dims.") :]
    raise ValueError(f"认不出地址：{rest!r}")


def _entity_edits(what: str, path: str, a: dict[str, Any] | None, b: dict[str, Any] | None) -> list[Edit]:
    addr = f"{what}s.{path}"
    if a is None:
        return [Edit(f"{what}_add", addr, None, _payload(b))]
    if b is None:
        return [Edit(f"{what}_del", addr, _payload(a), None)]
    out: list[Edit] = []
    for field in sorted(set(a) | set(b)):
        if field == "flags":
            fa, fb = set(a.get(field, ())), set(b.get(field, ()))
            for f in sorted(fb - fa):
                out.append(Edit("flag_add", f"{addr}.flags", None, f))
            for f in sorted(fa - fb):
                out.append(Edit("flag_del", f"{addr}.flags", f, None))
            continue
        if field == "dims":
            da, db = a.get("dims", {}), b.get("dims", {})
            for k in sorted(set(da) | set(db)):
                va, vb = da.get(k, MISSING), db.get(k, MISSING)
                if va == vb:
                    continue
                out.append(
                    Edit("dim", f"{addr}.dims.{k}", None if va is MISSING else va, None if vb is MISSING else vb)
                )
            continue
        va, vb = a.get(field, MISSING), b.get(field, MISSING)
        if va == vb:
            continue
        if field == "kind":
            kind = "op"
        elif field == "variant":
            kind = "variant"
        elif field == "shape":
            kind = "shape"
        elif field == "tied_with":
            kind = "tie"
        else:  # pragma: no cover - 字段表封闭
            raise ValueError(f"未知字段 {field!r}")
        out.append(Edit(kind, f"{addr}.{field}", None if va is MISSING else va, None if vb is MISSING else vb))
    # 同一实体内部的编辑必须按一个**在取反下不变**的键排序。
    # 否则 `invert(diff(a,b)) ≡ diff(b,a)` 会在「同一地址上既加又删」时破掉：
    # 生成顺序是「先 flag_add 再 flag_del」，取反之后变成「先 flag_del 再 flag_add」，
    # 而 `diff(b,a)` 自己算出来仍然是「先加后删」—— 集合一样，顺序不同，逐位比较就不等。
    # `apply` 本来就是顺序无关的（删除 → 新增 → 赋值），所以排序不影响语义。
    return sorted(out, key=_edit_order)


#: 排序时把 add / del **看成同一个种类** —— 它们正是取反时会互换的那一对，
#: 所以任何把 `add` 和 `del` 分开的排序都必然不是取反不变的。
_PAIR_KIND = {
    "node_add": "node", "node_del": "node",
    "tensor_add": "tensor", "tensor_del": "tensor",
    "edge_add": "edge", "edge_del": "edge",
    "flag_add": "flag", "flag_del": "flag",
}


def _edit_order(e: Edit) -> tuple[str, str, str]:
    """同一实体内部编辑的规范顺序。**必须在取反下逐位不变** —— 可逆性依赖这一点。

    键里的每个分量都不能被 `inverted()` 改动：
    - 种类用 `_PAIR_KIND` 折叠（`flag_add`/`flag_del` → `flag`）；
    - 标志位名从 add 取 `new`、从 del 取 `old`，交换之后是同一个名字；
    - 其余种类（`dim` / `attr` / `shape` / `variant` / `op` / `tie`）取反时种类不动。
    """
    kind = _PAIR_KIND.get(e.kind, e.kind)
    if kind == "flag":
        flag = e.new if e.kind == "flag_add" else e.old
        return (kind, e.addr, str(flag))
    return (kind, e.addr, "")


def _payload(slot: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "kind": slot.get("kind"),
        "variant": slot.get("variant"),
        "flags": tuple(slot.get("flags", ())),
        "dims": dict(slot.get("dims", {})),
        "shape": tuple(slot.get("shape", ())),
        "tied_with": slot.get("tied_with"),
    }


def rehydrate_payload(p: Any) -> Any:
    """`node_add` / `tensor_add` 的**载荷**里也有元组（`flags` / `shape`）。

    `schema.rehydrate` 只认地址，认不出「这个列表是载荷里的 `shape`」——
    载荷是一个嵌套字典，它的键名（`shape` / `flags`）不在地址空间里。
    所以载入侧要单独还原它，否则「载入的 diff」与「内存里算出的 diff」只差这一处：
    逐位比较会失败，而 `apply` 出来的 Genome 会带着 `[4096, 4096]` 这种列表形状。
    """
    if not isinstance(p, Mapping):
        return p
    out = dict(p)
    if isinstance(out.get("flags"), list):
        out["flags"] = tuple(out["flags"])
    if isinstance(out.get("shape"), list):
        out["shape"] = tuple(out["shape"])
    return out


# ---------------------------------------------------------------- 施加与互逆


def apply(g: Genome, d: Diff) -> Genome:
    """把补丁施加到 `g` 上，返回新的 Genome。

    **不依赖编辑顺序**：先落删除与新增，再落字段级修改。所以
    `diff(b,a)` 与 `invert(diff(a,b))` 即使顺序不同也能得到同一个结果。
    """
    flat = dict(g.primary())
    dels, adds, sets = [], [], []
    for e in d.edits:
        (dels if e.kind in _DEL_KINDS else adds if e.kind in _ADD_KINDS else sets).append(e)

    for e in dels:
        prefix = e.addr + "."
        for key in [k for k in flat if k == e.addr or k.startswith(prefix)]:
            del flat[key]
    for e in adds:
        _expand_add(flat, e)
    for e in sets:
        if e.kind == "flag_add":
            cur = tuple(flat.get(e.addr, ()))
            flat[e.addr] = tuple(sorted(set(cur) | {e.new}))
        elif e.kind == "flag_del":
            cur = tuple(flat.get(e.addr, ()))
            flat[e.addr] = tuple(sorted(set(cur) - {e.old}))
        elif e.new is None and e.kind not in _NULL_IS_A_VALUE:
            # 地址空间里 `None` 默认表示**这个地址不存在**，不是「值等于 None」。
            # 这里必须真的删掉键：早先写成 `flat[e.addr] = None`，于是
            # `nodes.blocks[*].attn.dims.sliding_window` 会留下一个值为 `None` 的幽灵键
            # （Mistral 有滑动窗口、Mixtral 没有）。`same_primary` 用 dict 相等判出来了，
            # 而调试用的 `primary_diff_keys` 当时用 `.get()` 比较，把「缺席」和
            # 「值为 None」看成一样，正好把这条差异吞掉了 —— 两处一起改掉。
            flat.pop(e.addr, None)
        else:
            flat[e.addr] = e.new
    return rebuild(g, flat)


def _expand_add(flat: dict[str, Any], e: Edit) -> None:
    payload = e.new
    if e.kind == "edge_add":
        flat[e.addr] = True
        return
    flat[f"{e.addr}.kind"] = payload["kind"]
    flat[f"{e.addr}.variant"] = payload["variant"]
    flat[f"{e.addr}.flags"] = tuple(payload.get("flags", ()))
    for k, v in (payload.get("dims") or {}).items():
        flat[f"{e.addr}.dims.{k}"] = v
    if e.kind == "tensor_add":
        flat[f"{e.addr}.shape"] = tuple(payload["shape"])
        flat[f"{e.addr}.tied_with"] = payload.get("tied_with")


def invert(d: Diff) -> Diff:
    """补丁取反。**保持顺序**（不 reverse）—— 因为 `diff` 按地址排序，
    反方向算出来的编辑序列也是按地址排序的，两者必须逐位相同。"""
    return Diff(
        left=d.right,
        right=d.left,
        edits=tuple(e.inverted() for e in d.edits),
        left_fingerprint=d.right_fingerprint,
        right_fingerprint=d.left_fingerprint,
        left_shape_fingerprint=d.right_shape_fingerprint,
        right_shape_fingerprint=d.left_shape_fingerprint,
    )


def is_minimal(a: Genome, d: Diff) -> bool:
    """逐条验证最小性：去掉任意一条编辑，补丁就还原不出 `b`。

    这是**穷尽**检查（`len(edits)` 次 apply），不是抽查 —— 编辑序列小（几十条），
    代价可以接受，而「补丁里有废话」这种问题抽查是抓不到的。

    去掉某条编辑后 `apply` **抛异常**也算「还原不出来」：比如抽掉
    `topology.n_experts` 之后，`experts[*]` 还在而专家数变成 0，结构自相矛盾。
    矛盾当然不等于相等，所以这条编辑是必要的。不接住这个异常的话，
    「最小性」检查会在一个**本来正确**的补丁上崩掉。
    """
    from .genome import same_primary

    target = apply(a, d)
    for i in range(len(d.edits)):
        pruned = Diff(
            left=d.left,
            right=d.right,
            edits=tuple(e for j, e in enumerate(d.edits) if j != i),
            left_fingerprint=d.left_fingerprint,
            right_fingerprint=d.right_fingerprint,
            left_shape_fingerprint=d.left_shape_fingerprint,
            right_shape_fingerprint=d.right_shape_fingerprint,
        )
        try:
            pruned_result = apply(a, pruned)
        except (ValueError, KeyError):
            continue                      # 自相矛盾 ⇒ 必然不等于 target ⇒ 这条编辑必要
        if same_primary(pruned_result, target):
            return False
    return True


# ---------------------------------------------------------------- 摘要


def summary(d: Diff) -> dict[str, Any]:
    """按编辑种类与突变坐标聚合。报告里的「结构差异 N 处」就是它。"""
    by_kind: dict[str, int] = {}
    for e in d.edits:
        by_kind[e.kind] = by_kind.get(e.kind, 0) + 1
    by_coord: dict[str, int] = {}
    for e in d.edits:
        if e.coordinate:
            by_coord[e.coordinate] = by_coord.get(e.coordinate, 0) + 1
    return {
        "left": d.left,
        "right": d.right,
        "n_edits": len(d.edits),
        "n_scale": len(d.scale_edits()),
        "n_structural": len(d.structural_edits()),
        "is_mutation": d.is_mutation,
        "coordinates": list(d.coordinates),
        "by_kind": dict(sorted(by_kind.items())),
        "by_coordinate": dict(sorted(by_coord.items())),
    }


def group_edits(d: Diff, limit: int = 8) -> tuple[tuple[str, int, tuple[Edit, ...]], ...]:
    """把编辑按「同一类改动」聚成组，便于报告展示。

    分组键是 `(kind, 地址最后一段)` —— 于是「8 个专家 × 3 张张量 = 24 条 tensor_add」
    会聚成一行，而不是刷满整页。
    """
    buckets: dict[tuple[str, str], list[Edit]] = {}
    for e in d.edits:
        buckets.setdefault((e.kind, e.addr.rsplit(".", 1)[-1]), []).append(e)
    rows = sorted(
        ((k[0], len(v), tuple(v)) for k, v in buckets.items()),
        key=lambda r: (_SIGNIFICANCE.get(r[0], 99), -r[1]),
    )
    return tuple(rows[:limit])


def headline(d: Diff, limit: int = 6, *, verdict: bool = True) -> tuple[str, ...]:
    """最值得先看的若干条差异（按显著性排序，同类聚合成一行）。

    `verdict=False` 去掉末尾那句「只有规模变化 / 结构突变坐标」——
    调用方已经把判定单独印出来了（`Diff.is_mutation` / `Diff.coordinates`），
    再在差异列表里重复一遍，读起来像多了一条差异。
    """
    if d.is_empty:
        return ("结构零差异：这两个模型的指纹逐位相同。",)
    out: list[str] = []
    for kind, n, members in group_edits(d, limit=limit):
        sample = members[0]
        if n == 1:
            out.append(sample.render())
        else:
            tail = sample.addr.rsplit(".", 1)[-1]
            out.append(f"{kind:10s} ×{n:<3d} {tail}（例：{sample.addr} : {_short(sample.old)} → {_short(sample.new)}）")
    if verdict:
        if d.is_mutation:
            out.append(f"结构突变坐标：{', '.join(d.coordinates)}")
        else:
            out.append("只有规模变化（骨架指纹相同）。")
    return tuple(out)


def _short(v: Any, limit: int = 28) -> str:
    s = repr(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"
