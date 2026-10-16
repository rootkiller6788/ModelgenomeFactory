"""Diff 代数 —— 本项目最硬的一块。

「对任意两个开源模型输出**可复现**的结构 Diff」这句话里的「可复现」不是形容词，
它是四条**可以逐条断言**的代数性质：

| 性质 | 断言 |
| --- | --- |
| 完备 | `apply(a, diff(a, b)).primary() ≡ b.primary()` |
| 最小 | 去掉任意一条编辑，就还原不出 `b` |
| 可逆 | `diff(b, a) ≡ invert(diff(a, b))` |
| 确定 | 同样的输入跑两次，编辑序列逐位相同 |

四条一起才叫「补丁」。缺了完备性，Diff 只是「一份差异清单」；缺了最小性，
它会把「派生量」也写进去，于是 `apply` 出来的东西和 `b` 差在一个不该管的地方。

测试分两层：
- `tiny`（7 个手工 Genome，毫秒级）：覆盖**每一类**编辑，逐条查代数恒等式；
- `result`（13 个真实模型 × 78 个无序对）：只跑一遍，确认性质在真实规模上成立。
"""

from __future__ import annotations

import pytest

from modelgenomefactory import diff as diff_mod
from modelgenomefactory.diff import apply, invert, is_minimal, structural_diff
from modelgenomefactory.genome import same_primary
from modelgenomefactory.schema import (
    DERIVED_GENOME_FIELDS,
    EDIT_INVERSE,
    MUTATION_EDIT_KINDS,
    SCALE_EDIT_KINDS,
)

#: 每一类编辑都要在 tiny 里被至少一对模型触发 —— 否则「代数成立」只覆盖了一部分。
ALL_KINDS = frozenset(SCALE_EDIT_KINDS) | frozenset(MUTATION_EDIT_KINDS)

#: `tiny` 的 8 个手工 Genome 能触发的类别。规范张量的形状约定是 `(out, in)`，
#: 所以 KV 变窄体现为**第 0 维**变小。
_TINY_KINDS = frozenset({"attr", "dim", "shape", "op", "variant", "tie", "flag_add"})


# ------------------------------------------------------------------ tiny 层


def test_tiny_zoo_covers_the_operator_branch(tiny) -> None:
    """tiny 覆盖 7 类编辑，**其中 `op` 只有 tiny 能触发**。

    覆盖不了的是**结构增删**（`node_add` / `node_del` / `tensor_add` /
    `tensor_del` / `edge_add` / `edge_del`）与 `flag_del` —— 它们需要
    「节点集不同的两个模型」，而 `_ir` 是一个固定模板，手工给不出；
    这 7 类由真实 78 对覆盖（见下一条测试）。

    两条测试合起来才是完整的覆盖声明：**没有任何一类编辑是零覆盖的**。
    """
    seen: set[str] = set()
    names = sorted(tiny)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            seen.update(e.kind for e in structural_diff(tiny[a], tiny[b]).edits)
    assert seen == _TINY_KINDS, sorted(seen ^ _TINY_KINDS)


def test_real_zoo_covers_every_edit_kind_except_op(result) -> None:
    """真实规模上再数一遍，并**显式记下缺的那一类**。

    「缺 `op`」不是 bug，是这 13 个模型的结构事实：`blocks[*].ffn` 在所有模型里
    都是 `ffn` 种类（MLP / SwiGLU / MoE 是它的**变体**），`blocks[*].attn` 都是
    `attention`。所以 `op` 这条分支在真实数据上是零覆盖的 —— 记录它，而不是
    假装覆盖到了。`op` 的代数性质由 `tiny["op_swap"]` 验。
    """
    seen: set[str] = set()
    for d in result.pairs_ordered():
        seen.update(e.kind for e in d.edits)
    assert "op" not in seen, "zoo 上出现了 op —— 那说明有模型换了算子种类，附录 B 要更新"
    assert seen == set(ALL_KINDS) - {"op"}, sorted(set(ALL_KINDS) - seen)


@pytest.mark.parametrize("a_name", ["base", "wide", "moe", "window", "untied", "layernorm", "gqa"])
def test_apply_is_complete_on_tiny(tiny, a_name: str) -> None:
    """① 完备：`apply(a, diff(a, b))` 与 `b` 在主地址空间上**逐位相同**。"""
    a = tiny[a_name]
    for b_name in sorted(tiny):
        b = tiny[b_name]
        assert same_primary(apply(a, structural_diff(a, b)), b), (a_name, b_name)


@pytest.mark.parametrize("a_name", ["base", "wide", "moe", "window", "untied", "layernorm", "gqa"])
def test_diff_is_minimal_on_tiny(tiny, a_name: str) -> None:
    """② 最小：去掉任意一条编辑就还原不出 `b`（**穷尽**检查，不是抽查）。"""
    a = tiny[a_name]
    for b_name in sorted(tiny):
        if b_name == a_name:
            continue
        d = structural_diff(a, tiny[b_name])
        assert is_minimal(a, d), (a_name, b_name)


@pytest.mark.parametrize("a_name", ["base", "wide", "moe", "window", "untied", "layernorm", "gqa"])
def test_diff_is_invertible_on_tiny(tiny, a_name: str) -> None:
    """③ 可逆：`diff(b, a)` 与 `invert(diff(a, b))` **逐位**相同（含顺序）。

    早先 `inverted()` 里 `reversed()` 了编辑序列，于是这条只在碰巧回文时成立 ——
    而现在**只读命令依赖这条性质**：产物里每个无序对只存一个方向，
    反方向靠 `invert` 得到。它坏了 `diff b a` 就会给出一个不同的答案。
    """
    a = tiny[a_name]
    for b_name in sorted(tiny):
        b = tiny[b_name]
        fwd, rev = structural_diff(a, b), structural_diff(b, a)
        assert invert(fwd) == rev, (a_name, b_name)
        assert fwd.edits == tuple(e.inverted() for e in rev.edits), (a_name, b_name)


def test_diff_is_deterministic_on_tiny(tiny) -> None:
    """④ 确定：同样的输入，编辑序列**逐位**相同（含顺序与旧值/新值）。"""
    a, b = tiny["base"], tiny["gqa"]
    assert structural_diff(a, b) == structural_diff(a, b)
    assert structural_diff(a, b).to_dict() == structural_diff(a, b).to_dict()


def test_invert_twice_is_identity_on_tiny(tiny) -> None:
    a, b = tiny["base"], tiny["moe"]
    d = structural_diff(a, b)
    assert invert(invert(d)) == d


def test_self_diff_is_empty_and_minimal(tiny) -> None:
    for name, g in tiny.items():
        d = structural_diff(g, g)
        assert d.is_empty, name
        assert d.is_scale_only and not d.is_mutation, name
        assert is_minimal(g, d), name


# ------------------------------------------------------------------ 判据


def test_scale_only_change_keeps_the_skeleton_fingerprint(tiny) -> None:
    """**量变的判据是骨架指纹没变**，不是「编辑条数少」。

    `wide` 改了宽度与 FFN 宽度（多条 `dim`/`shape` 编辑），但它仍然是同一个结构：
    32 层、同样的算子、同样的连接、同样的张量契约。所以它是**量变**。
    """
    d = structural_diff(tiny["base"], tiny["wide"])
    assert not d.is_empty
    assert d.is_scale_only, [e.kind for e in d.edits]
    assert not d.is_mutation
    assert d.left_shape_fingerprint == d.right_shape_fingerprint
    assert d.left_fingerprint != d.right_fingerprint, "规模变了，全指纹必须变"


@pytest.mark.parametrize("variant", ["moe", "layernorm", "window", "untied", "gqa", "op_swap"])
def test_operator_and_contract_changes_are_mutations(tiny, variant: str) -> None:
    """**突变的判据是骨架指纹变了** —— 换算子、换范数、加标志、拆绑定、改张量契约都算。

    每一条都值得点名，因为它们是「结构演化」在真实模型上的几种典型形态：
    `moe` = MLP→MoE、`layernorm` = RMSNorm→LayerNorm、`window` = 加滑窗、
    `untied` = 拆权重绑定、`gqa` = MHA→GQA（改了 KV 的张量契约）、
    `op_swap` = 换了算子种类（合成，见 `tiny` 夹具的说明）。
    """
    d = structural_diff(tiny["base"], tiny[variant])
    assert d.is_mutation, (variant, [e.kind for e in d.edits])
    assert d.left_shape_fingerprint != d.right_shape_fingerprint
    assert d.structural_edits(), variant
    assert d.coordinates, f"{variant} 是突变，必须给出坐标"


def test_moe_is_a_variant_change_not_an_op_change(tiny) -> None:
    """**MLP→MoE 是 `variant`，不是 `op`。** 这条区分很关键：

    `blocks[*].ffn` 这个算子在两种模型里都是「FFN」，变的是它的**形态**
    （稠密 → 稀疏专家）。把它记成「换了算子」会让报告说「这两个模型有一个算子
    对不上」，而事实是「同一个算子用了不同实现」。MoE 额外带来的
    `router` / `experts[*]` 子节点由 `node_add` 表达。
    """
    d = structural_diff(tiny["base"], tiny["moe"])
    assert d.coordinates == ("variant",), d.coordinates
    assert "op" not in {e.kind for e in d.edits}


def test_op_swap_is_the_only_op_edit(tiny) -> None:
    """`op_swap` 恰好产生一条 `op` 编辑，坐标是 `operator`。"""
    d = structural_diff(tiny["base"], tiny["op_swap"])
    ops = [e for e in d.edits if e.kind == "op"]
    assert len(ops) == 1, [e.render() for e in d.edits]
    assert ops[0].addr == "nodes.blocks[*].ffn.kind"
    assert (ops[0].old, ops[0].new) == ("ffn", "expert")
    assert d.coordinates == ("operator",), d.coordinates


def test_gqa_is_a_variant_change_plus_a_dim_change(tiny) -> None:
    """MHA→GQA 的坐标是 `variant` —— 改的是注意力的**形态**。

    KV 投影变窄（`n_kv_heads` 与 `k_proj` 的形状）是**它的后果**，记成
    `dim` / `shape` 编辑，属于量变坐标。把它们一起算进「突变坐标」会让
    「改宽度」也看起来像换结构。
    """
    d = structural_diff(tiny["base"], tiny["gqa"])
    assert d.coordinates == ("variant",), d.coordinates
    assert d.is_mutation
    kinds = {e.kind for e in d.edits}
    assert "dim" in kinds and "shape" in kinds, kinds
    # KV 宽度必须真的变窄：4 头 → 2 头。规范张量的形状是 `(out, in)`，所以第 0 维变小。
    kv = [e for e in d.edits if e.addr == "tensors.blocks[*].attn.k_proj.weight.shape"]
    assert kv and kv[0].new[0] < kv[0].old[0], [e.render() for e in kv]


def test_window_is_a_flag_add_and_its_inverse_is_a_flag_del(tiny) -> None:
    d = structural_diff(tiny["base"], tiny["window"])
    kinds = {e.kind for e in d.edits}
    assert kinds == {"flag_add"}, kinds
    assert {e.kind for e in structural_diff(tiny["window"], tiny["base"]).edits} == {"flag_del"}


def test_untied_is_a_tie_edit(tiny) -> None:
    d = structural_diff(tiny["base"], tiny["untied"])
    assert {e.kind for e in d.edits} == {"tie"}, [e.render() for e in d.edits]
    assert "topology" in d.coordinates


# ------------------------------------------------------------------ 真实规模


def test_all_78_pairs_satisfy_the_four_laws(result) -> None:
    """四条性质在**全部 78 个无序对**上一次跑完。

    不拆成 78 个测试：那会让每次收集都重算一遍流水线，而这条断言本身是 O(编辑数²)。
    """
    pairs = result.pairs_ordered()
    assert len(pairs) == 78, len(pairs)
    for d in pairs:
        a, b = result.genome(d.left), result.genome(d.right)
        rebuilt = apply(a, d)
        assert same_primary(rebuilt, b), d.left + "→" + d.right
        assert invert(structural_diff(b, a)) == d, d.left + "→" + d.right
        assert is_minimal(a, d), d.left + "→" + d.right


def test_pairs_are_stored_once_per_unordered_pair(result) -> None:
    """产物里每个**无序对**只存一份 —— 反方向靠 `invert`，不重复存。"""
    keys = sorted(result.pairs)
    assert len(keys) == len(set(keys)) == 78
    for key in keys:
        left, right = key.split("|")
        assert left < right, f"{key} 不是规范方向（应当按字典序）"
        assert result.pairs[key].left == left


def test_diff_lookup_is_symmetric_and_uses_invert(result) -> None:
    """`diff(a, b)` 与 `diff(b, a)` 是两个方向，且**不重复计算**。"""
    fwd = result.diff("llama-2-7b", "llama-3-8b")
    rev = result.diff("llama-3-8b", "llama-2-7b")
    assert fwd.left == "llama-2-7b" and fwd.right == "llama-3-8b"
    assert rev.left == "llama-3-8b" and rev.right == "llama-2-7b"
    assert invert(fwd) == rev


def test_diff_does_not_carry_derived_fields(result) -> None:
    """派生量（`cost.*` / `role` / `name` / `dialect`）**不进地址空间**。

    它们由别的事实唯一决定；写进补丁只会让「最小性」变成「冗余」，
    而且 `apply` 之后还得复算一遍 —— 两份真相。
    """
    bad = []
    for d in result.pairs_ordered():
        for e in d.edits:
            if any(e.addr.startswith(f"{f}.") or e.addr == f for f in DERIVED_GENOME_FIELDS):
                bad.append(e.addr)
            if ".role" in e.addr:
                bad.append(e.addr)
    assert not bad, f"这些编辑落在派生字段上：{sorted(set(bad))[:5]}"


def test_edit_kind_inverse_table_is_an_involution() -> None:
    """互逆映射必须是**对合**：`inv(inv(k)) == k`。

    `EDIT_INVERSE` 只登记**增删成对**的种类（8 条）。其余种类（`attr` / `dim` /
    `shape` / `op` / `variant` / `tie`）是**自逆**的 —— 所以取反是
    `EDIT_INVERSE.get(kind, kind)` 而不是 `EDIT_INVERSE[kind]`。
    写成后者会在第一次遇到 `shape` 时 `KeyError`，而那时恰好没人测到它。
    """
    for kind in ALL_KINDS:
        inv = EDIT_INVERSE.get(kind, kind)
        assert EDIT_INVERSE.get(inv, inv) == kind, kind
    # 增删必须成对：只加不删的映射是半个补丁。
    assert set(EDIT_INVERSE) == {"node_add", "node_del", "tensor_add", "tensor_del",
                                 "edge_add", "edge_del", "flag_add", "flag_del"}
    assert set(EDIT_INVERSE) <= ALL_KINDS


def test_inverted_edit_swaps_old_and_new() -> None:
    for kind in ALL_KINDS:
        e = diff_mod.Edit(kind, "topology.hidden", 4096, 8192)
        inv = e.inverted()
        assert inv.old == 8192 and inv.new == 4096
        assert inv.addr == e.addr


def test_none_means_absent_except_for_tie(tiny) -> None:
    """`None` 在编辑里表示「该地址原本不存在」—— **只有 `tie` 例外**。

    把「值就是 None」和「地址不存在」混成一件事，会让 `apply` 在一个不存在的
    地址上写入 `None`（凭空造出一个拓扑键），而完备性检查只有在那个键恰好被
    读到的时候才会炸 —— 也就是在别的模型上炸。
    """
    # `untied`：base 的 `head.weight.tied_with` 是 `"embed.weight"`，untied 是 `None`。
    # 这里的 `None` 是一个**值**（「没有绑定」这个事实本身），不是「地址缺席」。
    d = structural_diff(tiny["base"], tiny["untied"])
    tie = [e for e in d.edits if e.kind == "tie"]
    assert tie and tie[0].new is None, [e.render() for e in d.edits]

    # 反过来：`tie` 之外的编辑里，`None` 一定意味着那个地址在另一边**不存在**。
    for pair in (("base", "layernorm"), ("base", "op_swap"), ("base", "gqa")):
        d = structural_diff(tiny[pair[0]], tiny[pair[1]])
        for e in d.edits:
            if e.kind != "tie" and e.old is None:
                assert e.addr not in tiny[pair[0]].primary(), e.render()
            if e.kind != "tie" and e.new is None:
                assert e.addr not in tiny[pair[1]].primary(), e.render()
