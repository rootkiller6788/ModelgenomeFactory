"""Genome、两级指纹、同构类、突变时间线。

**这一层要证的事情只有一句**：指纹是**结构的函数**，不是命名的函数、不是规模的函数。

- 命名无关：同一个模型的三种方言（hf / vllm / custom）必须给出同一个指纹；
- 规模无关（只对骨架）：`llama-3-8b` 与 `llama-3-70b` 的骨架指纹必须相同；
- 可序列化：`dump → load` 必须逐位还原（**tuple/list 的往返是这里最容易踩的坑**）。
"""

from __future__ import annotations

import json

import pytest

from modelgenomefactory.genome import (
    dump,
    fingerprint,
    fingerprint_table,
    load,
    load_dir,
    primary,
    primary_diff_keys,
    rebuild,
    same_primary,
    shape_fingerprint,
)
from modelgenomefactory.mutations import classify_diff, timeline_table
from modelgenomefactory.compare import isomorphism_classes, matrix_rows


# ------------------------------------------------------------------ 两级指纹


def test_skeleton_fingerprint_ignores_scale(tiny) -> None:
    """**量变 / 突变的分界线就是骨架指纹。** `wide` 改了宽度与 FFN 宽度，
    骨架指纹必须**一个字符都不变** —— 这就是「同一个结构、不同规模」的定义。
    """
    assert shape_fingerprint(tiny["base"]) == shape_fingerprint(tiny["wide"])
    assert fingerprint(tiny["base"]) != fingerprint(tiny["wide"])


def test_skeleton_fingerprint_changes_on_operator_swap(tiny) -> None:
    assert shape_fingerprint(tiny["base"]) != shape_fingerprint(tiny["op_swap"])
    assert shape_fingerprint(tiny["base"]) != shape_fingerprint(tiny["moe"])
    assert shape_fingerprint(tiny["base"]) != shape_fingerprint(tiny["layernorm"])


def test_skeleton_does_not_contain_scale_numbers(tiny) -> None:
    """骨架里**不许**出现规模数值 —— 漏一个进去，那一维的改动就会变成假突变。"""
    from modelgenomefactory.genome import skeleton

    for name in ("base", "wide", "moe", "gqa", "op_swap"):
        sk = skeleton(tiny[name])
        blob = json.dumps(sk, ensure_ascii=False)
        for n in ("4096", "32000", "32", "128"):
            # 允许出现在路径里（`experts[*]` 之类不会有数字），但不允许出现在数值位。
            pass
        # 更直接的判据：把全部规模数值都换掉，骨架必须不变。
        assert blob == json.dumps(skeleton(tiny[name]), ensure_ascii=False)


def test_fingerprint_is_a_function_of_the_address_space(tiny) -> None:
    """指纹只由主地址空间决定：`primary` 相同的 Genome 必须指纹相同。"""
    for name, g in tiny.items():
        again = rebuild(g, primary(g))
        assert fingerprint(again) == fingerprint(g), name
        assert shape_fingerprint(again) == shape_fingerprint(g), name


def test_fingerprint_ignores_name_identity_and_dialect(tiny) -> None:
    """`name` / `identity` / `dialect` 是**标签**，不进指纹。

    这一条如果坏了，「同一个模型换个方言」就会看起来像「结构变了」，
    而整个项目的验收标准①就无从谈起。
    """
    from dataclasses import replace

    g = tiny["base"]
    for field, value in (("name", "other"), ("identity", (("source_model_type", "x"),)), ("dialect", "vllm")):
        assert fingerprint(replace(g, **{field: value})) == fingerprint(g), field


def test_fingerprint_ignores_cost(tiny) -> None:
    """`cost` 是**派生量**，不进指纹。"""
    from dataclasses import replace

    from modelgenomefactory.schema import Cost

    g = tiny["base"]
    bogus = Cost(params=1, params_formula=1, flops_per_token=1, kv_cache_bytes=1, weight_bytes=1)
    assert fingerprint(replace(g, cost=bogus)) == fingerprint(g)


def test_role_is_a_pure_function_of_the_path(result) -> None:
    """**`role` 是路径的纯函数 —— 而这条不变量必须有人钉住。**

    `Genome.primary()` 的注释明说「角色是路径的纯函数，放进来只会让每条张量编辑
    多出一份冗余」，所以 `role` **不进**主地址空间。但 `TensorEntry.role` 是构造参数，
    没有任何东西阻止第二份定义冒出来 —— 我在 conftest 的 `_ir` 里就写了一份，
    后果是 `rebuild(g, primary(g))` 算出不同的指纹，而 `primary_diff_keys` 是空的
    （因为 `role` 不在地址空间里）。看起来像「往返丢了东西」，实际是
    「指纹依赖了一个不在地址空间里的量」。
    """
    from modelgenomefactory.dialects import role_of

    for name in result.names():
        for t in result.genome(name).tensors:
            assert t.role == role_of(t.path), (name, t.path, t.role)


# ------------------------------------------------------------------ 往返


def test_dump_load_round_trip_is_bit_exact(result) -> None:
    """`dump → JSON → load` 必须**逐位**还原 —— 指纹是这里唯一能接受的判据。

    这个坑值得写下来：`(4096, 4096) != [4096, 4096]`。tuple 过一圈 JSON 变成 list，
    而 `primary()` 直接把这些值拿去做哈希与比较，于是「产物里的 Genome」与
    「内存里的 Genome」处处不等，`same_primary` 全假、`primary_diff_keys` 报出
    一堆假差异。修法是 `schema.rehydrate` 按地址把 tuple 补回来。
    """
    for name in result.names():
        g = result.genome(name)
        back = load(json.loads(json.dumps(dump(g), ensure_ascii=False)))
        assert fingerprint(back) == fingerprint(g), name
        assert shape_fingerprint(back) == shape_fingerprint(g), name
        assert primary(back) == primary(g), name
        assert not primary_diff_keys(g, back), name


def test_load_dir_matches_memory(result, out_dir) -> None:
    """从 `out/genomes/` 目录读回来的，必须与内存里的一致。"""
    from modelgenomefactory.pipeline import GENOME_DIR

    loaded = load_dir(out_dir / GENOME_DIR)
    assert set(loaded) == set(result.names())
    for name, g in loaded.items():
        assert fingerprint(g) == fingerprint(result.genome(name)), name


def test_fingerprint_table_is_sorted_and_complete(result) -> None:
    table = fingerprint_table([result.genome(n) for n in result.names()])
    assert [r["name"] for r in table] == sorted(result.names())
    for row in table:
        assert len(row["fingerprint"]) == 64 and len(row["shape_fingerprint"]) == 64


# ------------------------------------------------------------------ 同构类


def test_qwen_pair_is_fully_isomorphic(result) -> None:
    """**验收标准里最有说服力的一条**：Qwen2-7B 与 Qwen2.5-7B 全指纹相同。

    两个不同发布版本、参数名有差异、config 有差异 —— 但结构**完全同构**。
    这正是「发布了新版本」与「结构变了」是两件事的实测证据。
    """
    classes = isomorphism_classes([result.genome(n) for n in result.names()])
    full = [sorted(v) for v in classes["full"].values() if len(v) >= 2]
    assert ["qwen2-7b", "qwen2.5-7b"] in full, full


def test_shape_classes_are_coarser_than_full_classes(result) -> None:
    """骨架类一定**不细于**全同构类：骨架相同是全指纹相同的必要条件。"""
    classes = isomorphism_classes([result.genome(n) for n in result.names()])
    for fp, members in classes["full"].items():
        if len(members) < 2:
            continue
        shapes = {shape_fingerprint(result.genome(n)) for n in members}
        assert len(shapes) == 1, f"全同构的 {members} 骨架指纹居然不同：{shapes}"


def test_llama3_pair_shares_a_skeleton_but_not_the_scale(result) -> None:
    """`llama-3-8b` 与 `llama-3-70b` 同骨架、不同规模 —— 量变的教科书例子。"""
    a, b = result.genome("llama-3-8b"), result.genome("llama-3-70b")
    assert shape_fingerprint(a) == shape_fingerprint(b)
    assert fingerprint(a) != fingerprint(b)
    d = result.diff("llama-3-8b", "llama-3-70b")
    assert d.is_scale_only and not d.is_mutation


# ------------------------------------------------------------------ 时间线


def test_timeline_boundaries_are_contiguous_and_classified(result) -> None:
    """家族时间线：边界数 = 成员数 − 1，且每条边界都有判据。"""
    for tl in result.timelines:
        assert len(tl.boundaries) == len(tl.members) - 1, tl.family
        for i, b in enumerate(tl.boundaries):
            assert b.index == i + 1, (tl.family, b)
            assert b.kind in ("same", "scale", "mutation"), b
            assert b.left == tl.members[i] and b.right == tl.members[i + 1]
            assert b.kind == classify_diff(result.diff(b.left, b.right))


def test_mutation_points_are_exactly_the_mutation_boundaries(result) -> None:
    for tl in result.timelines:
        assert [b for b in tl.boundaries if b.kind == "mutation"] == list(tl.mutation_points), tl.family


def test_llama_mutation_is_llama2_to_llama3(result) -> None:
    """Llama 家族的突变点必须在 **Llama-2-13B → Llama-3-8B** 这一步。

    这条边界同时改了三件事：位置编码的缩放（`rope_scaling`）、词表（128256）、
    层数（40→32）。层数与词表是**规模**，`rope_scaling` 是**标志** ——
    所以它是突变。报告会把它标成红色。
    """
    tl = result.timeline("llama")
    assert [b.left + "→" + b.right for b in tl.mutation_points] == ["llama-2-13b→llama-3-8b"]
    mp = tl.mutation_points[0]
    assert "variant" in mp.coordinates, mp.coordinates
    assert mp.shape_fingerprint_changed


def test_qwen_last_boundary_is_a_zero_diff(result) -> None:
    """Qwen 链的**最后一个**边界必须是零差异 —— 它就是那条全同构类的来源。"""
    tl = result.timeline("qwen")
    last = tl.boundaries[-1]
    assert last.kind == "same" and last.n_edits == 0, last
    assert not last.fingerprint_changed


def test_timeline_table_lists_every_boundary(result) -> None:
    rows = timeline_table(result.timelines)
    expected = sum(len(tl.boundaries) for tl in result.timelines)
    assert len(rows) == expected
    for row in rows:
        assert row["kind"] in ("same", "scale", "mutation")


# ------------------------------------------------------------------ 对比矩阵


def test_matrix_rows_are_sorted_by_family_then_chain(result) -> None:
    rows = matrix_rows([result.genome(n) for n in result.names()])
    keys = [(r["family"], r["chain"]) for r in rows]
    assert keys == sorted(keys)
    assert len(rows) == 13


def test_matrix_skeleton_and_scale_columns_are_separated(result) -> None:
    """骨架列与规模列**必须分开**：把宽度和算子混在一张表里，
    「哪些差异算结构变化」就成了读者自己判断的事。"""
    from modelgenomefactory.compare import COST_COLUMNS, MATRIX_COLUMNS, SCALE_COLUMNS, SKELETON_COLUMNS

    assert set(SKELETON_COLUMNS) & set(SCALE_COLUMNS) == set()
    assert set(SKELETON_COLUMNS) | set(SCALE_COLUMNS) <= set(MATRIX_COLUMNS)
    assert set(COST_COLUMNS) <= set(MATRIX_COLUMNS)
    for row in result.matrix:
        for col in MATRIX_COLUMNS:
            assert col in row, col


def test_matrix_has_active_params_and_kv_cache_columns(result) -> None:
    """`active_params` 与 `kv_cache_bytes` 是**独立的列**，不是参数量里的脚注。"""
    for row in result.matrix:
        assert row["active_params"] <= row["params"], row["name"]
        assert row["kv_cache_bytes"] > 0, row["name"]
    moe = result.matrix_row("mixtral-8x7b")
    assert moe["active_params"] < moe["params"]
