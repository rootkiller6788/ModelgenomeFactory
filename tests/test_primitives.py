"""基元层：rng / schema / genome。

这一层是**全项目可复现性的地基**：如果 `canonical_json` 对字典键顺序敏感，
或者 `Edit.inverted` 漏了某个种类，上面所有的指纹与 diff 结论都会跟着错，
而且错得静默。所以这些测试盯的是**恒等式**，不是「跑得通」。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from modelgenomefactory import genome as genome_mod
from modelgenomefactory import rng
from modelgenomefactory.diff import apply, invert, structural_diff
from modelgenomefactory.schema import (
    DERIVED_GENOME_FIELDS,
    EDIT_INVERSE,
    TUPLE_ADDRS,
    Edit,
    rehydrate,
)


# ------------------------------------------------------------------ rng


def test_canonical_json_is_key_order_independent() -> None:
    a = {"b": 1, "a": [{"y": 2, "x": 3}]}
    b = {"a": [{"x": 3, "y": 2}], "b": 1}
    assert rng.canonical_json(a) == rng.canonical_json(b)


def test_canonical_json_renders_tuples_like_lists() -> None:
    """元组与列表在**指纹**里必须等价：JSON 存不下元组，落盘再读回来会变成列表。
    指纹要是区分它们，「从产物载入的」与「内存里算出的」就会给出不同的指纹。"""
    assert rng.canonical_json({"s": (1, 2)}) == rng.canonical_json({"s": [1, 2]})


def test_digest_is_64_hex_and_deterministic() -> None:
    d = rng.sha256_text("modelgenomefactory")
    assert len(d) == 64 and all(c in "0123456789abcdef" for c in d)
    assert d == rng.sha256_text("modelgenomefactory")


def test_rng_module_never_calls_builtin_hash() -> None:
    """内置 `hash()` 对 str 是**按进程加盐**的（`PYTHONHASHSEED`）—— 用它算指纹，
    同一份结构在两个进程里会得到不同结果，而这一点在同一个进程里永远测不出来。

    用 AST 找**调用**，不做字符串匹配：`rng.py` 的模块 docstring 里就写着
    「禁用内置 `hash()`」这句话，字符串匹配会把它自己当成违规。
    """
    import ast

    tree = ast.parse(Path(rng.__file__).read_text(encoding="utf-8"))
    calls = [
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "hash" not in calls, "rng.py 里出现了内置 hash() 调用"


def test_derive_seed_is_stable_and_varies_by_label() -> None:
    assert rng.derive_seed("a", 0) == rng.derive_seed("a", 0)
    assert rng.derive_seed("a", 0) != rng.derive_seed("b", 0)
    assert rng.derive_seed("a", 0) != rng.derive_seed("a", 1)


# ------------------------------------------------------------------ schema


def test_edit_inverse_covers_every_add_del_pair() -> None:
    for kind, inv in EDIT_INVERSE.items():
        assert EDIT_INVERSE[inv] == kind, f"{kind} 与 {inv} 不是互逆对"


def test_edit_inverted_flips_kind_not_only_values() -> None:
    """早先 `inverted()` 只交换 `old`/`new`，种类不动 —— 于是 `flag_del` 取反之后
    还是 `flag_del`，`apply` 会去删一个本来就不在的标志位。补丁看着条条齐全，
    实际少恢复了一处，而「可逆性」这条断言是唯一会响的地方。"""
    e = Edit("flag_del", "nodes.x.flags", "sliding_window", None)
    assert e.inverted().kind == "flag_add"
    assert e.inverted().new == "sliding_window"
    assert e.inverted().inverted() == e


def test_rehydrate_only_touches_tuple_addresses() -> None:
    assert rehydrate("tensors.x.shape", [1, 2]) == (1, 2)
    assert rehydrate("nodes.x.flags", ["a"]) == ("a",)
    assert rehydrate("topology.block_groups", [[0, 3]]) == ((0, 3),)
    assert rehydrate("topology.block_group_shape", ["all"]) == ("all",)
    # 非元组地址必须原样保留 —— 一刀切会把「载入的」与「构建的」反过来变成不相等
    assert rehydrate("topology.architectures", ["LlamaForCausalLM"]) == ["LlamaForCausalLM"]
    assert rehydrate("nodes.x.dims.hidden", 32) == 32
    assert rehydrate("nodes.x.shape", None) is None
    assert "topology.block_groups" in TUPLE_ADDRS


# ------------------------------------------------------------------ genome


def test_primary_covers_every_topology_node_tensor_and_edge(result) -> None:
    """`primary()` 的键必须**完整覆盖**结构事实。

    哪天给 Genome 加了新字段却忘了接进 `primary()`，`apply(a, diff(a,b)) ≡ b`
    会静默失效（少还原一个字段），而只有这条测试与 diff 的完备性会响。
    """
    for name in result.names():
        g = result.genome(name)
        keys = set(g.primary())
        for k, _v in g.topology:
            assert f"topology.{k}" in keys
        for n in g.nodes:
            assert f"nodes.{n.path}.kind" in keys
            assert f"nodes.{n.path}.variant" in keys
            assert f"nodes.{n.path}.flags" in keys
            for dk, _dv in n.dims:
                assert f"nodes.{n.path}.dims.{dk}" in keys
        for t in g.tensors:
            assert f"tensors.{t.path}.shape" in keys
            assert f"tensors.{t.path}.tied_with" in keys
        for e in g.edges:
            assert f"edges.{e.src}->{e.dst}:{e.label}" in keys


def test_primary_excludes_derived_fields(result) -> None:
    """派生量与标签**不进**主地址空间：`cost` 是它的函数，`name`/`identity`/`dialect`
    是标签。把标签放进去会让「换个文件名」看起来像「结构变了」。"""
    for name in result.names():
        g = result.genome(name)
        keys = set(g.primary())
        for field in DERIVED_GENOME_FIELDS:
            assert not any(k.startswith(f"{field}.") or k == field for k in keys), field
        # 张量角色是路径的纯函数，也刻意不进（冗余事实会让一处改动变成两条编辑）
        assert not any(k.endswith(".role") for k in keys)


def test_fingerprint_levels_separate_scale_from_structure(tiny) -> None:
    """两级指纹的核心断言：**规模变 ⟹ 骨架指纹不变、全指纹变**；
    **结构变 ⟹ 骨架指纹也变**。没有阈值，这是定义。"""
    base = tiny["base"]
    wide = tiny["wide"]
    assert genome_mod.shape_fingerprint(base) == genome_mod.shape_fingerprint(wide)
    assert genome_mod.fingerprint(base) != genome_mod.fingerprint(wide)

    for key in ("moe", "window", "layernorm", "untied", "gqa"):
        other = tiny[key]
        assert genome_mod.shape_fingerprint(base) != genome_mod.shape_fingerprint(other), key
        assert genome_mod.fingerprint(base) != genome_mod.fingerprint(other), key


def test_genome_round_trip_through_json_is_bit_identical(result) -> None:
    """**产物载入 ≡ 内存构建（逐位）**。

    JSON 存不下元组，所以 `shape` / `flags` / `block_groups` 读回来会变成列表 ——
    而 `(4096, 4096) != [4096, 4096]`。这一条测试是 `schema.rehydrate` 的守门人：
    漏掉任何一类元组地址，它就会在这里失败。
    """
    for name in result.names():
        g = result.genome(name)
        again = genome_mod.load(json.loads(json.dumps(genome_mod.dump(g), ensure_ascii=False)))
        assert again.primary() == g.primary(), name
        assert genome_mod.fingerprint(again) == genome_mod.fingerprint(g), name
        assert genome_mod.shape_fingerprint(again) == genome_mod.shape_fingerprint(g), name


def test_cost_is_recomputable_after_apply(tiny) -> None:
    """「代价不是独立事实」：`apply` 重建 Genome 之后必须能逐位复算代价。

    但**只有从地址空间推得出来的那部分**能复算：`params_formula` 与
    `params_published_b` 是外部旁证（来自 config 与公开值），原样带过来。
    把它们也要求「复算相等」是错的 —— 它们是**另一条路**的结果，
    这正是三路互校能成立的原因。
    """
    a, b = tiny["base"], tiny["wide"]
    d = structural_diff(a, b)
    rebuilt = apply(a, d)
    assert rebuilt.cost.params == b.cost.params
    assert rebuilt.cost.flops_per_token == b.cost.flops_per_token
    assert rebuilt.cost.kv_cache_bytes == b.cost.kv_cache_bytes
    assert rebuilt.cost.weight_bytes == b.cost.weight_bytes
    assert rebuilt.cost.breakdown == b.cost.breakdown
    assert rebuilt.cost.params_formula == a.cost.params_formula  # 外部旁证，原样带过来
    assert rebuilt.cost.params != a.cost.params


def test_same_primary_and_primary_diff_keys_agree(tiny) -> None:
    """调试工具与断言必须用同一套判据。早先 `primary_diff_keys` 用 `.get()` 比较，
    把「键缺席」和「值为 None」看成一样，正好吞掉了「幽灵键」那一类差异。"""
    a, b = tiny["base"], tiny["moe"]
    assert (genome_mod.primary_diff_keys(a, b) == ()) == genome_mod.same_primary(a, b)
    assert not genome_mod.same_primary(a, b)
    assert genome_mod.same_primary(a, a)


@pytest.mark.parametrize("key", ["base", "wide", "moe", "window", "untied", "layernorm", "gqa"])
def test_invert_invert_is_identity(tiny, key) -> None:
    d = structural_diff(tiny["base"], tiny[key])
    assert invert(invert(d)) == d
