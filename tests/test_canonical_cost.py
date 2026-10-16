"""规范 IR 与代价。

这两件事放在一个文件里，是因为它们**互为证据**：

- `CanonicalIR` 是「清单 → 结构」的归一结果；
- `Cost` 是「结构 → 数字」的推演，而且**有两条独立的路**：
  ① 张量表求和（`params_from_schema`）、② config 标量公式（`params_from_formula`）。
  第 ③ 路是真实 checkpoint 元数据（`safetensors.total − buffers`），
  它不进这个文件 —— 它是**外部**证据，由 `test_zoo_carries_checkpoint_ground_truth` 与
  `test_three_way_cost_crosscheck_is_bit_exact` 负责。

①和②必须逐位相同。对不上就说明公式写错了，而**报告只会显示「参数量的两个来源不一致」**，
指不出是哪一项 —— 所以这里逐项拆开验。
"""

from __future__ import annotations

import pytest

from modelgenomefactory import cost as cost_mod
from modelgenomefactory.canonical import CanonicalError, block_groups_of, canonicalize


# ------------------------------------------------------------------ 块组


def test_block_groups_split_on_the_first_difference() -> None:
    """块组只按「相邻块的签名是否相同」切，不猜语义。

    DeepSeek-V2 的第 0 块是稠密 FFN、其余是 MoE —— 签名不同，所以切出两组。
    这正是「首块例外」能被表达出来的地方。
    """
    assert block_groups_of({0: "dense", 1: "moe", 2: "moe", 3: "moe"}) == [(0, 1), (1, 3)]
    assert block_groups_of({i: "same" for i in range(5)}) == [(0, 5)]


def test_block_groups_reject_an_empty_block_set() -> None:
    with pytest.raises(CanonicalError, match="一个块都没有"):
        block_groups_of({})


def test_block_groups_do_not_require_contiguous_indices() -> None:
    """**不连续的层号不是错误。** 有些发布的清单会跳号（被裁掉的层）。

    早先这里断言 `idxs == list(range(len(idxs)))`，于是跳号的清单被拒了 ——
    而「跳号」和「结构不一致」是两件事。分组只关心**相邻块的签名**。
    """
    assert block_groups_of({0: "a", 1: "a", 5: "b"}) == [(0, 2), (5, 1)]


# ------------------------------------------------------------------ 代价 ①②


def test_two_cost_paths_agree_bit_for_bit(result) -> None:
    """① 张量表求和 ≡ ② config 公式，**逐位**。"""
    for name in result.names():
        c = result.genome(name).cost
        assert c.params == c.params_formula, (
            f"{name}: 表={c.params:,} 公式={c.params_formula:,} 差 {c.params - c.params_formula:,}"
        )


def test_cost_breakdown_sums_to_params(result) -> None:
    """分项之和必须等于总量 —— 否则分解图会缺一块而没人发现。"""
    for name in result.names():
        c = result.genome(name).cost
        assert sum(v for _k, v in c.breakdown) == c.params, name


def test_published_params_is_within_one_percent(result) -> None:
    """外部声明（`params_b`）与实测参数量：**只做量级核对**，不做逐位断言。

    zoo 里写的是「多少 B」这种两位有效数字的公开口径，逐位比会把
    「四舍五入」误报成「读错了」。真正的逐位核对在第 ③ 路（safetensors 元数据）。
    """
    for name in result.names():
        c = result.genome(name).cost
        if c.params_published_b is None:
            continue
        published = c.params_published_b * 1e9
        assert abs(c.params - published) / published < 0.01, (name, c.params, published)


def test_kv_cache_is_a_first_class_column(result) -> None:
    """KV cache 是 MHA/GQA/MLA 的**后果**，不是可以省略的脚注。

    - GQA 的 KV 头比 Q 头少，所以缓存更小：`llama-3-8b`（8 KV 头）比
      同样 32 头的 MHA 模型小 4 倍；
    - MLA 缓存的是**压缩潜向量**（`kv_lora_rank + qk_rope_head_dim`），
      所以 `deepseek-v2-lite` 的缓存与它的层数/宽度不成简单比例。
    """
    a = result.genome("llama-2-7b").cost.kv_cache_bytes
    b = result.genome("llama-3-8b").cost.kv_cache_bytes
    assert b < a, "GQA 的 KV 缓存必须比同规模 MHA 小"
    # 32 头 / 8 KV 头 = 4 倍；层数一样，所以比值就是 4。
    assert a == 4 * b, (a, b)

    mla = result.genome("deepseek-v2-lite").cost.kv_cache_bytes
    assert 0 < mla < a, "MLA 的压缩缓存应当比 MHA 小"


def test_active_params_counts_experts_with_top_k(result) -> None:
    """MoE 的「激活参数量」按 `top_k` 数专家，不是全部专家。

    Mixtral-8x7B 的 46.7B 总参数里每次只用 12.9B —— 这个差额正是
    「总参数量」和「激活参数量」必须分成两列的原因。
    """
    dense = result.genome("mistral-7b")
    moe = result.genome("mixtral-8x7b")
    assert cost_mod.active_params(dense) == dense.cost.params
    assert cost_mod.active_params(moe) < moe.cost.params
    # 共享专家**总是**参与：DeepSeek-V2 的激活量必须严格大于「只算 top_k 个专家」。
    deep = result.genome("deepseek-v2-lite")
    assert cost_mod.active_params(deep) < deep.cost.params


def test_cost_is_recomputable_from_the_address_space(tiny) -> None:
    """代价是**主地址空间的函数** —— 所以 `apply` 重建的 Genome 必须算出同一个代价。

    唯一不算的是 `params_formula` 与 `params_published_b`：前者是「另一条路的独立结论」，
    后者是外部声明，两者都**不是**地址空间的函数。把它们一起比会得出
    「重建后的代价变了」这种假警报。
    """
    from modelgenomefactory.diff import apply, structural_diff

    a, b = tiny["base"], tiny["wide"]
    rebuilt = apply(a, structural_diff(a, b))
    for field in ("params", "flops_per_token", "kv_cache_bytes", "weight_bytes", "breakdown"):
        assert getattr(rebuilt.cost, field) == getattr(b.cost, field), field
    assert rebuilt.cost.params_formula == a.cost.params_formula, "公式结论是外部证据，不该被重建改写"
    assert rebuilt.cost.seq_len == b.cost.seq_len


def test_flops_scale_with_seq_len(tiny_of) -> None:
    """FLOPs 是**口径的函数**：seq 翻倍，注意力项翻倍，权重项不变。

    「FLOPs 是多少」在没有「@seq=2048」的情况下没有意义 —— 所以口径存在 `Cost` 里，
    而不是当常数用。
    """
    g1 = tiny_of(seq_len=128)
    g2 = tiny_of(seq_len=256)
    assert g2.cost.flops_per_token > g1.cost.flops_per_token
    assert g2.cost.weight_bytes == g1.cost.weight_bytes
    assert g2.cost.kv_cache_bytes == 2 * g1.cost.kv_cache_bytes
    # 参数量与口径无关 —— 它只是权重表的大小。
    assert g2.cost.params == g1.cost.params


# ------------------------------------------------------------------ 清单 → IR


def test_every_dialect_gives_the_same_ir_signature(result) -> None:
    """同一个模型的不同方言必须归一到**同一份**结构签名。

    这是「指纹与命名无关」的直接证据：方言只影响 `source_name`，不影响任何结构事实。
    """
    for name in result.names():
        sigs = set()
        for d in result.dialects_of(name):
            ir = result.ir(name, d)
            sigs.add(
                (
                    tuple(sorted(ir.topology)),
                    tuple((n.path, n.kind, n.variant, tuple(n.flags), tuple(sorted(n.dims))) for n in ir.nodes),
                    tuple((t.path, tuple(t.shape), t.tied_with) for t in ir.tensors),
                    tuple(sorted((e.src, e.dst, e.label) for e in ir.edges)),
                )
            )
        assert len(sigs) == 1, f"{name} 的方言给出了 {len(sigs)} 种结构签名"


def test_crosschecks_are_recorded_and_all_agree(result) -> None:
    """两路互校的**每一项**都要留下痕迹，而且全部一致。

    `crosschecks` 是 `(事实, config 说, 清单量出)` 的三元组。它是「读不懂但假装
    读懂了」的唯一防线：数量必须够多（否则等于没查），而且不许有一条不一致。
    """
    total = 0
    for name in result.names():
        items = result.crosschecks(name)
        assert len(items) >= 20, f"{name} 只有 {len(items)} 项互校，太少了"
        for fact, said, measured in items:
            assert str(said) == str(measured), f"{name}/{fact}: config 说 {said}，清单量出 {measured}"
            total += 1
    assert total >= 400, f"全部互校只有 {total} 项"


def test_crosscheck_rejects_a_config_that_disagrees_with_the_manifest(zoo, root, facts_of) -> None:
    """互校**真的会拦人**，不是装饰。

    把 config 的层数改小，而清单还是原来的 —— 必须报错，且错误信息要点名层数。
    """
    from modelgenomefactory.extract import extract_model
    from modelgenomefactory.pipeline import load_config_file, manifest_path

    entry = {e.name: e for e in zoo}["llama-2-7b"]
    raw = facts_of(extract_model(entry), n_layers=16)
    manifest = load_config_file(manifest_path(root, "llama-2-7b", "hf"))
    with pytest.raises(CanonicalError, match="config 说 16 层，清单里有"):
        canonicalize(raw, manifest, name="llama-2-7b", dialect="hf")


def test_manifest_render_uses_config_scalars_only(zoo) -> None:
    """渲染清单的那条路**完全不看参数名** —— 它从 config 标量算出来。

    所以它才能和解析路互校：两条路唯一的共同输入是「这个模型是什么」。
    """
    from modelgenomefactory.extract import (
        block_groups_for,
        expected_tensors,
        extract_model,
        load_config_file,
        position_kind_of,
    )

    for e in zoo:
        raw = extract_model(e)
        tensors = expected_tensors(raw, position_kind=position_kind_of(load_config_file(e.config_path)))
        # 渲染用的张量表里**不含**任何源名字 —— 只有规范路径。
        for t in tensors:
            assert set(t) == {"path", "shape", "tied_with"}, t
            assert "model.layers" not in t["path"], t
        groups = block_groups_for(raw)
        assert sum(c for _s, c in groups) == int(raw.require("n_layers")), e.name
