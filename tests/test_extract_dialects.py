"""抽取层与命名方言层。

这一层要做的事只有一件：**把「名字」变成「结构」，并且拒绝任何看不懂的东西。**
所以测试的重点是「覆盖」与「拒绝」，不是「跑得通」：

- 每个 config 键要么被归一化，要么**带着理由**被忽略 —— 两栏分开，不合并；
- 每条命名规则都必须**被某份清单用到**（死规则 = 一个没人验过的猜测）；
- 渲染 → 解析必须回到同一组规范路径（否则「换个方言指纹不变」无从谈起）。
"""

from __future__ import annotations

import pytest

from modelgenomefactory import dialects as dl
from modelgenomefactory.canonical import canonicalize
from modelgenomefactory.extract import extract_model


# ------------------------------------------------------------------ zoo


def test_zoo_has_thirteen_models_with_contiguous_chains(zoo) -> None:
    assert len(zoo) == 13
    by_family: dict[str, list[int]] = {}
    for e in zoo:
        by_family.setdefault(e.family, []).append(e.chain)
    for family, chains in by_family.items():
        assert sorted(chains) == list(range(1, len(chains) + 1)), family


def test_every_model_declares_at_least_two_dialects(zoo) -> None:
    for e in zoo:
        assert len(e.dialects) >= 2, e.name
        assert e.has_dialect("hf"), f"{e.name} 必须有 hf —— 真实发布清单只有这一种"
        for _d, layout in e.dialects:
            assert layout in dl.LAYOUTS, (e.name, layout)


def test_every_model_carries_checkpoint_ground_truth(zoo) -> None:
    """外部真值必须每条都有，否则「三路互校」会在某个模型上退化成两路，
    而报告不会说哪一路缺席。"""
    for e in zoo:
        ck = dict(e.checkpoint or {})
        assert "safetensors_total" in ck, e.name
        assert "params" in ck, e.name
        assert int(ck["params"]) + int(ck.get("buffers") or 0) == int(ck["safetensors_total"]), e.name


# ------------------------------------------------------------------ config 归一化


def test_every_config_key_is_either_normalized_or_ignored_with_a_reason(zoo) -> None:
    """**「读不懂」与「决定不看」必须分两栏。**

    合并成一栏是很容易犯的错：把 dropout 和一个真正陌生的键一起报成「未识别」，
    报告会显示一个假的低覆盖率；反过来把它们都算成已处理，又会把
    「读不懂但假装读懂了」这个最危险的失败模式藏起来。
    """
    for e in zoo:
        raw = extract_model(e)
        assert raw.unknown_keys == (), f"{e.name} 有读不懂的键：{raw.unknown_keys}"
        assert raw.ignored_keys, f"{e.name} 一个键都没忽略？说明忽略表根本没生效"
        for key, reason in raw.ignored_keys:
            assert reason.strip(), f"{e.name}/{key} 的忽略理由为空"


def test_conventions_are_recorded_with_their_own_provenance(zoo) -> None:
    """家族约定填进去的每一项都必须**标明来源** —— 否则它看起来就像 config 说的。"""
    by_name = {e.name: e for e in zoo}
    assert dict(by_name["qwen2-7b"].conventions)["qkv_bias"] is True
    raw = extract_model(by_name["qwen2-7b"])
    rules = {f.key: f.rule for f in raw.facts}
    assert rules["qkv_bias"] == "zoo:convention:qkv_bias"
    assert rules["o_bias"] == "zoo:convention:o_bias"


def test_position_kind_is_a_config_fact(zoo) -> None:
    by_name = {e.name: e for e in zoo}
    assert extract_model(by_name["gpt2"]).get("position_kind") == "learned"
    assert extract_model(by_name["llama-2-7b"]).get("position_kind") == "rope"


def test_norm_kind_is_config_only(zoo) -> None:
    """Llama 的键名叫 `input_layernorm`，实际是 RMSNorm —— 张量清单**给不出**这个信息。
    所以它必须是 config 事实，并被显式标出来，而不是从名字猜。"""
    by_name = {e.name: e for e in zoo}
    assert extract_model(by_name["llama-2-7b"]).get("norm_kind") == "rmsnorm"
    assert extract_model(by_name["gpt2"]).get("norm_kind") == "layernorm"


def test_facts_are_sorted_and_traceable(zoo) -> None:
    for e in zoo:
        raw = extract_model(e)
        keys = [f.key for f in raw.facts]
        assert keys == sorted(keys), e.name
        for f in raw.facts:
            assert f.rule, f"{e.name}/{f.key} 没有来源规则"


# ------------------------------------------------------------------ 渲染 ↔ 解析


def test_render_then_parse_returns_the_same_canonical_paths(zoo) -> None:
    """渲染 → 解析必须回到**同一组规范路径**。

    这是「换个命名方言，指纹不变」的地基：如果这一圈丢了或多了张量，
    指纹就会变，而报告只会说「这两个方言不一致」，指不出是哪一张。

    **两边不在同一个空间里**，这是第一次写这条测试时踩的坑：规范 IR 是**模式**
    （`blocks[*]`，32 层压成一条），清单是**实例**（`model.layers.7.*`）。
    直接比会得到「`blocks[*]` ≠ `blocks[0]`」，看起来像失败，其实两边说的是同一件事。
    所以先把 IR 按同一组 `block_groups` / `n_experts` 展开，再比。
    """
    from modelgenomefactory.extract import block_groups_for, expected_tensors, load_config_file, position_kind_of

    for e in zoo:
        raw = extract_model(e)
        cfg_raw = load_config_file(e.config_path)
        tensors = expected_tensors(raw, position_kind=position_kind_of(cfg_raw))
        groups = block_groups_for(raw)
        n_experts = int(raw.get("n_experts") or 0)
        for dialect, layout in e.dialects:
            manifest = dl.render_manifest(
                tensors, layout=layout, model=e.name, dialect=dialect,
                block_groups=groups, n_experts=n_experts,
            )
            parsed, _notes = dl.parse_manifest(
                manifest,
                n_heads=int(raw.get("n_heads")),
                n_kv_heads=int(raw.get("n_kv_heads")),
                head_dim=int(raw.get("head_dim")),
            )
            ir = canonicalize(raw, manifest, name=e.name, dialect=dialect)
            expanded = dl.expand_paths(
                [{"path": t.path, "shape": list(t.shape)} for t in ir.tensors],
                block_groups=groups,
                n_experts=n_experts,
            )
            assert {t["path"] for t in expanded} == {p["path"] for p in parsed}, (e.name, dialect)
            assert {tuple(t["shape"]) for t in expanded} == {tuple(p["shape"]) for p in parsed}, (e.name, dialect)


def _with_facts(raw, **overrides):
    """把指定事实换成探针值。实现在 `conftest.with_facts`（两个测试模块共用）。"""
    from conftest import with_facts

    return with_facts(raw, **overrides)


def _zoo_manifests(zoo) -> list[dict]:
    from modelgenomefactory.extract import load_config_file
    from modelgenomefactory.pipeline import manifest_path

    return [load_config_file(manifest_path(".", e.name, d)) for e in zoo for d, _l in e.dialects]


def test_every_name_rule_is_used_or_declared_conditional(zoo) -> None:
    """**死规则就是没人验过的猜测。**

    规则表里躺着一条永远匹配不上的规则，看起来像「支持了某种布局」，
    实际上它对结构没有任何影响 —— 而某天它被匹配上时，行为是未经验证的。

    所以每条规则必须落在两栏之一，**不允许第三栏**：

    - **实测**：某份 zoo 清单的解析路径上真的走到过它；
    - **条件**：只在某个门槛（config 事实 / 清单标志）成立时才会出现，
      且 `test_conditional_rules_are_reachable` 有探针把它真的产出来。

    这条测试第一次跑就抓到了两类问题，都不是笔误：

    1. **声明过宽**：MLA 的 8 条规则挂在 `llama` 前缀下、MoE 的 7 条挂在 `llama` 下、
       稠密 FFN 的 3 条挂在 `mixtral` 下、k/v 投影的 2 条挂在 `deepseek` 下 ——
       而 zoo 里没有「llama 布局的 MLA 模型」「llama 布局的 MoE 模型」
       「mixtral 布局的稠密 FFN」「deepseek 布局的未压缩 KV」。
    2. **结构性死代码**：自研方言的 `layers.{i}.attn.Wqkv` 与 `layers.{i}.mlp.W12`。
       渲染方向先试一对一规则、再试融合规则，而 `custom_v1` 同时有
       `Wq/Wk/Wv` 与 `Wgate/Wup/Wdown`，所以那两条融合规则**永远轮不到**。
       它们不是「备用方言」，是删掉更诚实的死代码。
    """
    cov = dl.rule_coverage(_zoo_manifests(zoo))
    dead = {layout: info["dead"] for layout, info in cov.items() if info["dead"]}
    assert not dead, f"有规则从未被任何清单用到（死规则）：{dead}"
    # 反向也要查：登记成「条件」的必须真的没被走到，否则这张表就成了藏死规则的抽屉。
    for layout, info in cov.items():
        for src in info["conditional"]:
            assert src in dl.CONDITIONAL_RULES, f"{layout} 的 {src} 既没实测也没登记门槛"
    total = sum(i["n_rules"] for i in cov.values())
    classified = sum(len(i["exercised"]) + len(i["conditional"]) for i in cov.values())
    assert total == classified, "有规则没被分到任何一栏"


# --------------------------------------------------- 条件规则：逐条探针

#: **渲染方向**探针：`(门槛, 模型, 方言, 布局, 要翻的事实, 期望渲染出来的源名字)`。
#:
#: 探针是**真的把门槛翻到成立、再走一遍渲染**，不是「我相信它会出来」。
_RENDER_PROBES: tuple[tuple[str, str, str, str, dict, tuple[str, ...]], ...] = (
    (
        "qkv_bias", "mixtral-8x7b", "hf", "mixtral", {"qkv_bias": True},
        (
            "model.layers.{i}.self_attn.q_proj.bias",
            "model.layers.{i}.self_attn.k_proj.bias",
            "model.layers.{i}.self_attn.v_proj.bias",
        ),
    ),
    (
        "o_bias", "llama-2-7b", "hf", "llama", {"o_bias": True},
        ("model.layers.{i}.self_attn.o_proj.bias",),
    ),
    (
        "mlp_bias", "llama-2-7b", "hf", "llama", {"mlp_bias": True},
        (
            "model.layers.{i}.mlp.gate_proj.bias",
            "model.layers.{i}.mlp.up_proj.bias",
            "model.layers.{i}.mlp.down_proj.bias",
        ),
    ),
    (
        "mlp_bias", "llama-2-7b", "vllm", "vllm_llama", {"mlp_bias": True},
        ("model.layers.{i}.mlp.gate_up_proj.bias", "model.layers.{i}.mlp.down_proj.bias"),
    ),
    (
        "o_bias", "llama-2-7b", "vllm", "vllm_llama", {"o_bias": True},
        ("model.layers.{i}.self_attn.o_proj.bias",),
    ),
    (
        "q_lora_rank", "deepseek-v2-lite", "hf", "deepseek", {"q_lora_rank": 512},
        (
            "model.layers.{i}.self_attn.q_a_proj.weight",
            "model.layers.{i}.self_attn.q_b_proj.weight",
            "model.layers.{i}.self_attn.q_a_layernorm.weight",
        ),
    ),
    (
        "q_lora_rank", "deepseek-v2-lite", "custom", "custom_v1", {"q_lora_rank": 512},
        ("layers.{i}.attn.Wqa", "layers.{i}.attn.Wqb", "layers.{i}.attn.Nqa"),
    ),
    # `n_experts`：把 Mixtral 的 MoE 张量渲染到 **llama 布局**上 —— 这正是
    # Qwen1.5-MoE / DeepSeek HF 清单的命名（llama 前缀 + MoE 的 MLP）。
    # 不翻任何事实：mixtral 本来就是 MoE，门槛天然成立；变的只是**布局**。
    (
        "n_experts", "mixtral-8x7b", "hf", "llama", {},
        (
            "model.layers.{i}.mlp.gate.weight",
            "model.layers.{i}.mlp.experts[{e}].gate_proj.weight",
            "model.layers.{i}.mlp.experts[{e}].up_proj.weight",
            "model.layers.{i}.mlp.experts[{e}].down_proj.weight",
        ),
    ),
)

#: **解析方向**探针：`(门槛, 布局, ((源名字模板, 规范路径), ...))`。
#:
#: 有些条件规则**渲染不出来**（zoo 里没有那个结构），但它们仍然是真实清单的词汇 ——
#: 这时唯一的验法是手写一份清单，看解析器认不认。「能解析出来」就是这条规则的全部内容。
#: 源名字写成**模板**（`{i}` / `{e}`），与 `CONDITIONAL_RULES` 的键一致；
#: 规范路径是**具体**的（`blocks[0]`），因为解析器要的就是具体下标。
_PARSE_PROBES: tuple[tuple[str, str, tuple[tuple[str, str], ...]], ...] = (
    (
        "qk_norm", "llama",
        (
            ("model.layers.{i}.self_attn.q_norm.weight", "blocks[0].attn.q_norm.weight"),
            ("model.layers.{i}.self_attn.k_norm.weight", "blocks[0].attn.k_norm.weight"),
        ),
    ),
    (
        "n_experts", "llama",
        (
            ("model.layers.{i}.mlp.shared_experts.gate_proj.weight",
             "blocks[0].ffn.shared_experts.gate_proj.weight"),
            ("model.layers.{i}.mlp.shared_experts.up_proj.weight",
             "blocks[0].ffn.shared_experts.up_proj.weight"),
            ("model.layers.{i}.mlp.shared_experts.down_proj.weight",
             "blocks[0].ffn.shared_experts.down_proj.weight"),
        ),
    ),
)


def test_conditional_rules_are_reachable(zoo) -> None:
    """**每一个条件规则都必须被至少一条探针覆盖。**

    「条件规则」如果只写在注释里，它就退化成了「我相信它会出来」—— 而那正是
    死规则的另一种说法。这条测试把门槛翻到成立，走一遍**真实的渲染路径**
    （或解析路径），断言期望的名字**确实出现**。

    只查「每个门槛有探针」是不够的：一个门槛下可能挂着 7 条规则，探针只验了 4 条，
    剩下 3 条照样没人管。所以断言落在**规则**上，不是门槛上。
    """
    from modelgenomefactory.extract import (
        block_groups_for,
        expected_tensors,
        load_config_file,
        position_kind_of,
    )

    by_name = {e.name: e for e in zoo}
    covered: set[str] = set()

    for gate, model, dialect, layout, overrides, expect in _RENDER_PROBES:
        entry = by_name[model]
        raw = _with_facts(extract_model(entry), **overrides)
        tensors = expected_tensors(raw, position_kind=position_kind_of(load_config_file(entry.config_path)))
        manifest = dl.render_manifest(
            tensors, layout=layout, model=model, dialect=dialect,
            block_groups=block_groups_for(raw), n_experts=int(raw.get("n_experts") or 0),
        )
        produced = {t["name"] for t in manifest["tensors"]}
        for want in expect:
            # 规则表里写的是模板（`{i}` 是层号、`{e}` 是专家号），渲染出来的是具体名字。
            concrete = want.replace("{i}", "0").replace("{e}", "0")
            assert concrete in produced, (
                f"门槛 {gate} 成立时 {want} 仍没被渲染出来（探针：{model}/{dialect}→{layout}）"
            )
            assert dl.CONDITIONAL_RULES.get(want) == gate, f"{want} 的门槛登记不是 {gate}"
            covered.add(want)

    for gate, layout, pairs in _PARSE_PROBES:
        handwritten = {
            "model": "probe", "dialect": "hf", "layout": layout,
            "tensors": [{"name": src.replace("{i}", "0").replace("{e}", "0"), "shape": [128]} for src, _c in pairs],
            "absent_but_tied": [],
        }
        parsed, _notes = dl.parse_manifest(handwritten, n_heads=8, n_kv_heads=8, head_dim=16)
        paths = {p["path"] for p in parsed}
        for src, canon in pairs:
            assert canon in paths, f"解析方向认不出 {src}（期望规范路径 {canon}）"
            assert dl.CONDITIONAL_RULES.get(src) == gate, f"{src} 的门槛登记不是 {gate}"
            covered.add(src)

    # 覆盖率断言：**每条条件规则都要被覆盖**，一个都不许漏。
    registered = set(dl.CONDITIONAL_RULES)
    uncovered = registered - covered
    assert not uncovered, f"这些条件规则没有任何探针：{sorted(uncovered)}"
    assert not covered - registered, f"探针验了没登记的规则：{sorted(covered - registered)}"



def test_fused_split_rejects_wrong_width() -> None:
    """融合切分必须按 `(n_heads+2·n_kv)·head_dim` 校验宽度 —— 这是**唯一**
    能挡住「转置过的融合权重」的地方，而转置过的权重在形状上「看着像」。"""
    with pytest.raises(dl.ManifestError):
        dl._split_fused("qkv", "blocks[0].attn.qkv_proj.weight", (4096, 3072), 32, 8, 128)
    # 正确宽度： (32 + 2·8)·128 = 6144
    parts = dl._split_fused("qkv", "blocks[0].attn.qkv_proj.weight", (4096, 6144), 32, 8, 128)
    assert [p[0] for p in parts] == [
        "blocks[0].attn.q_proj.weight",
        "blocks[0].attn.k_proj.weight",
        "blocks[0].attn.v_proj.weight",
    ]
    assert [p[1][-1] for p in parts] == [4096, 1024, 1024]


def test_parse_manifest_rejects_an_unknown_name() -> None:
    manifest = {
        "model": "x", "dialect": "hf", "layout": "llama",
        "tensors": [{"name": "totally.made.up.name", "shape": [8]}],
        "absent_but_tied": [],
    }
    with pytest.raises(dl.ManifestError, match="没有规则能解释参数名"):
        dl.parse_manifest(manifest, n_heads=4, n_kv_heads=4, head_dim=8)


def test_parse_manifest_rejects_an_unknown_layout() -> None:
    with pytest.raises(dl.ManifestError, match="未知布局"):
        dl.parse_manifest({"layout": "nope", "tensors": []}, n_heads=1, n_kv_heads=1, head_dim=1)


def test_expand_paths_does_not_reexpand_concrete_indices() -> None:
    """早先的版本对任何 `experts[...]` 都重新展开，于是 `experts[3]` 被改写成
    `experts[0..7]` —— 一份 IR 里的 8 个专家变成 64 个，而且全挤到同一个融合桶里。"""
    out = dl.expand_paths(
        [{"path": "blocks[*].ffn.experts[3].gate_proj.weight", "shape": [4, 4]}],
        block_groups=[(0, 2)],
        n_experts=8,
    )
    assert sorted(o["path"] for o in out) == [
        "blocks[0].ffn.experts[3].gate_proj.weight",
        "blocks[1].ffn.experts[3].gate_proj.weight",
    ]


def test_expand_paths_expands_the_star_wildcard_once() -> None:
    out = dl.expand_paths(
        [{"path": "blocks[*].ffn.experts[*].gate_proj.weight", "shape": [4, 4]}],
        block_groups=[(0, 2)],
        n_experts=3,
    )
    assert len(out) == 2 * 3
    assert len({o["path"] for o in out}) == 6
