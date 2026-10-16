"""Canonicalizer：清单 + config → Canonical Model IR。

两条独立路径在这里汇合，**必须互校**：

| 路径 | 来源 | 能看到什么 | 看不到什么 |
| --- | --- | --- | --- |
| A：清单 | 参数名 + 形状 | 张量契约、层数、宽度、头结构、专家数 | 范数类型、激活、RoPE base、滑窗、top-k |
| B：config | 标量 | 上面那一列 | 张量的存在性与形状 |

重叠的部分（层数 / 宽度 / 头数 / 词表 / 专家数 / 权重绑定 / 偏置）逐条比对，
不一致就**报错**，并把每条比对结果记进 `CanonicalIR.crosschecks` 供报告打印。
这不是形式主义：`tie_word_embeddings` 这种键缺席时只能以清单为准，
而 `n_kv_heads` 这种键缺席时只能按 HF 的约定补默认值 —— 两条路各自都有盲区，
只有摆在一起才看得出哪一边在瞎猜。

## 块组（block group）

13 个标本里 12 个是「整堆同构」，只有 DeepSeek-V2-Lite 的**第 0 块是稠密、其余 26 块是
MoE**（`first_k_dense_replace=1`）。所以路径不能只有一种模式：

- 成员最多的那一组叫 `blocks[*]`；
- 其余组：单个的用 `blocks[3]`，多个的用 `blocks[3:6]`。

带计数的 `block_groups` 进全指纹，**去计数的** `block_group_shape` 才进骨架 ——
于是「1+26 块」与「1+3 块」是量变，「整堆同构」与「首块例外」才是突变。
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping, Sequence

from .dialects import ManifestError, parse_manifest, role_of
from .schema import (
    CanonicalIR,
    Edge,
    OpNode,
    RawStructure,
    TensorEntry,
)

__all__ = ["CanonicalError", "canonicalize", "block_groups_of", "pattern_of"]

_BLOCK_RE = re.compile(r"^blocks\[(\d+)\]\.(.+)$")
_BLOCK_TOKEN_RE = re.compile(r"blocks\[[^\]]+\]")


class CanonicalError(ValueError):
    """两条路径对不上，或结构自相矛盾。**一律报错**。"""


# ---------------------------------------------------------------- 块组


def block_groups_of(signatures: Mapping[int, Any]) -> list[tuple[int, int]]:
    """把「每块的张量签名」切成连续且相等的组。"""
    idxs = sorted(signatures)
    if not idxs:
        raise CanonicalError("一个块都没有")
    groups: list[tuple[int, int]] = []
    start, prev, count = idxs[0], signatures[idxs[0]], 1
    for i in idxs[1:]:
        if signatures[i] == prev:
            count += 1
        else:
            groups.append((start, count))
            start, prev, count = i, signatures[i], 1
    groups.append((start, count))
    return groups


def _markers(groups: Sequence[tuple[int, int]], n_layers: int) -> dict[tuple[int, int], str]:
    """给每组分配路径标记。成员最多的那组拿 `*`（并列时取靠前的）。"""
    best = max(groups, key=lambda g: (g[1], -g[0]))
    out: dict[tuple[int, int], str] = {}
    for g in groups:
        start, count = g
        if g == best:
            out[g] = "*"
        elif count == 1:
            out[g] = str(start)
        else:
            out[g] = f"{start}:{start + count}"
    return out


def pattern_of(block_index: int, groups: Sequence[tuple[int, int]], markers: Mapping[tuple[int, int], str]) -> str:
    for g in groups:
        start, count = g
        if start <= block_index < start + count:
            return markers[g]
    raise CanonicalError(f"块 {block_index} 不在任何组 {list(groups)} 里")


# ---------------------------------------------------------------- 主入口


def canonicalize(
    raw: RawStructure,
    manifest: Mapping[str, Any],
    *,
    name: str,
    dialect: str,
) -> CanonicalIR:
    """清单 + config → Canonical IR。"""
    f = raw.as_map()
    checks: list[tuple[str, str, str]] = []
    notes: list[str] = []

    def need(key: str) -> Any:
        if key not in f:
            raise CanonicalError(f"{name}: config 缺少 {key!r}")
        return f[key]

    n_layers = int(need("n_layers"))
    hidden = int(need("hidden"))
    vocab = int(need("vocab"))
    n_heads = int(need("n_heads"))
    n_kv_heads = int(need("n_kv_heads"))
    head_dim = int(need("head_dim"))
    intermediate = int(need("intermediate"))
    norm_kind = str(need("norm_kind"))
    norm_eps = float(need("norm_eps"))
    max_positions = int(need("max_positions"))
    n_experts = int(f.get("n_experts") or 0)
    top_k = int(f.get("top_k") or 0)
    n_shared = int(f.get("n_shared_experts") or 0)
    moe_intermediate = f.get("moe_intermediate")
    # Mixtral 系没有 `moe_intermediate_size`：它的 `intermediate_size` 就是专家宽度。
    # 回退之后「专家宽度」这一条互校对 Mixtral 也**生效**了（不回退的话它会静默跳过）。
    expert_width = int(moe_intermediate) if moe_intermediate else (intermediate if n_experts else None)
    first_k_dense = int(f.get("first_k_dense") or 0)
    n_groups = int(f.get("n_groups") or 1)
    topk_group = int(f.get("topk_group") or 1)
    rope_theta = f.get("rope_theta")
    rope_scaling = f.get("rope_scaling")
    sliding_window = f.get("sliding_window")
    use_sliding_window = f.get("use_sliding_window", True)
    max_window_layers = f.get("max_window_layers")
    tie_declared = f.get("tie_embeddings")
    position_kind = f.get("position_kind")
    # 偏置三个分量各自的**声明值**。为 `None` 就表示「config 与 zoo 约定都没表态」，
    # 此时不做互校（而不是假设一个值）。Qwen1.5/Qwen2 靠 zoo 约定把 `qkv_bias`
    # 填成 `True`，于是「官方 checkpoint 里 q/k/v 有偏置」这件事变成了一条被检查的断言。
    qkv_bias = f.get("qkv_bias")
    o_bias = f.get("o_bias")
    mlp_bias = f.get("mlp_bias")
    q_lora_rank = f.get("q_lora_rank")
    kv_lora_rank = f.get("kv_lora_rank")
    qk_nope = f.get("qk_nope_head_dim")
    qk_rope = f.get("qk_rope_head_dim")
    v_head_dim = f.get("v_head_dim")

    # ---- 路径 A：清单 ----
    try:
        concrete, parse_notes = parse_manifest(
            manifest, n_heads=n_heads, n_kv_heads=n_kv_heads, head_dim=head_dim
        )
    except ManifestError as exc:
        raise CanonicalError(f"{name}/{dialect}: {exc}") from exc
    notes.extend(parse_notes)

    def _check(label: str, cfg: Any, got: Any) -> None:
        checks.append((label, _fmt(cfg), _fmt(got)))
        if cfg != got:
            raise CanonicalError(f"{name}: {label} 两路对不上 —— config 说 {cfg!r}，清单量出 {got!r}")

    by_path = {t["path"]: t for t in concrete}
    if "embed.weight" not in by_path:
        raise CanonicalError(f"{name}: 清单里没有 embed.weight")

    # ---- 顶层张量 ----
    embed = by_path["embed.weight"]
    _check("词表大小 vocab", vocab, embed["shape"][0])
    _check("隐藏宽度 hidden", hidden, embed["shape"][1])

    has_pos = "pos_embed.weight" in by_path
    if has_pos:
        _check("位置嵌入长度", max_positions, by_path["pos_embed.weight"]["shape"][0])
    # 位置编码类型：`learned` 就**必须**有位置表，`rope` 就**必须**没有。
    # 这条检查的价值在于它把「渲染期的启发式」变成了「可证伪的断言」：配置说 RoPE
    # 而清单里躺着 `pos_embed.weight`，那就是渲染器和解析器对不上，必须报错。
    if position_kind:
        _check("位置编码类型（learned⇒有位置表）", position_kind == "learned", has_pos)

    head = by_path.get("head.weight")
    if head is None:
        raise CanonicalError(f"{name}: 清单里没有 head.weight（也没有 tied 声明）")
    tie_actual = head["tied_with"] is not None
    if tie_declared is not None:
        _check("权重绑定 tie_word_embeddings", bool(tie_declared), tie_actual)
    else:
        notes.append("config 没有声明 tie_word_embeddings，以清单为准（GPT-2 就是这种情况）")

    # ---- 分块 ----
    # 这里**保留**专家的具体下标（`ffn.experts[3].gate_proj.weight`），不折成 `experts[*]`。
    # 折了就没法回答「第 7 号专家是不是和第 0 号一样大」—— 而「专家参差不齐」是真实存在
    # 的结构缺陷，恰恰不会自己报错。折叠只发生在**往外写张量**的那一步（`experts[*]`）。
    blocks: dict[int, dict[str, Any]] = {}
    for t in concrete:
        m = _BLOCK_RE.match(t["path"])
        if not m:
            continue
        idx = int(m.group(1))
        blocks.setdefault(idx, {})[m.group(2)] = t
    if sorted(blocks) != list(range(n_layers)):
        raise CanonicalError(f"{name}: config 说 {n_layers} 层，清单里有 {sorted(blocks)}")
    _check("层数 n_layers", n_layers, len(blocks))

    signatures = {i: tuple(sorted((k, tuple(v["shape"])) for k, v in blk.items())) for i, blk in blocks.items()}
    groups = block_groups_of(signatures)
    markers = _markers(groups, n_layers)
    shape = tuple("all" if c == n_layers else ("one" if c == 1 else "many") for _, c in groups)

    # ---- 算子变体判定（只看清单 + 少量 config）----
    rep: dict[str, dict[str, Any]] = {}
    for g in groups:
        start = g[0]
        rep[markers[g]] = blocks[start]

    nodes: list[OpNode] = []
    tensors: list[TensorEntry] = []

    def add_tensor(path: str, source_name: str, shape_: Sequence[int], tied: str | None, prov: str) -> None:
        tensors.append(
            TensorEntry(
                path=path,
                role=role_of(path),
                shape=tuple(shape_),
                tied_with=tied,
                provenance=prov,
                source_name=source_name,
            )
        )

    # 顶层张量
    add_tensor("embed.weight", embed["source_name"], embed["shape"], None, "both")
    if has_pos:
        pe = by_path["pos_embed.weight"]
        add_tensor("pos_embed.weight", pe["source_name"], pe["shape"], None, "both")
    add_tensor("head.weight", head["source_name"], head["shape"], head["tied_with"], "both")

    nodes.append(OpNode("embed", "embedding", "token", (), (("vocab", vocab), ("hidden", hidden)), "both"))
    if has_pos:
        nodes.append(
            OpNode("position", "position", "learned", (), (("max_positions", max_positions),), "both")
        )
    else:
        flags = ("rope_scaling",) if rope_scaling is not None else ()
        nodes.append(OpNode("position", "position", "rope", flags, (("theta", int(rope_theta or 10000)),), "both"))

    # 每块
    for marker in [markers[g] for g in groups]:
        blk = rep[marker]
        pfx = f"blocks[{marker}]"
        attn_keys = [k for k in blk if k.startswith("attn.")]
        has_mla = any("kv_a_proj_with_mqa" in k or "kv_b_proj" in k or "q_a_proj" in k for k in attn_keys)
        if has_mla:
            variant = "mla"
        elif n_kv_heads < n_heads:
            variant = "gqa"
        else:
            variant = "mha"

        q_key = "attn.q_a_proj.weight" if "attn.q_a_proj.weight" in blk else "attn.q_proj.weight"
        if q_key not in blk:
            raise CanonicalError(f"{name}: {pfx} 没有 q 投影（{sorted(attn_keys)}）")
        if variant == "mla":
            q_width = n_heads * (int(qk_nope) + int(qk_rope))
            kv_width = int(kv_lora_rank) + int(qk_rope)
            o_rows = n_heads * int(v_head_dim)
        else:
            q_width = n_heads * head_dim
            kv_width = n_kv_heads * head_dim
            o_rows = n_heads * head_dim
        _check(f"{pfx} q 投影宽度", q_width, blk[q_key]["shape"][-1])
        if variant == "mla":
            _check(f"{pfx} kv_a_proj 宽度", kv_width, blk["attn.kv_a_proj_with_mqa.weight"]["shape"][-1])
        else:
            _check(f"{pfx} k 投影宽度", kv_width, blk["attn.k_proj.weight"]["shape"][-1])
            _check(f"{pfx} v 投影宽度", kv_width, blk["attn.v_proj.weight"]["shape"][-1])
        _check(f"{pfx} o 投影行数", o_rows, blk["attn.o_proj.weight"]["shape"][0])
        if variant == "mla":
            # MLA 把 KV 压到 `kv_lora_rank` 维之后**还要过一道 RMSNorm**
            # （`kv_a_layernorm`，宽度就是 `kv_lora_rank`）。这道范数很容易被漏掉：
            # 它不在任何「投影」列表里，而 DeepSeek-V2-Lite 的 checkpoint 恰好因此
            # 多出 27 × 512 = 13,824 个参数 —— 逐字节数出来的总数会差这么一点点，
            # 肉眼看不出来，等式看得出来。
            if "attn.kv_a_layernorm.weight" not in blk:
                raise CanonicalError(f"{name}: {pfx} 是 MLA 但没有 kv_a_layernorm")
            _check(f"{pfx} kv_a_layernorm 宽度", int(kv_lora_rank), blk["attn.kv_a_layernorm.weight"]["shape"][-1])

        aflags: list[str] = []
        has_attn_bias = any(k.endswith(".bias") for k in attn_keys)
        if has_attn_bias:
            aflags.append("attention_bias")
        # 逐个分量对质：q/k/v 一个布尔，o_proj 一个布尔。合成一个布尔会漏掉
        # 「只有 QKV 有偏置」这一族（Qwen1.5 / Qwen2 / Qwen2.5 就是），
        # 而漏掉它的代价是参数量少算 `n_layers × (n_heads + 2·n_kv_heads)·head_dim`。
        has_qkv_bias = any(
            k.endswith(".bias") for k in attn_keys
            if not k.startswith("attn.o_proj") and not k.startswith("attn.q_a_layernorm")
        )
        has_o_bias = "attn.o_proj.bias" in blk
        if qkv_bias is not None:
            _check(f"{pfx} qkv 偏置", bool(qkv_bias), has_qkv_bias)
        if o_bias is not None:
            _check(f"{pfx} o_proj 偏置", bool(o_bias), has_o_bias)
        if "attn.q_norm.weight" in blk or "attn.k_norm.weight" in blk:
            aflags.append("qk_norm")
        swa = sliding_window is not None and use_sliding_window is not False
        if swa:
            aflags.append("sliding_window")
        if rope_scaling is not None:
            aflags.append("rope_scaling")
        adims: list[tuple[str, int]] = [("n_heads", n_heads), ("n_kv_heads", n_kv_heads), ("head_dim", head_dim)]
        if variant == "mla":
            adims += [
                ("qk_nope_head_dim", int(qk_nope)),
                ("qk_rope_head_dim", int(qk_rope)),
                ("v_head_dim", int(v_head_dim)),
                ("kv_lora_rank", int(kv_lora_rank)),
            ]
        if swa:
            adims.append(("sliding_window", int(sliding_window)))
        if max_window_layers is not None and int(max_window_layers) < n_layers and swa:
            adims.append(("max_window_layers", int(max_window_layers)))
        nodes.append(OpNode(f"{pfx}.attn", "attention", variant, tuple(sorted(aflags)), tuple(sorted(adims)), "both"))

        for suffix in ("q_proj", "k_proj", "v_proj", "o_proj", "q_a_proj", "q_b_proj",
                       "kv_a_proj_with_mqa", "kv_b_proj", "q_norm", "k_norm",
                       "q_a_layernorm", "kv_a_layernorm"):
            for param in ("weight", "bias"):
                key = f"attn.{suffix}.{param}"
                if key in blk:
                    t = blk[key]
                    add_tensor(f"{pfx}.{key}", t["source_name"], t["shape"], None, "both")

        ffn_keys = [k for k in blk if k.startswith("ffn.")]
        is_moe = any("experts[" in k for k in ffn_keys)
        if is_moe:
            ffn_variant = "moe"
        elif "ffn.gate_proj.weight" in blk:
            ffn_variant = "swiglu"
        elif "ffn.fc_in.weight" in blk:
            ffn_variant = "mlp"
        else:
            raise CanonicalError(f"{name}: {pfx} 的 FFN 认不出来（{sorted(ffn_keys)}）")

        fflags: list[str] = []
        has_mlp_bias = any(k.endswith(".bias") for k in ffn_keys)
        if has_mlp_bias:
            fflags.append("mlp_bias")
        if mlp_bias is not None:
            _check(f"{pfx} FFN 偏置", bool(mlp_bias), has_mlp_bias)
        nodes.append(OpNode(f"{pfx}.ffn", "ffn", ffn_variant, tuple(sorted(fflags)), (("intermediate", intermediate),), "both"))
        for suffix in ("gate_proj", "up_proj", "down_proj", "fc_in", "fc_out"):
            for param in ("weight", "bias"):
                key = f"ffn.{suffix}.{param}"
                if key in blk:
                    t = blk[key]
                    add_tensor(f"{pfx}.{key}", t["source_name"], t["shape"], None, "both")

        if is_moe:
            if "ffn.router.weight" not in blk:
                raise CanonicalError(f"{name}: {pfx} 是 MoE 但没有 router")
            _check(f"{pfx} 路由专家数", n_experts, _count_experts(blk))
            _check(f"{pfx} router 输出宽度", n_experts, blk["ffn.router.weight"]["shape"][0])
            nodes.append(
                OpNode(f"{pfx}.ffn.router", "router", str(f.get("score_func") or "softmax"), (),
                       (("n_experts", n_experts), ("top_k", top_k)), "both")
            )
            add_tensor(f"{pfx}.ffn.router.weight", blk["ffn.router.weight"]["source_name"],
                       blk["ffn.router.weight"]["shape"], None, "both")
            exp_dims = (("intermediate", expert_width),) if expert_width else ()
            nodes.append(OpNode(f"{pfx}.ffn.experts[*]", "expert", "swiglu", (), exp_dims, "both"))
            # 专家是**一条模式**：路径里的 `experts[*]` 只出现一次，形状取自第 0 号，
            # 其余专家的形状逐条比对 —— 形状不一致就报错（「专家参差不齐」是真实存在的
            # 结构缺陷，而它不会自己报错）。
            for suffix in ("gate_proj", "up_proj", "down_proj"):
                first_key = f"ffn.experts[0].{suffix}.weight"
                if first_key not in blk:
                    raise CanonicalError(f"{name}: {pfx} 缺专家 0 的 {suffix}")
                t0 = blk[first_key]
                for e in range(1, n_experts):
                    key = f"ffn.experts[{e}].{suffix}.weight"
                    if key not in blk:
                        raise CanonicalError(f"{name}: {pfx} 缺专家 {e} 的 {suffix}")
                    if tuple(blk[key]["shape"]) != tuple(t0["shape"]):
                        raise CanonicalError(
                            f"{name}: {pfx} 专家 {e} 的 {suffix} 形状 {blk[key]['shape']} "
                            f"与专家 0 的 {t0['shape']} 不一致"
                        )
                add_tensor(f"{pfx}.ffn.experts[*].{suffix}.weight", t0["source_name"], t0["shape"], None, "both")
                if expert_width is not None:
                    # 专家宽度的**位置取决于矩阵方向**：`gate_proj`/`up_proj` 是
                    # `(hidden, width)`，`down_proj` 是 `(width, hidden)`。一律取 `[-1]`
                    # 会在 `down_proj` 上悄悄拿隐藏宽度去比专家宽度（DeepSeek-V2-Lite
                    # 的 1408 对 2048 就是这么撞出来的）。
                    got = t0["shape"][-1] if suffix != "down_proj" else t0["shape"][0]
                    _check(f"{pfx} 专家宽度 {suffix}", expert_width, got)
            sflags = ("shared_experts",) if any(k.startswith("ffn.shared_experts.") for k in blk) else ()
            if n_groups > 1:
                sflags = tuple(sorted(sflags + ("grouped_routing",)))
            if sflags:
                nodes.append(
                    OpNode(f"{pfx}.ffn.shared_experts", "shared_expert", "swiglu", sflags,
                           (("n_shared_experts", n_shared), ("n_groups", n_groups), ("topk_group", topk_group)), "both")
                )
                for suffix in ("gate_proj", "up_proj", "down_proj"):
                    key = f"ffn.shared_experts.{suffix}.weight"
                    if key in blk:
                        t = blk[key]
                        # 共享专家是**一个** MLP，宽度是 `n_shared_experts × 专家宽度`
                        # （DeepSeek-V2-Lite 的 2816 = 2 × 1408）。按单个专家宽度去比
                        # 会误报，而按 2816 去比能抓住「共享专家被写窄了」这种真缺陷。
                        if expert_width and n_shared:
                            want = n_shared * expert_width
                            got = t["shape"][-1] if suffix != "down_proj" else t["shape"][0]
                            _check(f"{pfx} 共享专家宽度 {suffix}", want, got)
                        add_tensor(f"{pfx}.ffn.shared_experts.{suffix}.weight", t["source_name"], t["shape"], None, "both")

        # 范数
        for norm_key, label in (("norm1", "pre"), ("norm2", "post")):
            key = f"{norm_key}.weight"
            if key not in blk:
                raise CanonicalError(f"{name}: {pfx} 缺 {key}")
            nodes.append(OpNode(f"{pfx}.{norm_key}", "norm", norm_kind, (), (("eps", _eps_key(norm_eps)),), "both"))
            t = blk[key]
            add_tensor(f"{pfx}.{norm_key}.weight", t["source_name"], t["shape"], None, "both")
            if f"{norm_key}.bias" in blk:
                t2 = blk[f"{norm_key}.bias"]
                add_tensor(f"{pfx}.{norm_key}.bias", t2["source_name"], t2["shape"], None, "both")

    # 首块例外与 first_k_dense 的一致性（DeepSeek 的 first_k_dense_replace=1）
    if first_k_dense:
        for i in range(first_k_dense):
            if any("experts[" in k for k in blocks[i]):
                raise CanonicalError(f"{name}: first_k_dense={first_k_dense} 说块 {i} 是稠密，但清单里有专家")

    fn = by_path.get("final_norm.weight")
    if fn is None:
        raise CanonicalError(f"{name}: 清单里没有 final_norm.weight")
    nodes.append(OpNode("final_norm", "norm", norm_kind, (), (("eps", _eps_key(norm_eps)),), "both"))
    add_tensor("final_norm.weight", fn["source_name"], fn["shape"], None, "both")
    if "final_norm.bias" in by_path:
        fb = by_path["final_norm.bias"]
        add_tensor("final_norm.bias", fb["source_name"], fb["shape"], None, "both")
    nodes.append(OpNode("head", "head", "causal_lm", (), (("vocab", vocab), ("hidden", hidden)), "both"))

    nodes.sort(key=lambda n: n.path)
    tensors.sort(key=lambda t: t.path)

    # ---- 计算图 ----
    edges = _build_edges(groups, markers, has_pos, nodes)

    topology: list[tuple[str, Any]] = [
        ("n_layers", n_layers),
        ("block_groups", tuple(tuple(g) for g in groups)),
        ("block_group_shape", shape),
        ("hidden", hidden),
        ("vocab", vocab),
        ("max_positions", max_positions),
        ("has_position_embedding", has_pos),
        ("tie_embeddings", tie_actual),
        ("first_k_dense", first_k_dense),
        ("n_experts", n_experts),
        ("top_k", top_k),
        ("n_shared_experts", n_shared),
        ("n_groups", n_groups),
        ("topk_group", topk_group),
        ("norm_kind", norm_kind),
    ]
    topology.sort(key=lambda kv: kv[0])

    identity = [
        ("source_model_type", str(f.get("source_model_type") or "")),
        ("architectures", tuple(f.get("architectures") or ())),
        ("activation", str(f.get("activation") or "")),
        ("release", ""),
    ]

    return CanonicalIR(
        name=name,
        topology=tuple(topology),
        nodes=tuple(nodes),
        tensors=tuple(tensors),
        edges=tuple(edges),
        identity=tuple(sorted(identity)),
        dialect=dialect,
        notes=tuple(notes),
        crosschecks=tuple(checks),
    )


# ---------------------------------------------------------------- 辅助


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _eps_key(eps: float) -> int:
    """eps 进指纹时转成整数「纳误差」刻度，避免 1e-05 与 1e-5 的浮点表示差异。"""
    return int(round(eps * 1e12))


def _count_experts(blk: Mapping[str, Any]) -> int:
    idxs = {int(m.group(1)) for k in blk if (m := re.search(r"ffn\.experts\[(\d+)\]", k))}
    return len(idxs)


def _first_index(groups: Sequence[tuple[int, int]], markers: Mapping[tuple[int, int], str], marker: str) -> int:
    for g in groups:
        if markers[g] == marker:
            return g[0]
    raise CanonicalError(f"标记 {marker!r} 不对应任何组")


def _build_edges(
    groups: Sequence[tuple[int, int]],
    markers: Mapping[tuple[int, int], str],
    has_pos: bool,
    nodes: Sequence[OpNode],
) -> list[Edge]:
    """计算 DAG。

    重复堆叠用**一条自环**表示（`blocks[*].ffn → blocks[*].norm1`），这是「模式」而不是
    「展开的实例」的直接后果：32 层不会变成 32 条边。最后一组的输出额外接 `final_norm`，
    于是「堆叠怎么收尾」也是图里可读的一环。
    """
    kinds = {(n.path, n.kind) for n in nodes}
    variant_of = {n.path: n.variant for n in nodes}
    order = [markers[g] for g in groups]
    out: list[Edge] = []
    first = order[0]
    out.append(Edge("embed", f"blocks[{first}].norm1", "hidden"))
    if has_pos:
        out.append(Edge("position", "embed", "add"))
    else:
        for m in order:
            out.append(Edge("position", f"blocks[{m}].attn", "rope"))
    for idx, m in enumerate(order):
        pfx = f"blocks[{m}]"
        out.append(Edge(f"{pfx}.norm1", f"{pfx}.attn", "normed"))
        out.append(Edge(f"{pfx}.attn", f"{pfx}.norm2", "residual"))
        is_moe = variant_of.get(f"{pfx}.ffn") == "moe"
        if is_moe:
            out.append(Edge(f"{pfx}.norm2", f"{pfx}.ffn.router", "normed"))
            out.append(Edge(f"{pfx}.ffn.router", f"{pfx}.ffn.experts[*]", "route"))
            out.append(Edge(f"{pfx}.ffn.experts[*]", f"{pfx}.ffn", "combine"))
            if (f"{pfx}.ffn.shared_experts", "shared_expert") in kinds:
                out.append(Edge(f"{pfx}.norm2", f"{pfx}.ffn.shared_experts", "shared"))
                out.append(Edge(f"{pfx}.ffn.shared_experts", f"{pfx}.ffn", "combine"))
        else:
            out.append(Edge(f"{pfx}.norm2", f"{pfx}.ffn", "normed"))
        nxt = order[idx + 1] if idx + 1 < len(order) else order[0]
        out.append(Edge(f"{pfx}.ffn", f"blocks[{nxt}].norm1", "next_block"))
        out.append(Edge(f"{pfx}.ffn", "final_norm", "exit"))
    out.append(Edge("final_norm", "head", "normed"))
    uniq = sorted({(e.src, e.dst, e.label) for e in out})
    return [Edge(*t) for t in uniq]
