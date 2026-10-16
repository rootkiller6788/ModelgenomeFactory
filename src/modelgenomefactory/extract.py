"""抽取层：读 zoo 索引、读 config、把 config 归一化成事实。

这一层**完全不看参数名**。它只做两件事：把 config 的键名归一化（`n_layer` 与
`num_hidden_layers` 是同一个东西），以及把缺失键的**默认值**显式补上
（`num_key_value_heads` 缺席意味着「等于 num_attention_heads」，也就是 MHA ——
这一条猜错就会把 Llama-1 认成 GQA，而且不会报错）。

「读不懂的键」在这里被**收集**而不是被忽略：`unknown_keys` 会进报告，
`tests/test_extract.py` 断言它为空。这是「读不懂但假装读懂了」的唯一防线。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .dialects import ManifestError, parse_config, position_kind_of
from .schema import RawStructure

__all__ = ["ZooEntry", "ZooError", "load_zoo", "load_config_file", "extract_model", "coverage_report",
           "expected_tensors", "block_groups_for", "position_kind_of"]


class ZooError(ValueError):
    pass


@dataclass(frozen=True)
class ZooEntry:
    """zoo 里的一条标本。"""

    name: str
    family: str
    chain: int
    release: str
    params_b: float | None
    source_url: str
    note: str
    config_path: Path
    dialects: tuple[tuple[str, str], ...]      # (方言, 布局)
    #: 按官方 checkpoint 声明的家族约定，**只**用于填 config 沉默的键。
    #: 存成元组而不是字典，是为了让 `ZooEntry` 保持可哈希。
    conventions: tuple[tuple[str, Any], ...] = ()
    #: 外部真值：checkpoint 的 `safetensors.total`、非参数缓冲、以及两者之差。
    checkpoint: tuple[tuple[str, Any], ...] = ()

    def layout(self, dialect: str) -> str:
        for d, lay in self.dialects:
            if d == dialect:
                return lay
        raise ZooError(f"{self.name} 没有声明方言 {dialect!r}；已声明：{[d for d, _ in self.dialects]}")

    def has_dialect(self, dialect: str) -> bool:
        return any(d == dialect for d, _ in self.dialects)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "chain": self.chain,
            "release": self.release,
            "params_b": self.params_b,
            "source_url": self.source_url,
            "note": self.note,
            "dialects": {d: lay for d, lay in self.dialects},
            "conventions": dict(self.conventions),
            "checkpoint": dict(self.checkpoint),
        }


def load_zoo(root: Path | str) -> tuple[ZooEntry, ...]:
    """读 `data/zoo/zoo.json`。

    家族链的顺序由 `chain` 给出，**不按名字排序** —— 名字排序会把
    `llama-2-13b` 排在 `llama-2-7b` 之后（碰巧对）却把 `llama-3-8b` 排在
    `llama-2-13b` 之前（错）。链的顺序是数据，不是字符串序。
    """
    root = Path(root)
    index_path = root / "data" / "zoo" / "zoo.json"
    if not index_path.exists():
        raise ZooError(f"找不到 zoo 索引 {index_path}")
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    dialect_map = raw.get("dialects", {})
    out: list[ZooEntry] = []
    for item in raw["models"]:
        name = item["name"]
        dialects = dialect_map.get(name)
        if not dialects:
            raise ZooError(f"{name} 没有声明任何命名方言")
        cfg = root / "data" / "zoo" / name / "config.json"
        if not cfg.exists():
            raise ZooError(f"{name} 缺 config：{cfg}")
        out.append(
            ZooEntry(
                name=name,
                family=item["family"],
                chain=int(item["chain"]),
                release=item.get("release", ""),
                params_b=item.get("params_b"),
                source_url=item.get("source_url", ""),
                note=item.get("note", ""),
                config_path=cfg,
                dialects=tuple(sorted(dialects.items())),
                conventions=tuple(sorted((item.get("conventions") or {}).items())),
                checkpoint=tuple(sorted((item.get("checkpoint") or {}).items())),
            )
        )
    _validate_chain(out)
    return tuple(out)


def _validate_chain(entries: Iterable[ZooEntry]) -> None:
    """每个家族的 chain 必须是 1..n 的连续整数，且不能重复。

    这一条看着像洁癖，其实是在防「链上少了中间那一代」—— 少了的话相邻边界
    就跨代了，突变点会被算到错误的边界上，而报告里看不出任何异常。
    """
    by_family: dict[str, list[ZooEntry]] = {}
    for e in entries:
        by_family.setdefault(e.family, []).append(e)
    for family, members in by_family.items():
        chains = sorted(m.chain for m in members)
        if chains != list(range(1, len(members) + 1)):
            raise ZooError(f"家族 {family} 的 chain 不是 1..{len(members)} 的连续整数：{chains}")


def load_config_file(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def extract_model(entry: ZooEntry) -> RawStructure:
    """zoo 条目 → RawStructure（归一化后的 config 视图）。

    名字是 `extract_model` 而不是 `extract`：`extract` 是这个模块的名字，
    让函数与模块同名会在 `from modelgenomefactory import extract` 时**静默地**把模块遮掉
    （拿到的是函数），而 `import modelgenomefactory.extract` 又能正常用 —— 这种不一致
    排查起来很费时间。
    """
    raw = load_config_file(entry.config_path)
    try:
        facts, unknown, ignored = parse_config(raw, dict(entry.conventions))
    except (KeyError, ValueError) as exc:
        raise ManifestError(f"{entry.name}: config 归一化失败：{exc}") from exc
    return RawStructure(
        name=entry.name,
        facts=tuple(_to_facts(facts)),
        unknown_keys=tuple(unknown),
        ignored_keys=tuple(ignored),
    )


def _to_facts(rows: Iterable[tuple[str, Any, str, str]]):
    from .schema import ConfigFact

    return [ConfigFact(key=k, value=v, source_key=s, rule=r) for k, v, s, r in rows]


def coverage_report(structures: Iterable[RawStructure]) -> dict[str, Any]:
    """config 键的覆盖率报告。**「读不懂」与「决定不看」分两栏。**

    合并成一栏是很容易犯的错：把 dropout 和某个真正陌生的键一起报成
    「未识别」，报告就会显示「覆盖率 92%」；反过来把它们都算成已处理，
    又会把「读不懂但假装读懂了」这个最危险的失败模式藏起来。
    进产物、进报告 —— `unknown` 为空才是健康状态。
    """
    unknown: dict[str, list[str]] = {}
    ignored: dict[str, dict[str, str]] = {}
    reasons: dict[str, int] = {}
    n = 0
    for s in structures:
        n += 1
        if s.unknown_keys:
            unknown[s.name] = list(s.unknown_keys)
        if s.ignored_keys:
            ignored[s.name] = {k: r for k, r in s.ignored_keys}
            for _k, r in s.ignored_keys:
                reasons[r] = reasons.get(r, 0) + 1
    return {
        "n_models": n,
        "unknown": unknown,
        "clean": not unknown,
        "ignored": ignored,
        "ignored_reasons": dict(sorted(reasons.items())),
        "n_ignored": sum(len(v) for v in ignored.values()),
    }


# ---------------------------------------------------------------- 清单的作者期渲染


def _facts_of(raw: RawStructure) -> dict[str, Any]:
    """事实表 → 字典。缺 `position_kind` 的老结构按家族指纹补一次。"""
    f = raw.as_map()
    if not f.get("position_kind"):
        f["position_kind"] = position_kind_of(f)
    return f


def expected_tensors(raw: RawStructure, *, position_kind: str | None = None) -> list[dict[str, Any]]:
    """从 config 标量推出**规范张量表**（模式路径 + 形状 + 绑定）。

    只在 `materialize` 里用：把「按 config 算出来的结构」渲染成某种命名方言的清单。
    流水线（`run`）读的是落盘清单，**不调用这个函数** —— 于是「清单侧」与
    「config 侧」在 `canonical.py` 的 `crosschecks` 里互校时是两条独立的路。
    """
    f = _facts_of(raw)
    H = int(f["hidden"])
    V = int(f["vocab"])
    L = int(f["n_layers"])
    nh = int(f["n_heads"])
    nkv = int(f["n_kv_heads"])
    hd = int(f["head_dim"])
    I = int(f["intermediate"])
    P = int(f["max_positions"])
    E = int(f.get("n_experts") or 0)
    top_k = int(f.get("top_k") or 0)
    S = int(f.get("n_shared_experts") or 0)
    Im = f.get("moe_intermediate")
    if E and Im is None:
        # Mixtral 系没有 `moe_intermediate_size` 这个键：它的 `intermediate_size`
        # **就是**专家宽度（每一层都是 MoE，不存在独立的稠密 FFN 宽度）。
        # DeepSeek 系两个键都有：`intermediate_size` 给前 K 个稠密块，
        # `moe_intermediate_size` 给专家。所以这里只在「有专家且没显式给专家宽度」时回退。
        Im = I
    K = int(f.get("first_k_dense") or 0)
    norm_kind = str(f["norm_kind"])
    act = str(f.get("activation") or "")
    tie = f.get("tie_embeddings")
    # 偏置是**三个**独立事实，不是一个布尔。`attention_bias`（若 config 声明了）
    # 已经在 `parse_config` 里被拆成 `qkv_bias` / `o_bias`；没声明的就落到这里，
    # 先看 zoo 按 checkpoint 声明的家族约定，再退到 `norm_kind` 兜底
    # （GPT-2 系 = LayerNorm + 全偏置 + 权重绑定，Llama 系 = RMSNorm + 无偏置 + 不绑定）。
    # 这套补值只是**渲染清单时的启发式**：解析器不看它，它从清单里的张量存在性与
    # tied 声明判定，那条路才权威。
    if tie is None:
        tie = norm_kind == "layernorm"
    fallback_bias = norm_kind == "layernorm"
    qkv_bias = f.get("qkv_bias")
    o_bias = f.get("o_bias")
    mlp_bias = f.get("mlp_bias")
    qkv_bias = fallback_bias if qkv_bias is None else bool(qkv_bias)
    o_bias = fallback_bias if o_bias is None else bool(o_bias)
    mlp_bias = fallback_bias if mlp_bias is None else bool(mlp_bias)
    kv_lora = f.get("kv_lora_rank")
    q_lora = f.get("q_lora_rank")
    qk_nope = f.get("qk_nope_head_dim")
    qk_rope = f.get("qk_rope_head_dim")
    v_head = f.get("v_head_dim")
    if position_kind is None:
        position_kind = str(f.get("position_kind") or position_kind_of(f))

    if E and top_k == 0:
        raise ManifestError(f"{raw.name}: 有 {E} 个专家但没有 top_k")
    if E and not Im:
        raise ManifestError(f"{raw.name}: MoE 缺 moe_intermediate_size")
    if E and K >= L:
        raise ManifestError(f"{raw.name}: first_k_dense={K} 不小于层数 {L}")

    mla = kv_lora is not None
    if mla:
        q_width = nh * (int(qk_nope) + int(qk_rope))
        kv_width = int(kv_lora) + int(qk_rope)
        o_rows = nh * int(v_head)
    else:
        q_width = nh * hd
        kv_width = nkv * hd
        o_rows = nh * hd

    # 块标记：前 K 块单独列，其余合成尾组。
    markers: list[str] = [str(i) for i in range(K)] + ["*"]

    out: list[dict[str, Any]] = []

    def add(path: str, shape: Sequence[int], tied: str | None = None) -> None:
        out.append({"path": path, "shape": tuple(shape), "tied_with": tied})

    # 顶层
    add("embed.weight", (V, H))
    if position_kind == "learned":
        add("pos_embed.weight", (P, H))
    add("head.weight", (V, H), tied="embed.weight" if tie else None)
    add("final_norm.weight", (H,))
    if norm_kind == "layernorm":
        add("final_norm.bias", (H,))

    for marker in markers:
        p = f"blocks[{marker}]"
        add(f"{p}.norm1.weight", (H,))
        add(f"{p}.norm2.weight", (H,))
        if norm_kind == "layernorm":
            add(f"{p}.norm1.bias", (H,))
            add(f"{p}.norm2.bias", (H,))

        if mla:
            if q_lora:
                add(f"{p}.attn.q_a_proj.weight", (H, int(q_lora)))
                add(f"{p}.attn.q_b_proj.weight", (int(q_lora), q_width))
                add(f"{p}.attn.q_a_layernorm.weight", (int(q_lora),))
            else:
                add(f"{p}.attn.q_proj.weight", (H, q_width))
            add(f"{p}.attn.kv_a_proj_with_mqa.weight", (H, kv_width))
            # 压缩后的 KV 潜向量还要过一道 RMSNorm，宽度就是 kv_lora_rank。
            # 漏掉它 DeepSeek-V2-Lite 会少 27 × 512 = 13,824 个参数。
            add(f"{p}.attn.kv_a_layernorm.weight", (int(kv_lora),))
            add(f"{p}.attn.kv_b_proj.weight", (int(kv_lora), nh * (int(qk_nope) + int(v_head))))
            add(f"{p}.attn.o_proj.weight", (o_rows, H))
        else:
            add(f"{p}.attn.q_proj.weight", (H, q_width))
            add(f"{p}.attn.k_proj.weight", (H, kv_width))
            add(f"{p}.attn.v_proj.weight", (H, kv_width))
            add(f"{p}.attn.o_proj.weight", (o_rows, H))
        if qkv_bias or o_bias:
            if mla and q_lora:
                if qkv_bias:
                    add(f"{p}.attn.q_b_proj.bias", (q_width,))
            elif qkv_bias:
                add(f"{p}.attn.q_proj.bias", (q_width,))
            if qkv_bias:
                if mla:
                    add(f"{p}.attn.kv_b_proj.bias", (nh * (int(qk_nope) + int(v_head)),))
                else:
                    add(f"{p}.attn.k_proj.bias", (kv_width,))
                    add(f"{p}.attn.v_proj.bias", (kv_width,))
            if o_bias:
                add(f"{p}.attn.o_proj.bias", (o_rows,))

        is_dense_layer = marker != "*" or not E
        if is_dense_layer:
            if act in ("silu", "gelu", "gelu_new") and norm_kind == "rmsnorm":
                add(f"{p}.ffn.gate_proj.weight", (H, I))
                add(f"{p}.ffn.up_proj.weight", (H, I))
                add(f"{p}.ffn.down_proj.weight", (I, H))
                if mlp_bias:
                    add(f"{p}.ffn.gate_proj.bias", (I,))
                    add(f"{p}.ffn.up_proj.bias", (I,))
                    add(f"{p}.ffn.down_proj.bias", (H,))
            else:
                add(f"{p}.ffn.fc_in.weight", (I, H))
                add(f"{p}.ffn.fc_out.weight", (H, I))
                if mlp_bias:
                    add(f"{p}.ffn.fc_in.bias", (I,))
                    add(f"{p}.ffn.fc_out.bias", (H,))
        else:
            add(f"{p}.ffn.router.weight", (E, H))
            # 专家写成**一条模式**（`experts[*]`），与 `canonical.py` 从清单折出来的形状
            # 一致。形状只取第 0 号；「每个专家都一样大」这条断言由 `canonicalize` 在
            # 清单侧逐号核对 —— 配置侧根本看不到第 7 号专家是不是被谁改小了。
            add(f"{p}.ffn.experts[*].gate_proj.weight", (H, int(Im)))
            add(f"{p}.ffn.experts[*].up_proj.weight", (H, int(Im)))
            add(f"{p}.ffn.experts[*].down_proj.weight", (int(Im), H))
            if S:
                # 共享专家在 checkpoint 里是**一个** MLP，宽度是 `n_shared_experts ×
                # moe_intermediate_size`（DeepSeek-V2 的 `DeepseekV2MLP(hidden,
                # intermediate * n_shared_experts)`）。按单个专家的宽度算会少一半：
                # DeepSeek-V2-Lite 的共享专家就是 2816 = 2 × 1408，不是 1408。
                Iw = S * int(Im)
                add(f"{p}.ffn.shared_experts.gate_proj.weight", (H, Iw))
                add(f"{p}.ffn.shared_experts.up_proj.weight", (H, Iw))
                add(f"{p}.ffn.shared_experts.down_proj.weight", (Iw, H))

    out.sort(key=lambda t: t["path"])
    return out


def block_groups_for(raw: RawStructure) -> list[tuple[int, int]]:
    """清单渲染用的块组（与 `canonical.py` 从清单量出来的那份应当一致）。"""
    L = int(raw.require("n_layers"))
    K = int(raw.get("first_k_dense") or 0)
    if K <= 0:
        return [(0, L)]
    if K >= L:
        raise ManifestError(f"{raw.name}: first_k_dense={K} 不小于层数 {L}")
    return [(0, K), (K, L - K)]
