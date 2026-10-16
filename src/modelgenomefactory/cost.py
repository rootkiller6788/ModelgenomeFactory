"""结构代价：参数量 / FLOPs / 显存。

代价是**主地址空间的函数**，不是独立事实 —— 它不进 diff 的地址空间，但必须能被
重新算出来（`tests/test_genome.py` 有一条测试钉住「apply 之后 cost 也对得上」）。

## 参数量走三路，必须互相印证

| 路径 | 数据来源 | 强度 |
| --- | --- | --- |
| ① 张量表求和 | 清单里的每个名字与形状 | **逐位精确** |
| ② 解析公式 | config 的标量 | **逐位精确** |
| ③ checkpoint 元数据 | HF 的 `safetensors.total` 减去非参数缓冲 | **逐位精确** |

① 与 ② 必须**逐位相等**。这两条是两份独立实现：① 从张量清单里数，② 从 config 的
标量里算。任何一处漂移都会让 `explain` 报出一个「没人会因此报错」的错数
（`L2-architecturefactory` 的坑 53 就是这么来的）。

③ 是**外部真值**：`safetensors.total` 是从 checkpoint 文件里逐字节数出来的，不是
谁的宣传数字。它比四舍五入的公开值强得多，但也带一个陷阱 —— 它把**非参数缓冲**
也算进去（GPT-2 的 checkpoint 里有 12 个 `1×1×1024×1024` 的因果掩码，共
12,582,912 个元素；Llama-1/2 里有 `rotary_emb.inv_freq`）。所以 zoo 里把缓冲量
单列，等式写成 `params + buffers == total`。这条等式抓到的东西是路径 ① ② 抓不到的：
**两张表可能一起错在同一个地方**（比如都漏掉 MLA 的 `kv_a_layernorm`），
只有外部真值能证伪。

FLOPs 的口径必须**可逐位复现**，所以它是「矩阵乘 2·in·out 之和 + 注意力项」，
不是实测延迟。这一点在 README 附录里写清楚 —— 用实测延迟当成本轴，换台机器就不可比。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

from .dialects import position_kind_of
from .schema import CanonicalIR, Cost, Genome, RawStructure

__all__ = [
    "params_from_schema",
    "params_from_formula",
    "flops_per_token",
    "kv_cache_bytes",
    "derive",
    "derive_from_flat",
    "compute",
    "active_params",
    "marker_count",
    "expand_counts",
]

_MARKER_RE = re.compile(r"blocks\[([^\]]+)\]")

#: `_role_bucket` 的前缀剥离与判别式。`_role_bucket` 在 `params_from_schema` 里
#: 每个张量都要走一遍，内联写法会让这两条正则被反复编译。
_BLOCKS_PREFIX = re.compile(r"^blocks\[[^\]]+\]\.")
_NORM_PREFIX = re.compile(r"^norm[12]\.")


def marker_count(marker: str, block_groups: Sequence[Sequence[int]]) -> int:
    """路径标记 → 实例数。`*` 指成员最多的那一组。"""
    if marker == "*":
        return max(g[1] for g in block_groups)
    if ":" in marker:
        a, b = marker.split(":")
        return int(b) - int(a)
    return 1


def expand_counts(ir: CanonicalIR) -> list[tuple[str, int, int]]:
    """把模式张量展开成 `(路径, 实例数, 单实例参数量)`。

    参数量的唯一权威口径就在这里：**实例数来自 `block_groups` 与 `n_experts`，
    不是数出来的**。权重绑定的那一份只算一次。
    """
    groups = [tuple(g) for g in ir.topo("block_groups", ((0, int(ir.topo("n_layers", 1))),))]
    n_experts = int(ir.topo("n_experts", 0) or 0)
    out: list[tuple[str, int, int]] = []
    for t in ir.tensors:
        if t.tied_with is not None:
            continue                       # 与别人共享，不重复计
        n = 1
        m = _MARKER_RE.search(t.path)
        if m:
            n *= marker_count(m.group(1), groups)
        if "experts[*]" in t.path:
            if n_experts == 0:
                raise ValueError(f"{t.path}: 路径带 experts[*] 但 n_experts=0")
            n *= n_experts
        out.append((t.path, n, t.numel))
    return out


def params_from_schema(ir: CanonicalIR | Genome) -> tuple[int, tuple[tuple[str, int], ...]]:
    """路径 ①：从张量表求和。返回 `(总参数量, 分项)`。

    签名收 `CanonicalIR | Genome`：两者都有 `.tensors` / `.nodes` / `.topo()`，
    所以同一份实现既能给 IR 算，也能给「diff apply 之后重建的 Genome」算 ——
    代价必须能复算，这是它「不是独立事实」的证据。
    """
    per_role: dict[str, int] = {}
    total = 0
    for path, n, numel in expand_counts(ir):
        sub = n * numel
        total += sub
        per_role[_role_bucket(path)] = per_role.get(_role_bucket(path), 0) + sub
    breakdown = tuple(sorted(per_role.items()))
    return total, breakdown


def _role_bucket(path: str) -> str:
    p = _BLOCKS_PREFIX.sub("", path)
    if p in ("embed.weight", "head.weight"):
        return "embed/head"
    if p.startswith("pos_embed"):
        return "position"
    if p.startswith("final_norm") or _NORM_PREFIX.match(p):
        return "norm"
    if p.startswith("attn."):
        return "attn"
    if p.startswith("ffn.experts"):
        return "ffn.experts"
    if p.startswith("ffn.shared_experts"):
        return "ffn.shared_experts"
    if p.startswith("ffn.router"):
        return "ffn.router"
    return "ffn"


def active_params(obj: CanonicalIR | Genome) -> int:
    """每 token 实际激活的参数量。

    MoE 的关键列：Mixtral-8x7B 总参数 46.7B、激活参数 12.9B，**差 3.6 倍**。
    只报总参数会让「MoE 更省」这件事从矩阵里消失，所以这一列必须单列。

    口径：`experts[*]` 那部分按 `top_k` 而不是 `n_experts` 计；稠密模型上
    它必然等于 `params`（有测试钉住这一点）。
    """
    groups = [tuple(g) for g in obj.topo("block_groups", ((0, int(obj.topo("n_layers", 1))),))]
    n_experts = int(obj.topo("n_experts", 0) or 0)
    top_k = 0
    for n in obj.nodes:
        if n.kind == "router":
            top_k = int(n.dim("top_k", 0) or 0)
            break
    if n_experts and top_k == 0:
        raise ValueError(f"{obj.name}: 是 MoE（{n_experts} 专家）但没有 top_k")
    total = 0
    for t in obj.tensors:
        if t.tied_with is not None:
            continue
        n = 1
        m = _MARKER_RE.search(t.path)
        if m:
            n *= marker_count(m.group(1), groups)
        if "experts[*]" in t.path:
            n *= top_k
        total += n * t.numel
    return total


def params_from_formula(raw: RawStructure) -> int:
    """路径 ②：从 config 标量按解析公式算。与路径 ① 必须**逐位相等**。

    这份实现**不看张量表、不看清单**：只读 `hidden / n_layers / n_heads / ...` 这些
    标量，再按每个家族自己的约定把同一批张量重算一遍。所以它与 `params_from_schema`
    是两份独立实现 —— 而这份独立性真的抓到了东西：第一版公式把位置嵌入当成每个模型
    都有、把 FFN 一律当成 3 矩阵、把绑定词表当成不绑定，于是

    - GPT-2 多算 67M（未绑定的 38.6M 头 + 4 矩阵的 FFN 多算 28.3M），
    - Llama-2-7B 多算 17M（RoPE 根本没有位置表，却被加上 `max_positions × hidden`），
    - Mistral-7B 多算 134M（32768 的位置表）。

    三处偏差都被这条等式当场抓出来。**公式里的约定写错了，等式就会响** ——
    这就是为什么路径 ① 与路径 ② 必须都留着。

    MoE 的 router 是**每层**一个：第一版把它写成全模型一个，Mixtral 少算了 31 个。
    """
    f = raw.as_map()
    if not f.get("position_kind"):
        f["position_kind"] = position_kind_of(f)

    def g(key: str, default: Any = None) -> Any:
        return f.get(key, default)

    H = int(g("hidden"))
    V = int(g("vocab"))
    L = int(g("n_layers"))
    nh = int(g("n_heads"))
    nkv = int(g("n_kv_heads"))
    hd = int(g("head_dim"))
    I = int(g("intermediate"))
    P = int(g("max_positions"))
    E = int(g("n_experts") or 0)
    S = int(g("n_shared_experts") or 0)
    K = int(g("first_k_dense") or 0)
    norm_kind = str(g("norm_kind"))
    act = str(g("activation") or "")
    mla = g("kv_lora_rank") is not None

    # 家族约定（与 `extract.expected_tensors` 用同一套；两边任何一边写错，等式就响）
    Im = g("moe_intermediate")
    if E and Im is None:
        Im = I                      # Mixtral 系：intermediate_size 就是专家宽度
    Im = int(Im) if Im else 0
    learned_pos = str(g("position_kind") or "") == "learned"
    tie = g("tie_embeddings")
    if tie is None:
        tie = norm_kind == "layernorm"
    fb = norm_kind == "layernorm"
    qkv_bias = g("qkv_bias")
    o_bias = g("o_bias")
    mlp_bias = g("mlp_bias")
    qkv_bias = fb if qkv_bias is None else bool(qkv_bias)
    o_bias = fb if o_bias is None else bool(o_bias)
    mlp_bias = fb if mlp_bias is None else bool(mlp_bias)
    gated = act in ("silu", "gelu", "gelu_new") and norm_kind == "rmsnorm"
    ln = norm_kind == "layernorm"

    n_dense = K if E else L
    n_moe = (L - K) if E else 0

    # ---- 注意力（每层）----
    if mla:
        q_lora = g("q_lora_rank")
        qk_nope, qk_rope = int(g("qk_nope_head_dim")), int(g("qk_rope_head_dim"))
        v_head, kv_lora = int(g("v_head_dim")), int(g("kv_lora_rank"))
        q_width = nh * (qk_nope + qk_rope)
        kv_b_rows = nh * (qk_nope + v_head)
        attn = (H * int(q_lora) + int(q_lora) * q_width) if q_lora else H * q_width
        attn += H * (kv_lora + qk_rope)
        attn += kv_lora * kv_b_rows
        attn += (nh * v_head) * H
        # 压缩潜向量上的 RMSNorm：`q_a_layernorm` 宽度 = q_lora_rank，
        # `kv_a_layernorm` 宽度 = kv_lora_rank。漏掉后者，DeepSeek-V2-Lite 会少
        # 27 × 512 = 13,824 个参数 —— 肉眼看不出来，等式看得出来。
        attn += kv_lora
        if q_lora:
            attn += int(q_lora)
        if qkv_bias:
            attn += q_width + kv_b_rows
        if o_bias:
            attn += nh * v_head
    else:
        q_width, kv_width, o_rows = nh * hd, nkv * hd, nh * hd
        attn = H * q_width + H * kv_width * 2 + o_rows * H
        if qkv_bias:
            attn += q_width + kv_width * 2
        if o_bias:
            attn += o_rows

    # ---- 每块的范数 ----
    per_block_norm = 2 * H + (2 * H if ln else 0)

    # ---- 稠密 FFN（每个稠密层）----
    if gated:
        dense_ffn = 3 * H * I + ((2 * I + H) if mlp_bias else 0)
    else:
        dense_ffn = 2 * H * I + ((I + H) if mlp_bias else 0)

    # ---- MoE（每个 MoE 层）----
    moe_ffn = 0
    if n_moe:
        moe_ffn += E * H            # router：每层一个，不是全模型一个
        moe_ffn += E * 3 * H * Im   # 专家
        moe_ffn += S * 3 * H * Im   # 共享专家

    # ---- 顶层 ----
    top = V * H                                   # 词嵌入
    top += P * H if learned_pos else 0            # 可学习位置表；RoPE 没有
    top += 0 if tie else V * H                    # 绑定的话输出头不额外占参数
    top += H + (H if ln else 0)                   # final_norm

    return (attn + per_block_norm) * L + dense_ffn * n_dense + moe_ffn * n_moe + top


def flops_per_token(ir: CanonicalIR, seq_len: int) -> int:
    """前向 FLOPs / token。口径：矩阵乘 `2·in·out` 之和 + 注意力项 `4·seq·n_heads·head_dim`/层。

    注意力项是 `QK^T`（`2·seq·d`）与 `AV`（`2·seq·d`）两遍，所以系数是 4。
    归一化 / 激活 / 路由的逐元素代价量级上小两个数量级，**不计入**并在此写明 ——
    口径必须可逐位复现，含糊比漏算更糟。
    """
    total = 0
    for path, n, numel in expand_counts(ir):
        total += 2 * n * numel
    n_layers = int(ir.topo("n_layers", 0))
    variant = _attn_variant(ir)
    n_heads = int(_attn_dim(ir, "n_heads", 0))
    head_dim = int(_attn_dim(ir, "head_dim", 0))
    if variant == "mla":
        # MLA 的分数计算仍按 qk_nope + qk_rope 的等效宽度
        head_dim = int(_attn_dim(ir, "qk_nope_head_dim", head_dim)) + int(
            _attn_dim(ir, "qk_rope_head_dim", 0)
        )
    return total + n_layers * 4 * seq_len * n_heads * head_dim


def kv_cache_bytes(ir: CanonicalIR, seq_len: int, batch: int, dtype_bytes: int) -> int:
    """KV cache 显存。**这是 GQA / MLA 那条突变的直接后果，所以必须单列一列。**

    - MHA / GQA：`2 · n_layers · n_kv_heads · head_dim · seq · batch · dtype`
    - MLA：只缓存压缩潜变量 `kv_lora_rank + qk_rope_head_dim` 每 token，**不缓存 K/V**
    """
    n_layers = int(ir.topo("n_layers", 0))
    variant = _attn_variant(ir)
    if variant == "mla":
        per_token = int(_attn_dim(ir, "kv_lora_rank", 0)) + int(_attn_dim(ir, "qk_rope_head_dim", 0))
        return per_token * n_layers * seq_len * batch * dtype_bytes
    n_kv = int(_attn_dim(ir, "n_kv_heads", 0))
    head_dim = int(_attn_dim(ir, "head_dim", 0))
    return 2 * n_layers * n_kv * head_dim * seq_len * batch * dtype_bytes


def _attn_variant(ir: CanonicalIR) -> str:
    for n in ir.nodes:
        if n.kind == "attention":
            return n.variant
    raise ValueError(f"{ir.name}: 没有注意力节点")


def _attn_dim(ir: CanonicalIR, key: str, default: int) -> int:
    for n in ir.nodes:
        if n.kind == "attention":
            v = n.dim(key)
            return default if v is None else v
    return default


def derive(
    obj: CanonicalIR | Genome,
    *,
    seq_len: int,
    batch: int,
    dtype_bytes: int,
) -> dict[str, Any]:
    """从**主地址空间**能推出来的那部分代价。

    `CanonicalIR` 与 `Genome` 都满足这个签名（都有 `.tensors` / `.nodes` / `.topo()`），
    所以 `diff.apply` 重建之后可以逐位复算代价 —— 代价不是独立事实，这一条必须成立。
    """
    params, breakdown = params_from_schema(obj)
    return {
        "params": params,
        "flops_per_token": flops_per_token(obj, seq_len),
        "kv_cache_bytes": kv_cache_bytes(obj, seq_len, batch, dtype_bytes),
        "weight_bytes": params * dtype_bytes,
        "breakdown": breakdown,
    }


def compute(
    ir: CanonicalIR,
    raw: RawStructure,
    *,
    seq_len: int,
    batch: int,
    dtype_bytes: int,
    params_published_b: float | None = None,
) -> Cost:
    """完整代价：派生量走 `derive`，解析公式与公开值另算。"""
    d = derive(ir, seq_len=seq_len, batch=batch, dtype_bytes=dtype_bytes)
    return Cost(
        params=d["params"],
        params_formula=params_from_formula(raw),
        flops_per_token=d["flops_per_token"],
        kv_cache_bytes=d["kv_cache_bytes"],
        weight_bytes=d["weight_bytes"],
        breakdown=d["breakdown"],
        params_published_b=params_published_b,
        seq_len=seq_len,
        batch=batch,
        dtype_bytes=dtype_bytes,
    )


class _Parts:
    """只带 `.tensors` / `.nodes` / `.topo()` 的轻量壳。

    `genome.rebuild` 从主地址空间拼出零件后，用同一套 `derive` 复算代价 ——
    让「代价不是独立事实」这句话在代码层面成立，而不是靠文档声明。
    """

    __slots__ = ("topology", "nodes", "tensors")

    def __init__(self, topology, nodes, tensors):
        self.topology = topology
        self.nodes = nodes
        self.tensors = tensors

    def topo(self, key: str, default: Any = None) -> Any:
        for k, v in self.topology:
            if k == key:
                return v
        return default


def derive_from_flat(
    *,
    topology: Sequence[tuple[str, Any]],
    nodes: Sequence[Any],
    tensors: Sequence[Any],
    seq_len: int,
    batch: int,
    dtype_bytes: int,
) -> dict[str, Any]:
    return derive(
        _Parts(topology, nodes, tensors),
        seq_len=seq_len,
        batch=batch,
        dtype_bytes=dtype_bytes,
    )
