"""Canonicalizer：把不同命名体系归一到同一套规范路径。

这是整个项目的**真正难点**，也是最容易被写浅的地方。README 说「不同框架（HF / vLLM /
自研）的命名与层级先归一到统一 IR」，落到代码上就是两张表：

1. `CONFIG_ALIASES` —— config 键名归一。`n_layer` / `num_hidden_layers` 是同一个东西；
   `layer_norm_epsilon` 与 `rms_norm_eps` 还顺带告诉你**范数类型**。
2. `NAME_RULES` —— 参数名归一。`model.layers.3.self_attn.q_proj.weight` /
   `transformer.h.3.attn.c_attn.weight` / `layers.3.attn.Wq` 必须落到同一条规范路径。

四个真问题（都不是模式匹配能糊过去的）：

- **融合与拆分**：vLLM 把 q/k/v 合成一个 `qkv_proj`，把 gate/up 合成 `gate_up_proj`；
  Mixtral 在 vLLM 里是 `w13_weight` / `w2_weight`；GPT-2 官方 checkpoint 的 `c_attn`
  就是融合的。归一化必须**按头结构真的切开**，切不开就报错，绝不猜。
- **转置**：GPT-2 的线性层是 `Conv1D`，权重形状是 `(in, out)`；Llama 的 `Linear` 是
  `(out, in)`。不处理这一条，`c_fc.weight` 的 `(768, 3072)` 会被读成
  「out=768, in=3072」—— 也就是被误认成 `down_proj`，而**没有任何地方会报错**。
- **同名不同物**：Llama 的 `input_layernorm` 其实是 RMSNorm，名字里带 "layernorm"。
  所以**范数类型只能来自 config**，张量清单给不出。
- **信息缺口**：张量清单给不出激活函数、RoPE base、滑窗、top-k。这些只在 config 里。

于是每个事实都要带来源（`manifest` / `config` / `both`），重叠的部分必须互校 ——
见 `canonical.py` 的 `canonicalize`。

`NAME_RULES` 一张表同时供**渲染**（规范 IR → 清单，`materialize` 用）与**解析**
（清单 → 规范 IR，流水线用）。这一处看起来是「自己给自己判卷」，破法有三条，
都必须成立：

- 规范 IR 的**权威来源是 config**（`extract.py` 完全不看参数名，只做算术）；
  解析器则**完全不看 config 的规模字段**，只从名字与形状重建。两者互校。
- 融合切分需要的头数取自 config，但**宽度必须对得上**（`(n_heads+2·n_kv)·head_dim`
  必须等于清单里的实际宽度），对不上就报错 —— 所以它是一次真实的互校，不是自证。
- `data/zoo/_adversarial/` 里放着**人手写的**、渲染器产不出来的命名风格
  （`layers_3_attn_q` / `block3.mixer.Wq` / `h.3.attn.c_attn.bias`），专门盯
  「解析器是不是只认得自己渲染出来的那一种写法」。
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "CONFIG_ALIASES",
    "NON_STRUCTURAL_KEYS",
    "CONFIG_DEFAULTS",
    "NAME_RULES",
    "NameRule",
    "LAYOUTS",
    "CONDITIONAL_RULES",
    "ManifestError",
    "norm_kind_of",
    "parse_config",
    "parse_manifest",
    "render_manifest",
    "expand_paths",
    "role_of",
    "rule_coverage",
    "rule_report",
    "SPLIT_ARITY",
]

# ---------------------------------------------------------------- config 键名归一

#: 原 config 键 → 规范键。
CONFIG_ALIASES: dict[str, str] = {
    # 规模
    "num_hidden_layers": "n_layers",
    "n_layer": "n_layers",
    "hidden_size": "hidden",
    "n_embd": "hidden",
    "num_attention_heads": "n_heads",
    "n_head": "n_heads",
    "num_key_value_heads": "n_kv_heads",
    "intermediate_size": "intermediate",
    "n_inner": "intermediate",
    "vocab_size": "vocab",
    "max_position_embeddings": "max_positions",
    "n_positions": "max_positions",
    "n_ctx": "max_positions",
    # 定性
    "hidden_act": "activation",
    "activation_function": "activation",
    "rms_norm_eps": "norm_eps",
    "layer_norm_epsilon": "norm_eps",
    "rope_theta": "rope_theta",
    "rope_scaling": "rope_scaling",
    "sliding_window": "sliding_window",
    "use_sliding_window": "use_sliding_window",
    "max_window_layers": "max_window_layers",
    "tie_word_embeddings": "tie_embeddings",
    "attention_bias": "attention_bias",
    "mlp_bias": "mlp_bias",
    # MoE
    "num_local_experts": "n_experts",
    "n_routed_experts": "n_experts",
    "num_experts_per_tok": "top_k",
    "n_shared_experts": "n_shared_experts",
    "moe_intermediate_size": "moe_intermediate",
    "n_group": "n_groups",
    "topk_group": "topk_group",
    "first_k_dense_replace": "first_k_dense",
    "moe_layer_freq": "moe_layer_freq",
    "scoring_func": "score_func",
    # MLA
    "q_lora_rank": "q_lora_rank",
    "kv_lora_rank": "kv_lora_rank",
    "qk_nope_head_dim": "qk_nope_head_dim",
    "qk_rope_head_dim": "qk_rope_head_dim",
    "v_head_dim": "v_head_dim",
    # 身份（进 identity，不进指纹）
    "model_type": "source_model_type",
    "architectures": "architectures",
}

def _non_structural(groups: Mapping[str, tuple[str, ...]]) -> dict[str, str]:
    """`{键: 理由}` —— 把「按类别分组写」展开成扁平表。

    写成 `{理由: (键...)}` 而不是逐个键写一句话，是因为**理由只有那么几条**，
    而键有四十个。逐键写会让四十条注释变成四十种说法，读的人分不清它们到底是不是
    同一个理由 —— 而「为什么忽略这个键」恰恰是这张表唯一有价值的信息。
    """
    out: dict[str, str] = {}
    for reason, keys in groups.items():
        for k in keys:
            out[k] = reason
    return out


#: 明确**不结构**的键 → 忽略它的理由。它们出现与否不该改变指纹，否则
#: 「换个 transformers 版本」「把 dropout 从 0.1 调成 0」就成了「结构变了」。
#:
#: **这张表本身就是一条结论。** Qwen2-7B 与 Qwen2.5-7B 的 config 差异全部落在
#: 这张表里（`transformers_version` / `torch_dtype` / `rope_scaling` 的写法），
#: 所以它们的全指纹逐位相同 —— 「发布了新版本」和「结构变了」是两件事。
#:
#: 每条都必须给出理由：`coverage` 子命令与报告会把「忽略的键 + 理由」一并打印，
#: 于是「我们看不懂」与「我们决定不看」在报告里是**两栏**，不会被混成一句
#: 「覆盖率 100%」。
NON_STRUCTURAL_KEYS: dict[str, str] = _non_structural(
    {
        "元数据：记录训练 / 导出环境，不描述结构": (
            "_name_or_path",
            "transformers_version",
            "torch_dtype",
            "initializer_range",
            "pretraining_tp",
            "auto_map",
            "task_specific_params",
        ),
        "分词器 id：属于词表内容，不属于网络结构": (
            "bos_token_id",
            "eos_token_id",
            "pad_token_id",
        ),
        "dropout：训练期正则，前向图里没有它": (
            "attn_pdrop",
            "embd_pdrop",
            "resid_pdrop",
            "attention_dropout",
            "summary_first_dropout",
        ),
        "GPT-2 的摘要/分类头：本项目只建因果 LM 头，不做微调": (
            "summary_activation",
            "summary_proj_to_labels",
            "summary_type",
            "summary_use_proj",
        ),
        "运行时开关：影响执行方式，不影响图": (
            "use_cache",
            "output_router_logits",
            "use_mrope",
        ),
        "MoE 训练超参：辅助损失与路由平滑，不改变算子与连接": (
            "router_aux_loss_coef",
            "aux_loss_alpha",
            "seq_aux",
            "norm_topk_prob",
            "routed_scaling_factor",
            "topk_method",
        ),
        "与 max_position_embeddings 重复的长度声明，且长度本身只进全指纹": (
            "original_max_position_embeddings",
            "max_sequence_length",
        ),
    }
)

#: 缺失时的默认值。**必须显式列出**：`num_key_value_heads` 缺席在 HF 里意味着
#: 「等于 num_attention_heads」，也就是 MHA —— 猜错就会把 Llama-1 认成 GQA。
#:
#: `None` 在这里有特殊含义：**「config 没表态，以张量清单为准」**。`tie_embeddings`
#: 与 `attention_bias` 都必须是三态 —— GPT-2 的 config 里根本没有
#: `tie_word_embeddings` 这个键，但它其实是绑定的（`lm_head.weight` 在 checkpoint
#: 里缺席）。给一个 `False` 默认值就等于替它撒谎，而互校会因此误报。
CONFIG_DEFAULTS: dict[str, Any] = {
    "rope_theta": 10000.0,
    "rope_scaling": None,
    "sliding_window": None,
    "use_sliding_window": True,
    "max_window_layers": None,
    "tie_embeddings": None,
    "attention_bias": None,
    # 偏置不是一个布尔，是**三个**：
    #   - `qkv_bias`：q/k/v 三个投影带不带偏置；
    #   - `o_bias`：o_proj 带不带偏置；
    #   - `mlp_bias`：FFN 带不带偏置。
    # HF 的 `attention_bias` 一旦声明就同时管前两个，所以它只用来**填**这两个，
    # 不直接当判据。为什么必须拆开：Qwen1.5 / Qwen2 / Qwen2.5 的 `config.json`
    # **根本没有** `attention_bias` 这个键，但官方 checkpoint 的 q/k/v 带偏置、o_proj
    # 不带 —— 三个模型的精确参数量比「无偏置」多出的部分恰好是
    # `n_layers × (n_heads + 2·n_kv_heads)·head_dim`（Qwen2-7B：28 × 4608 = 129,024）。
    # 也就是说：这件事 config 说不出来，**只有参数名说得出来**。
    "qkv_bias": None,
    "o_bias": None,
    "mlp_bias": None,
    # 位置编码类型同样是 config-only 事实（`n_positions` / `activation_function` 是
    # GPT-2 系的指纹）。做成事实而不是到处传参，是为了让清单侧能反过来校它。
    "position_kind": None,
    "n_experts": 0,
    "top_k": 0,
    "n_shared_experts": 0,
    "moe_intermediate": None,
    "n_groups": 1,
    "topk_group": 1,
    "first_k_dense": 0,
    "moe_layer_freq": 1,
    "score_func": "softmax",
    "q_lora_rank": None,
    "kv_lora_rank": None,
    "qk_nope_head_dim": None,
    "qk_rope_head_dim": None,
    "v_head_dim": None,
}


def norm_kind_of(raw: Mapping[str, Any]) -> str:
    """范数类型**只能**从 config 推：`rms_norm_eps` ⇒ rmsnorm，`layer_norm_epsilon` ⇒ layernorm。

    张量清单给不出这个信息（Llama 的键名叫 `input_layernorm`，实际是 RMSNorm），
    所以这是一条 config-only 事实 —— 本项目把它显式标出来，而不是假装能推出来。
    """
    if "rms_norm_eps" in raw:
        return "rmsnorm"
    if "layer_norm_epsilon" in raw:
        return "layernorm"
    return "rmsnorm"


def position_kind_of(raw: Mapping[str, Any]) -> str:
    """`learned` / `rope`。

    **config-only 事实。** 真实 config 里没有「位置编码类型」这个键：GPT-2 用
    `n_positions` + 可学习嵌入，Llama 用 `rms_norm_eps` + RoPE。这里只能按家族指纹判。
    判错了不会静默：清单侧有 `pos_embed.weight` 就说明是可学习的，`canonicalize`
    拿这条事实去和清单对质，对不上就报错。

    做成 `config` 事实的另一个原因：路径 ②（公式）和清单渲染都要用它。第一版公式
    没读它，于是给每个 RoPE 模型都白加了一张 `max_positions × hidden` 的位置表
    （Llama-2-7B 多算 17M、Mistral-7B 多算 134M）。
    """
    if "n_positions" in raw or "activation_function" in raw:
        return "learned"
    return "rope"


def parse_config(
    raw: Mapping[str, Any],
    conventions: Mapping[str, Any] | None = None,
) -> tuple[list[tuple[str, Any, str, str]], list[str], list[tuple[str, str]]]:
    """config → 归一化事实 + **读不懂的键** + **有理由忽略的键**。

    第二、三项是覆盖率证据，**必须分开**：把「我们看不懂 `foo_bar`」和
    「我们决定不看 `attention_dropout`」混成一栏，报告就会把两者一起报成
    「覆盖率 100%」—— 而前者是真问题，后者不是。第二项应当为空。

    `conventions` 是 zoo 按官方 checkpoint 声明的家族约定，**只用来填 config
    沉默的那几个键**（`qkv_bias` / `o_bias` / `mlp_bias` 之类）。它不是「随便给的
    默认值」：填进去的每一项都会带着 `zoo:convention` 的来源标记，并在
    `canonicalize` 里和清单对质 —— 声明错了就报错。config 里**声明过**的键永远
    优先于约定。

    返回的每一项是 `(规范键, 值, 原键名, 命中规则)`。`facts` 子命令直接打印它，
    于是「这个数字从哪来」是可追溯的。
    """
    conv = dict(conventions or {})
    facts: list[tuple[str, Any, str, str]] = []
    unknown: list[str] = []
    ignored: list[tuple[str, str]] = []
    taken: dict[str, str] = {}

    for src_key, value in raw.items():
        if src_key in NON_STRUCTURAL_KEYS:
            ignored.append((src_key, NON_STRUCTURAL_KEYS[src_key]))
            continue
        canon = CONFIG_ALIASES.get(src_key)
        if canon is None:
            unknown.append(src_key)
            continue
        if canon in taken:
            # 两个别名同时出现（gpt2 的 n_ctx 与 n_positions）。**必须一致**，
            # 不一致就是两份真相，只能报错。
            prev = next(v for k, v, s, _ in facts if k == canon and s == taken[canon])
            if prev != value:
                raise ValueError(f"config 里 {taken[canon]}={prev!r} 与 {src_key}={value!r} 冲突（都归一为 {canon}）")
            continue
        taken[canon] = src_key
        facts.append((canon, value, src_key, f"alias:{src_key}"))

    norm_src = "rms_norm_eps" if "rms_norm_eps" in raw else "layer_norm_epsilon"
    facts.append(("norm_kind", norm_kind_of(raw), norm_src, "derived:norm_kind"))
    facts.append(("position_kind", position_kind_of(raw), "", "derived:position_kind"))

    # `attention_bias` 一旦声明就同时管 q/k/v 与 o（HF 的语义）。声明了就照它填，
    # 没声明就留 `None` —— 留给「zoo 里按 checkpoint 声明的家族约定」去填，
    # 再退到 `norm_kind` 兜底。**默认值不能替模型撒谎**：猜错了互校会误报。
    if "attention_bias" in raw:
        facts.append(("qkv_bias", bool(raw["attention_bias"]), "attention_bias", "derived:qkv_bias<-attention_bias"))
        facts.append(("o_bias", bool(raw["attention_bias"]), "attention_bias", "derived:o_bias<-attention_bias"))

    by_key = {k: v for k, v, _, _ in facts}
    for key, default in CONFIG_DEFAULTS.items():
        if key in by_key:
            continue
        if key in conv:
            facts.append((key, conv[key], "", f"zoo:convention:{key}"))
            continue
        facts.append((key, default, "", f"default:{key}"))

    # 派生量。注意 `n_kv_heads` 的默认值取决于 `n_heads`，所以只能在这里补。
    n_heads, hidden = by_key.get("n_heads"), by_key.get("hidden")
    if by_key.get("n_kv_heads") is None:
        if n_heads is None:
            raise ValueError("config 既没有 num_key_value_heads 也没有 num_attention_heads")
        facts.append(("n_kv_heads", n_heads, "", "default:n_kv_heads=n_heads"))
    if n_heads and hidden:
        facts.append(("head_dim", hidden // n_heads, "", "derived:head_dim=hidden//n_heads"))
    if by_key.get("intermediate") is None:
        if hidden is None:
            raise ValueError("config 既没有 intermediate_size 也没有 hidden_size")
        facts.append(("intermediate", 4 * hidden, "", "derived:gpt2_intermediate=4*hidden"))

    facts.sort(key=lambda t: t[0])
    return facts, sorted(unknown), sorted(ignored)


# ---------------------------------------------------------------- 参数名归一

@dataclass(frozen=True)
class NameRule:
    """一条命名规则。**同时**供渲染与解析使用。

    - `source`：原始清单里的名字模板，`{i}` 是层号、`{e}` 是专家号。
    - `canonical`：规范张量路径模板。多于一条 ⇒ 源名字是融合的，解析时拆成多条。
      单条且带 `split` ⇒ 那条路径本身是**融合路径**（规范 IR 里不存在），
      解析时按 `split` 拆开、渲染时由分量合成。
    - `split`：融合切分策略；`None` 表示一对一。
    - `transpose`：源权重是否 `(in, out)` 布局（GPT-2 的 `Conv1D`）。
    """

    source: str
    canonical: tuple[str, ...]
    layouts: tuple[str, ...]
    split: str | None = None
    transpose: bool = False

    # 这两个方法在 `parse_manifest` / `render_manifest` 里是 **张量 × 规则** 的内层循环，
    # 每次调用都重新拼串 + `re.compile`。规则本身是 frozen dataclass、哈希按值算，
    # 所以缓存挂在 `self` 上正合适：同值规则共用一份，返回的 `re.Pattern` 不可变、
    # `.match()` 也不会留下状态，缓存出去是安全的。
    @functools.lru_cache(maxsize=None)
    def source_regex(self) -> re.Pattern[str]:
        return re.compile("^" + _to_regex(self.source) + "$")

    @functools.lru_cache(maxsize=None)
    def canonical_regex(self) -> re.Pattern[str]:
        return re.compile("^" + _to_regex(self.canonical[0]) + "$")


def _to_regex(template: str) -> str:
    """把 `model.layers.{i}.self_attn.q_proj.weight` 变成带命名组的正则。"""
    out = re.escape(template)
    out = out.replace(re.escape("{i}"), r"(?P<i>\d+)")
    out = out.replace(re.escape("{e}"), r"(?P<e>\d+)")
    return out


def _rules(
    layouts: Sequence[str],
    pairs: Sequence[tuple[str, str]],
    *,
    split: str | None = None,
    transpose: bool = False,
) -> list[NameRule]:
    return [
        NameRule(source=src, canonical=(canon,), layouts=tuple(layouts), split=split, transpose=transpose)
        for src, canon in pairs
    ]


_L = ("llama",)          # model.layers.N + input_layernorm（Llama / Qwen2 系）
_V = ("vllm_llama",)     # vLLM 的融合布局（Phi-3 官方清单恰好就是这个形状）
_G = ("gpt2",)           # transformer.h.N + ln_1 + Conv1D（要转置）
_M = ("mixtral",)
_VM = ("vllm_mixtral",)  # vLLM 的 Mixtral：注意力/范数沿用 Mixtral 名，MoE 才融合
#: 共享规则用的合并组。vLLM 的 Mixtral 只在 **MoE** 上换了名字（`w13_weight`），
#: 注意力、范数、词表投影全是 Mixtral 原样，所以这些组要带上它。
#: 注意 MoE 那一组**不能**用它 —— 否则 `w1/w3` 的一对一规则会先命中，
#: `w13_weight` 的融合规则就永远轮不到。
_MM = _M + _VM
_D = ("deepseek",)
_C = ("custom_v1",)      # 自研：无 .weight 后缀、Wq / Wgate 这种短名

_RULE_LIST: list[NameRule] = [
    # ---- 顶层 ----
    *_rules(_L + _V + _MM + _D, [
        ("model.embed_tokens.weight", "embed.weight"),
        ("model.norm.weight", "final_norm.weight"),
        ("lm_head.weight", "head.weight"),
    ]),
    *_rules(_G, [
        ("transformer.wte.weight", "embed.weight"),
        ("transformer.wpe.weight", "pos_embed.weight"),
        ("transformer.ln_f.weight", "final_norm.weight"),
        ("transformer.ln_f.bias", "final_norm.bias"),
        # GPT-2 的 `lm_head.weight` 在官方 checkpoint 里**并不存在**：它与 `wte` 共享。
        # 规范 IR 里 `head.weight` 带着 `tied_with="embed.weight"`，所以渲染时走
        # `absent_but_tied` 分支（名字用规则算，形状不落盘），解析时再还原成绑定关系。
        # 也就是说这条规则只用来算名字，不代表清单里真有一个 `lm_head.weight`。
        ("lm_head.weight", "head.weight"),
    ]),
    *_rules(_C, [
        ("tok_emb", "embed.weight"),
        ("pos_emb", "pos_embed.weight"),
        ("out_proj", "head.weight"),
        ("final_norm", "final_norm.weight"),
    ]),
    # ---- 每块的范数 ----
    *_rules(_L + _V + _MM + _D, [
        ("model.layers.{i}.input_layernorm.weight", "blocks[{i}].norm1.weight"),
        ("model.layers.{i}.post_attention_layernorm.weight", "blocks[{i}].norm2.weight"),
    ]),
    *_rules(_G, [
        ("transformer.h.{i}.ln_1.weight", "blocks[{i}].norm1.weight"),
        ("transformer.h.{i}.ln_1.bias", "blocks[{i}].norm1.bias"),
        ("transformer.h.{i}.ln_2.weight", "blocks[{i}].norm2.weight"),
        ("transformer.h.{i}.ln_2.bias", "blocks[{i}].norm2.bias"),
    ]),
    *_rules(_C, [
        ("layers.{i}.pre_norm", "blocks[{i}].norm1.weight"),
        ("layers.{i}.post_norm", "blocks[{i}].norm2.weight"),
        ("layers.{i}.pre_norm.bias", "blocks[{i}].norm1.bias"),
        ("layers.{i}.post_norm.bias", "blocks[{i}].norm2.bias"),
        ("final_norm.bias", "final_norm.bias"),
    ]),
    # ---- 注意力（未融合）----
    *_rules(_L + _MM + _D, [
        ("model.layers.{i}.self_attn.q_proj.weight", "blocks[{i}].attn.q_proj.weight"),
        ("model.layers.{i}.self_attn.o_proj.weight", "blocks[{i}].attn.o_proj.weight"),
    ]),
    # k/v 投影**不属于 MLA 布局**：DeepSeek 的 KV 走 `kv_a_proj_with_mqa` + `kv_b_proj`。
    # 早先这一组写的是 `_L + _MM + _D`，于是 `deepseek` 布局里躺着两条永远匹配不上的
    # k/v 规则 —— 而「一条永远匹配不上的规则」看起来像「支持了某种布局」，实际上
    # 从没被任何清单验过。这条测试（`test_every_name_rule_is_used_or_declared`）就是
    # 为了逼出这种声明过宽。
    *_rules(_L + _MM, [
        ("model.layers.{i}.self_attn.k_proj.weight", "blocks[{i}].attn.k_proj.weight"),
        ("model.layers.{i}.self_attn.v_proj.weight", "blocks[{i}].attn.v_proj.weight"),
    ]),
    # MLA 的投影（DeepSeek 用 q_proj，或 q_a_proj + q_b_proj，取决于 q_lora_rank）。
    # 只声明给 `_D`：`llama` / `vllm_llama` 布局里不会有 MLA —— zoo 里的真 DeepSeek
    # 清单走 `deepseek` 布局，所以把它们挂到 llama 系前缀上只是**看起来**支持。
    *_rules(_D, [
        ("model.layers.{i}.self_attn.q_a_proj.weight", "blocks[{i}].attn.q_a_proj.weight"),
        ("model.layers.{i}.self_attn.q_b_proj.weight", "blocks[{i}].attn.q_b_proj.weight"),
        ("model.layers.{i}.self_attn.kv_a_proj_with_mqa.weight", "blocks[{i}].attn.kv_a_proj_with_mqa.weight"),
        ("model.layers.{i}.self_attn.kv_b_proj.weight", "blocks[{i}].attn.kv_b_proj.weight"),
        # 压缩潜向量上的那两道 RMSNorm。名字里带 `layernorm` 但其实是 RMSNorm，
        # 而且宽度是 `q_lora_rank` / `kv_lora_rank` 而不是 `hidden` —— 这两点都只有
        # config 说得清，所以它们既是「同名不同物」的例子，也是「config 才能给的数」。
        ("model.layers.{i}.self_attn.q_a_layernorm.weight", "blocks[{i}].attn.q_a_layernorm.weight"),
        ("model.layers.{i}.self_attn.kv_a_layernorm.weight", "blocks[{i}].attn.kv_a_layernorm.weight"),
    ]),
    # 逐头 RMSNorm：**不是** MLA 的一部分（MLA 用的是 `q_a_layernorm` / `kv_a_layernorm`）。
    # 这是 Qwen3 系在 llama 前缀下加的两张表，zoo 里没有 Qwen3，所以它目前只有解析方向
    # 能验 —— 见 `CONDITIONAL_RULES`。
    *_rules(_L, [
        ("model.layers.{i}.self_attn.q_norm.weight", "blocks[{i}].attn.q_norm.weight"),
        ("model.layers.{i}.self_attn.k_norm.weight", "blocks[{i}].attn.k_norm.weight"),
    ]),
    # MoE 的 router / 专家 / 共享专家（llama 前缀下的名字，Qwen1.5-MoE 与
    # DeepSeek 的 HF 清单都用它）。条件是 `n_experts > 0`。
    *_rules(_L, [
        ("model.layers.{i}.mlp.gate.weight", "blocks[{i}].ffn.router.weight"),
        ("model.layers.{i}.mlp.experts[{e}].gate_proj.weight", "blocks[{i}].ffn.experts[{e}].gate_proj.weight"),
        ("model.layers.{i}.mlp.experts[{e}].up_proj.weight", "blocks[{i}].ffn.experts[{e}].up_proj.weight"),
        ("model.layers.{i}.mlp.experts[{e}].down_proj.weight", "blocks[{i}].ffn.experts[{e}].down_proj.weight"),
        ("model.layers.{i}.mlp.shared_experts.gate_proj.weight", "blocks[{i}].ffn.shared_experts.gate_proj.weight"),
        ("model.layers.{i}.mlp.shared_experts.up_proj.weight", "blocks[{i}].ffn.shared_experts.up_proj.weight"),
        ("model.layers.{i}.mlp.shared_experts.down_proj.weight", "blocks[{i}].ffn.shared_experts.down_proj.weight"),
    ]),
    # ---- 注意力（融合）----
    *_rules(_V, [
        ("model.layers.{i}.self_attn.qkv_proj.weight", "blocks[{i}].attn.qkv_proj.weight"),
    ], split="qkv"),
    *_rules(_G, [
        ("transformer.h.{i}.attn.c_attn.weight", "blocks[{i}].attn.qkv_proj.weight"),
        ("transformer.h.{i}.attn.c_attn.bias", "blocks[{i}].attn.qkv_proj.bias"),
    ], split="qkv", transpose=True),
    *_rules(_G, [
        ("transformer.h.{i}.attn.c_proj.weight", "blocks[{i}].attn.o_proj.weight"),
        ("transformer.h.{i}.attn.c_proj.bias", "blocks[{i}].attn.o_proj.bias"),
    ], transpose=True),
    *_rules(_C, [
        ("layers.{i}.attn.Wq", "blocks[{i}].attn.q_proj.weight"),
        ("layers.{i}.attn.Wk", "blocks[{i}].attn.k_proj.weight"),
        ("layers.{i}.attn.Wv", "blocks[{i}].attn.v_proj.weight"),
        ("layers.{i}.attn.Wo", "blocks[{i}].attn.o_proj.weight"),
        ("layers.{i}.attn.Wkva", "blocks[{i}].attn.kv_a_proj_with_mqa.weight"),
        ("layers.{i}.attn.Wkvb", "blocks[{i}].attn.kv_b_proj.weight"),
        # 自研方言的 MLA q 侧（q_lora_rank 成立时才出现）—— 见 `CONDITIONAL_RULES`。
        ("layers.{i}.attn.Wqa", "blocks[{i}].attn.q_a_proj.weight"),
        ("layers.{i}.attn.Wqb", "blocks[{i}].attn.q_b_proj.weight"),
        ("layers.{i}.attn.Nqa", "blocks[{i}].attn.q_a_layernorm.weight"),
        ("layers.{i}.attn.Nkva", "blocks[{i}].attn.kv_a_layernorm.weight"),
    ]),
    # 这里**没有** `layers.{i}.attn.Wqkv` / `layers.{i}.mlp.W12`。
    # 自研方言不融合：`Wq`/`Wk`/`Wv` 与 `Wgate`/`Wup`/`Wdown` 是一对一规则，
    # 而 `rule_for` 先试一对一、再试融合，所以那两条融合规则**永远轮不到**
    # （融合规则的渲染方向靠分量名回填模板，一对一规则会先命中）。
    # 换句话说它们不是「备用方言」，是死代码：删掉比留着更诚实。
    # 融合的渲染方向由 `_V`（vLLM）与 `_G`（GPT-2 的 `c_attn`）覆盖。
    # ---- FFN ----
    # 注意 `_V` 不在这条里：vLLM 把 gate/up 融成 `gate_up_proj`，所以它只能由下面的
    # 融合规则产出。若这里也放 `_V`，一对一规则会先命中，融合规则就成了死代码。
    # `_MM` 也不在：Mixtral 的 32 层**全是** MoE，没有稠密 FFN，所以 `mixtral` /
    # `vllm_mixtral` 布局里不存在 `mlp.gate_proj.weight` 这种名字。
    # `_D` 在：DeepSeek-V2 的 `first_k_dense_replace=1`，第 0 块就是稠密 FFN。
    *_rules(_L + _D, [
        ("model.layers.{i}.mlp.gate_proj.weight", "blocks[{i}].ffn.gate_proj.weight"),
        ("model.layers.{i}.mlp.up_proj.weight", "blocks[{i}].ffn.up_proj.weight"),
        ("model.layers.{i}.mlp.down_proj.weight", "blocks[{i}].ffn.down_proj.weight"),
    ]),
    # vLLM 只融合 qkv 与 gate_up，**不**融合 o_proj / down_proj，这两条要单独给。
    *_rules(_V, [
        ("model.layers.{i}.self_attn.o_proj.weight", "blocks[{i}].attn.o_proj.weight"),
        ("model.layers.{i}.mlp.down_proj.weight", "blocks[{i}].ffn.down_proj.weight"),
    ]),
    *_rules(_V, [
        ("model.layers.{i}.mlp.gate_up_proj.weight", "blocks[{i}].ffn.gate_up_proj.weight"),
    ], split="gate_up"),
    *_rules(_G, [
        ("transformer.h.{i}.mlp.c_fc.weight", "blocks[{i}].ffn.fc_in.weight"),
        ("transformer.h.{i}.mlp.c_fc.bias", "blocks[{i}].ffn.fc_in.bias"),
        ("transformer.h.{i}.mlp.c_proj.weight", "blocks[{i}].ffn.fc_out.weight"),
        ("transformer.h.{i}.mlp.c_proj.bias", "blocks[{i}].ffn.fc_out.bias"),
    ], transpose=True),
    *_rules(_C, [
        ("layers.{i}.mlp.Wgate", "blocks[{i}].ffn.gate_proj.weight"),
        ("layers.{i}.mlp.Wup", "blocks[{i}].ffn.up_proj.weight"),
        ("layers.{i}.mlp.Wdown", "blocks[{i}].ffn.down_proj.weight"),
        ("layers.{i}.mlp.Wfc1", "blocks[{i}].ffn.fc_in.weight"),
        ("layers.{i}.mlp.Wfc2", "blocks[{i}].ffn.fc_out.weight"),
        ("layers.{i}.mlp.Wfc1.bias", "blocks[{i}].ffn.fc_in.bias"),
        ("layers.{i}.mlp.Wfc2.bias", "blocks[{i}].ffn.fc_out.bias"),
    ]),
    # ---- MoE ----
    # `_L` 在：**Qwen1.5-MoE / Qwen2-MoE 用的是 llama 前缀 + MoE 的 MLP**，
    # 所以 `model.layers.{i}.mlp.gate.weight` / `mlp.experts[{e}].*` /
    # `mlp.shared_experts.*` 确实是 `llama` 布局的词汇（DeepSeek 的 HF 清单同样
    # 用这套前缀，只是它另有 `deepseek` 布局）。它们是**条件规则** ——
    # 只在 `n_experts > 0` 时出现，zoo 里的 llama 系模型全是稠密的。
    #
    # 这条 `_L` 我删过一次：当时死规则测试报了 7 条「llama 布局下从没被匹配过的
    # MoE 规则」，我按「声明过宽」处理了。**那是过度修正** —— 紧接着对抗用例
    # `llama-2-7b.wrong_role.json` 就红了：它把 `mlp.gate_proj` 写成 `mlp.gate`，
    # 想验的是「名字匹配上了一条规则、但匹配错了，挡住它的是块级结构校验而不是名字表」。
    # 删掉 `_L` 之后这条规则不再匹配，拒绝提前到了名字表 —— 用例想证明的那件事没了。
    # 正确做法是把它登记进 `CONDITIONAL_RULES` 并配探针，不是删掉。
    *_rules(_L + _D, [
        ("model.layers.{i}.mlp.gate.weight", "blocks[{i}].ffn.router.weight"),
        ("model.layers.{i}.mlp.experts[{e}].gate_proj.weight", "blocks[{i}].ffn.experts[{e}].gate_proj.weight"),
        ("model.layers.{i}.mlp.experts[{e}].up_proj.weight", "blocks[{i}].ffn.experts[{e}].up_proj.weight"),
        ("model.layers.{i}.mlp.experts[{e}].down_proj.weight", "blocks[{i}].ffn.experts[{e}].down_proj.weight"),
        ("model.layers.{i}.mlp.shared_experts.gate_proj.weight", "blocks[{i}].ffn.shared_experts.gate_proj.weight"),
        ("model.layers.{i}.mlp.shared_experts.up_proj.weight", "blocks[{i}].ffn.shared_experts.up_proj.weight"),
        ("model.layers.{i}.mlp.shared_experts.down_proj.weight", "blocks[{i}].ffn.shared_experts.down_proj.weight"),
    ]),
    *_rules(_M, [
        ("model.layers.{i}.block_sparse_moe.gate.weight", "blocks[{i}].ffn.router.weight"),
        ("model.layers.{i}.block_sparse_moe.experts[{e}].w1.weight", "blocks[{i}].ffn.experts[{e}].gate_proj.weight"),
        ("model.layers.{i}.block_sparse_moe.experts[{e}].w3.weight", "blocks[{i}].ffn.experts[{e}].up_proj.weight"),
        ("model.layers.{i}.block_sparse_moe.experts[{e}].w2.weight", "blocks[{i}].ffn.experts[{e}].down_proj.weight"),
    ]),
    # vLLM 的 MoE 布局：`w13_weight` 是 w1+w3 的融合，`w2_weight` 与 gate 不融合。
    # `split=` 是**整次 `_rules` 调用**的属性，所以这三条不能写在一起 ——
    # 写在一起会把 router 和 w2 也标成融合规则（渲染时立刻报「模板里没有融合片段」）。
    *_rules(("vllm_mixtral",), [
        ("model.layers.{i}.block_sparse_moe.gate.weight", "blocks[{i}].ffn.router.weight"),
        ("model.layers.{i}.block_sparse_moe.experts[{e}].w2_weight", "blocks[{i}].ffn.experts[{e}].down_proj.weight"),
    ]),
    *_rules(("vllm_mixtral",), [
        ("model.layers.{i}.block_sparse_moe.experts[{e}].w13_weight", "blocks[{i}].ffn.experts[{e}].gate_up.weight"),
    ], split="w13"),
    *_rules(_C, [
        ("layers.{i}.moe.router", "blocks[{i}].ffn.router.weight"),
        ("layers.{i}.moe.experts[{e}].W1", "blocks[{i}].ffn.experts[{e}].gate_proj.weight"),
        ("layers.{i}.moe.experts[{e}].W3", "blocks[{i}].ffn.experts[{e}].up_proj.weight"),
        ("layers.{i}.moe.experts[{e}].W2", "blocks[{i}].ffn.experts[{e}].down_proj.weight"),
        ("layers.{i}.moe.shared.W1", "blocks[{i}].ffn.shared_experts.gate_proj.weight"),
        ("layers.{i}.moe.shared.W3", "blocks[{i}].ffn.shared_experts.up_proj.weight"),
        ("layers.{i}.moe.shared.W2", "blocks[{i}].ffn.shared_experts.down_proj.weight"),
    ]),
    # ---- 偏置 ----
    *_rules(_L + _MM + _D, [
        ("model.layers.{i}.self_attn.q_proj.bias", "blocks[{i}].attn.q_proj.bias"),
        ("model.layers.{i}.self_attn.k_proj.bias", "blocks[{i}].attn.k_proj.bias"),
        ("model.layers.{i}.self_attn.v_proj.bias", "blocks[{i}].attn.v_proj.bias"),
        ("model.layers.{i}.self_attn.o_proj.bias", "blocks[{i}].attn.o_proj.bias"),
        ("model.layers.{i}.mlp.gate_proj.bias", "blocks[{i}].ffn.gate_proj.bias"),
        ("model.layers.{i}.mlp.up_proj.bias", "blocks[{i}].ffn.up_proj.bias"),
        ("model.layers.{i}.mlp.down_proj.bias", "blocks[{i}].ffn.down_proj.bias"),
    ]),
    *_rules(_V, [
        ("model.layers.{i}.self_attn.o_proj.bias", "blocks[{i}].attn.o_proj.bias"),
        ("model.layers.{i}.mlp.down_proj.bias", "blocks[{i}].ffn.down_proj.bias"),
    ]),
    *_rules(_V, [
        ("model.layers.{i}.self_attn.qkv_proj.bias", "blocks[{i}].attn.qkv_proj.bias"),
    ], split="qkv"),
    *_rules(_V, [
        ("model.layers.{i}.mlp.gate_up_proj.bias", "blocks[{i}].ffn.gate_up_proj.bias"),
    ], split="gate_up"),
    *_rules(_C, [
        ("layers.{i}.attn.Wq.bias", "blocks[{i}].attn.q_proj.bias"),
        ("layers.{i}.attn.Wk.bias", "blocks[{i}].attn.k_proj.bias"),
        ("layers.{i}.attn.Wv.bias", "blocks[{i}].attn.v_proj.bias"),
        ("layers.{i}.attn.Wo.bias", "blocks[{i}].attn.o_proj.bias"),
    ]),
]

#: 布局 → 该布局下的规则（列表顺序即匹配优先级）。
NAME_RULES: dict[str, tuple[NameRule, ...]] = {}
for _rule in _RULE_LIST:
    for _layout in _rule.layouts:
        NAME_RULES.setdefault(_layout, [])  # type: ignore[arg-type]
        NAME_RULES[_layout].append(_rule)   # type: ignore[attr-defined]
NAME_RULES = {k: tuple(v) for k, v in NAME_RULES.items()}

#: 全部已知布局。`materialize` 与 `run` 都从这里取合法值。
LAYOUTS: tuple[str, ...] = tuple(sorted(NAME_RULES))

#: **条件规则**：只在某个 config 事实成立时才会出现的名字。
#:
#: 与 `NON_STRUCTURAL_KEYS` 同一套纪律 —— **「Zoo 实测过的」与「条件成立才出现的」
#: 必须分两栏**。合并成一栏的后果是：一条从没被任何清单匹配过的规则看起来像
#: 「支持了某种布局」，而它某天真的被匹配上时，行为是**未经验证的**。
#:
#: 右边那一栏是**门槛**：哪个事实为真时这条规则才会出现。
#:   - `qkv_bias` / `o_bias` / `mlp_bias`：`CONFIG_DEFAULTS` 里的三态偏置事实；
#:   - `q_lora_rank`：MLA 的 q 侧低秩压缩（DeepSeek-V2 主模型有，Lite 没有）；
#:   - `n_experts`：MoE 的专家数。`llama` 布局的 MoE 名字（Qwen1.5-MoE 系）
#:     只在有专家时出现，而 zoo 里的 llama 系模型全是稠密的；
#:   - `qk_norm`：**清单侧**标志（Qwen3 的逐头 RMSNorm），config 里没有对应的键。
#:
#: 每条都要能被**探针**验出来 —— `tests/test_extract_dialects.py` 里
#: `test_conditional_rules_are_reachable` 把门槛翻到成立、真的把名字渲染出来
#: （或从清单解析出来）。**每一个条件规则都必须被至少一条探针覆盖**，
#: 否则登记表就成了藏死规则的抽屉。
CONDITIONAL_RULES: dict[str, str] = {
    # 偏置：zoo 里只有 Qwen 的 q/k/v 带偏置，`o_bias` / `mlp_bias` 全体为假。
    "model.layers.{i}.self_attn.q_proj.bias": "qkv_bias",
    "model.layers.{i}.self_attn.k_proj.bias": "qkv_bias",
    "model.layers.{i}.self_attn.v_proj.bias": "qkv_bias",
    "model.layers.{i}.self_attn.o_proj.bias": "o_bias",
    "model.layers.{i}.mlp.gate_proj.bias": "mlp_bias",
    "model.layers.{i}.mlp.up_proj.bias": "mlp_bias",
    "model.layers.{i}.mlp.down_proj.bias": "mlp_bias",
    "model.layers.{i}.mlp.gate_up_proj.bias": "mlp_bias",
    # MLA 的 q 侧低秩：`deepseek-v2-lite` 的 `q_lora_rank=None`，所以走 `q_proj`。
    "model.layers.{i}.self_attn.q_a_proj.weight": "q_lora_rank",
    "model.layers.{i}.self_attn.q_b_proj.weight": "q_lora_rank",
    "model.layers.{i}.self_attn.q_a_layernorm.weight": "q_lora_rank",
    "layers.{i}.attn.Wqa": "q_lora_rank",
    "layers.{i}.attn.Wqb": "q_lora_rank",
    "layers.{i}.attn.Nqa": "q_lora_rank",
    # llama 前缀下的 MoE 名字（Qwen1.5-MoE / DeepSeek 的 HF 清单）。
    "model.layers.{i}.mlp.gate.weight": "n_experts",
    "model.layers.{i}.mlp.experts[{e}].gate_proj.weight": "n_experts",
    "model.layers.{i}.mlp.experts[{e}].up_proj.weight": "n_experts",
    "model.layers.{i}.mlp.experts[{e}].down_proj.weight": "n_experts",
    "model.layers.{i}.mlp.shared_experts.gate_proj.weight": "n_experts",
    "model.layers.{i}.mlp.shared_experts.up_proj.weight": "n_experts",
    "model.layers.{i}.mlp.shared_experts.down_proj.weight": "n_experts",
    # 逐头 RMSNorm：Qwen3 系，zoo 里没有这个家族。
    "model.layers.{i}.self_attn.q_norm.weight": "qk_norm",
    "model.layers.{i}.self_attn.k_norm.weight": "qk_norm",
}


def _match_rule(rules: Sequence[NameRule], name: str) -> NameRule | None:
    """按 `parse_manifest` 的优先级找出第一个命中的规则。"""
    for rule in rules:
        if rule.source_regex().match(name):
            return rule
    return None


def rule_coverage(manifests: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """每条命名规则落在哪一栏：**实测** / **条件** / **死规则**。

    分类的判据是「这条规则有没有被某份清单真正走到」，而清单就是 `run` 读的那些
    （zoo 里每个模型 × 每个方言）。`dead` 必须恒为空 —— 它不为空意味着规则表里
    躺着一条谁也没验过的猜测。

    返回 `{布局: {n_rules, exercised, conditional, dead}}`，进报告与 `rules` 子命令。
    """
    hit: set[tuple[str, str]] = set()
    for m in manifests:
        layout = m.get("layout")
        rules = NAME_RULES.get(layout)
        if rules is None:
            continue
        entries = list(m.get("tensors") or ()) + list(m.get("absent_but_tied") or ())
        for entry in entries:
            rule = _match_rule(rules, entry["name"])
            if rule is not None:
                hit.add((layout, rule.source))

    out: dict[str, dict[str, Any]] = {}
    for layout, rules in sorted(NAME_RULES.items()):
        exercised, conditional, dead = [], [], []
        for rule in rules:
            if (layout, rule.source) in hit:
                exercised.append(rule.source)
            elif rule.source in CONDITIONAL_RULES:
                conditional.append(rule.source)
            else:
                dead.append(rule.source)
        out[layout] = {
            "n_rules": len(rules),
            "exercised": exercised,
            "conditional": conditional,
            "dead": dead,
        }
    return out


def rule_report() -> dict[str, int]:
    """规则表的规模（进报告，也让「规则是不是死代码」可核对）。"""
    return {layout: len(rules) for layout, rules in sorted(NAME_RULES.items())}

#: 融合策略 → 拆分出的分量数。切分算不出来时直接报错。
SPLIT_ARITY = {"qkv": 3, "gate_up": 2, "w13": 2}

#: 融合路径片段 → 分量角色名（解析与渲染共用的命名约定）。
_SPLIT_PARTS = {
    "qkv": ("q_proj", "k_proj", "v_proj"),
    "gate_up": ("gate_proj", "up_proj"),
    "w13": ("gate_proj", "up_proj"),
}

#: 融合路径片段本身（规范 IR 里不该出现）。
#: `gate_up` 与 `gate_up_proj` 都要列：vLLM 的稠密布局用 `gate_up_proj`，
#: 而 vLLM 的 MoE 布局沿用 Mixtral 的 `w13` / `gate_up` 习惯，两者并不统一。
FUSED_MARKERS = ("qkv_proj", "gate_up_proj", "gate_up", "w13")

#: 最长优先的匹配顺序。`gate_up` 是 `gate_up_proj` 的子串，短的先试会切错位置 ——
#: 这条约束按常量排一次就够，不必在 `_fused_marker` 的每次调用里重排（它在
#: 解析与渲染的热路径上，每条融合规则都要走一遍）。
FUSED_MARKERS_LONGEST_FIRST = tuple(sorted(FUSED_MARKERS, key=len, reverse=True))


def role_of(path: str) -> str:
    """规范路径 → 粗粒度角色（用于代价分项与报告聚合）。"""
    p = re.sub(r"^blocks\[\d+\]\.", "", path)
    p = re.sub(r"^blocks\[\*\]\.", "", p)
    p = re.sub(r"^experts\[\d+\]\.", "", p)
    p = re.sub(r"^experts\[\*\]\.", "", p)
    if p in ("embed.weight", "head.weight"):
        return "embed"
    if p.startswith("pos_embed"):
        return "position"
    if p.startswith("final_norm") or re.match(r"^norm[12]\.", p):
        return "norm"
    return p


class ManifestError(ValueError):
    """清单里出现了规则表解释不了的名字，或融合宽度对不上。**一律报错，绝不猜。**"""


def _fill(template: str, subst: Mapping[str, str]) -> str:
    try:
        return template.format(**subst)
    except KeyError as exc:  # pragma: no cover - 规则表写错才会到这里
        raise ManifestError(f"模板 {template!r} 需要 {exc}，但名字里没有这个组") from exc


def parse_manifest(
    manifest: Mapping[str, Any],
    *,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
) -> tuple[list[dict[str, Any]], list[str]]:
    """清单 → 规范张量条目。

    解析器**只**看名字与形状，加一个头数（融合切分要用，且宽度必须对得上）。
    返回 `(tensors, notes)`；`tensors` 每项含 `path` / `shape` / `source_name` / `tied_with`。
    """
    layout = manifest["layout"]
    if layout not in NAME_RULES:
        raise ManifestError(f"未知布局 {layout!r}；已知：{list(LAYOUTS)}")
    rules = NAME_RULES[layout]
    out: list[dict[str, Any]] = []
    notes: list[str] = []

    for entry in manifest["tensors"]:
        name = entry["name"]
        shape = tuple(entry["shape"])
        matched = None
        for rule in rules:
            m = rule.source_regex().match(name)
            if m:
                matched = (rule, m)
                break
        if matched is None:
            raise ManifestError(f"{manifest.get('model')} / {layout}: 没有规则能解释参数名 {name!r}")
        rule, m = matched
        subst = {k: v for k, v in m.groupdict().items() if v is not None}
        src_shape = shape
        if rule.transpose:
            shape = tuple(reversed(shape))
            notes.append(f"转置 {name} {src_shape} → {shape}（Conv1D 是 (in, out) 布局）")
        if rule.split is None:
            out.append({"path": _fill(rule.canonical[0], subst), "shape": shape, "source_name": name, "tied_with": None})
            continue
        parts = _split_fused(rule.split, _fill(rule.canonical[0], subst), shape, n_heads, n_kv_heads, head_dim)
        out.extend({"path": p, "shape": s, "source_name": name, "tied_with": None} for p, s in parts)
        notes.append(f"拆分 {name} {src_shape} → {len(parts)} 条（{rule.split}）")

    for entry in manifest.get("absent_but_tied", []):
        name, shares = entry["name"], entry["shares_with"]
        matched = None
        for rule in rules:
            m = rule.source_regex().match(name)
            if m:
                matched = (rule, m)
                break
        if matched is None:
            raise ManifestError(f"{manifest.get('model')} / {layout}: 缺席共享张量 {name!r} 没有规则")
        rule, m = matched
        canon = _fill(rule.canonical[0], {k: v for k, v in m.groupdict().items() if v is not None})
        base = next((t for t in out if t["source_name"] == shares), None)
        if base is None:
            raise ManifestError(f"{manifest.get('model')}: {name!r} 声明共享 {shares!r}，但清单里没有它")
        out.append({"path": canon, "shape": base["shape"], "source_name": name, "tied_with": base["path"]})
        notes.append(f"共享 {name} = {shares}（清单里缺席，靠 tied 声明补出来）")

    leftovers = [t["path"] for t in out if any(f".{marker}." in t["path"] for marker in FUSED_MARKERS)]
    if leftovers:
        raise ManifestError(f"{manifest.get('model')}: 融合张量没被拆开：{leftovers}")

    out.sort(key=lambda t: t["path"])
    return out, notes


def _split_fused(
    kind: str, canon: str, shape: tuple[int, ...], n_heads: int, n_kv_heads: int, head_dim: int
) -> list[tuple[str, tuple[int, ...]]]:
    """把融合张量按**头结构**真的切开。切不开就报错。

    路径的改写是「把融合片段换成分量名」，不是「在尾巴上再拼一段」：
    `blocks[0].attn.qkv_proj.bias` → `blocks[0].attn.q_proj.bias`。
    早先的版本在 `rpartition('.')` 的 head 后面插分量名，于是融合名 `qkv_proj` 被留在
    了路径里（`...attn.qkv_proj.q_proj.bias`），下面那道「融合张量没被拆开」的闸门
    正好把它拦住 —— 闸门是有效的，只是错误发生在闸门上游。
    """
    parts = _SPLIT_PARTS[kind]
    marker = _fused_marker(canon)
    stem, _, leaf = canon.partition(f".{marker}.")

    def sub(part: str) -> str:
        return f"{stem}.{part}.{leaf}"

    if kind == "qkv":
        want = (n_heads + 2 * n_kv_heads) * head_dim
        if shape[-1] != want:
            raise ManifestError(
                f"融合 qkv 宽度对不上：清单 {shape[-1]}，按 (n_heads+2·n_kv_heads)·head_dim "
                f"= ({n_heads}+2·{n_kv_heads})·{head_dim} = {want}"
            )
        rows = shape[:-1]
        return [
            (sub(parts[0]), rows + (n_heads * head_dim,)),
            (sub(parts[1]), rows + (n_kv_heads * head_dim,)),
            (sub(parts[2]), rows + (n_kv_heads * head_dim,)),
        ]
    if kind in ("gate_up", "w13"):
        total = shape[0] if kind == "w13" else shape[-1]
        if total % 2:
            raise ManifestError(f"融合 gate/up 的宽度 {total} 不是偶数，无法二等分")
        half = total // 2
        if kind == "w13":
            return [(sub(parts[0]), (half,) + shape[1:]), (sub(parts[1]), (half,) + shape[1:])]
        rows = shape[:-1]
        return [(sub(parts[0]), rows + (half,)), (sub(parts[1]), rows + (half,))]
    raise ManifestError(f"未知融合策略 {kind!r}")


def expand_paths(
    tensors: Iterable[Mapping[str, Any]],
    *,
    block_groups: Sequence[tuple[int, int]],
    n_experts: int = 0,
) -> list[dict[str, Any]]:
    """把含 `[*]` 的**模式**路径展开成具体下标的路径。

    渲染清单时用：清单里必须是 `model.layers.7.self_attn.q_proj.weight` 这种**具体**名字，
    不能留 `blocks[*]`。展开顺序固定（组内升序、组间按 block_groups 顺序），
    所以同一份 IR 渲染两次逐位相同。

    只展开 `[*]` 通配符：`blocks[3]` / `experts[0]` 这种**已经具体**的下标原样保留。
    早先的版本对任何 `experts[...]` 都重新展开，于是 `experts[3]` 会被改写成
    `experts[0..7]` —— 一份 IR 里的 8 个专家变成 64 个，而且全挤到同一个融合桶里。
    """
    out: list[dict[str, Any]] = []
    for t in tensors:
        path = t["path"]
        block_idx = _group_indices(path, block_groups, "blocks")
        for i in block_idx:
            p1 = re.sub(r"blocks\[[^\]]+\]", f"blocks[{i}]", path)
            if "experts[*]" in p1:
                out.extend({**t, "path": p1.replace("experts[*]", f"experts[{e}]")} for e in range(n_experts))
            else:
                out.append({**t, "path": p1})
    out.sort(key=lambda d: d["path"])
    return out


def _group_indices(path: str, block_groups: Sequence[tuple[int, int]], token: str) -> list[int]:
    m = re.search(rf"{token}\[([^\]]+)\]", path)
    if m is None:
        return [0]
    marker = m.group(1)
    if marker == "*":
        # 最大的那组叫 `blocks[*]`；如果有多个组，取「非首组」中最大的那个。
        if len(block_groups) == 1:
            start, count = block_groups[0]
            return list(range(start, start + count))
        best = max(block_groups, key=lambda g: g[1])
        return list(range(best[0], best[0] + best[1]))
    idx = int(marker)
    for start, count in block_groups:
        if start <= idx < start + count:
            return [idx]
    raise ManifestError(f"{path}: 下标 {idx} 不在任何块组 {list(block_groups)} 里")


def _fused_marker(template: str) -> str:
    """从融合规则的规范模板里找出那个「融合片段」的名字（`qkv_proj` / `gate_up_proj` / `w13`…）。

    必须**最长优先**匹配：`gate_up` 是 `gate_up_proj` 的子串，短的先试会切错位置。
    """
    for marker in FUSED_MARKERS_LONGEST_FIRST:
        if f".{marker}." in template:
            return marker
    raise ManifestError(f"融合规则的模板 {template!r} 里找不到任何已知融合片段 {FUSED_MARKERS}")


def _component_pattern(rule: NameRule) -> re.Pattern[str]:
    """把融合规则的规范模板切成分量模板，得到一条「只匹配该融合策略的分量」的正则。

    渲染方向需要它：规范 IR 里永远只有分量（`q_proj` / `k_proj` / `v_proj`），而融合
    布局的规则是按融合名写的（`qkv_proj`）。命中后的命名组 `part` 必然是 `_SPLIT_PARTS`
    里的一个，因为它是用分量名做**交替**拼出来的 —— 通配符会顺带匹配到 `down_proj`
    这种同构但无关的名字，交替不会。
    """
    template = rule.canonical[0]
    parts = _SPLIT_PARTS[rule.split]  # type: ignore[index]
    marker = _fused_marker(template)
    head, _, tail = template.partition(f".{marker}.")
    if not head and not tail:
        raise ManifestError(f"融合模板 {template!r} 里 {marker} 占满了整条路径，切不出分量")
    alternatives = "|".join(parts)
    regex = "^" + _to_regex(head) + r"\.(?P<part>" + alternatives + r")\." + _to_regex(tail) + "$"
    return re.compile(regex)


def render_manifest(
    tensors: Iterable[Mapping[str, Any]],
    *,
    layout: str,
    model: str,
    dialect: str,
    block_groups: Sequence[tuple[int, int]],
    n_experts: int = 0,
) -> dict[str, Any]:
    """规范张量条目 → 某一命名方言的清单。

    渲染器与解析器共用 `NAME_RULES`。这不是「自己给自己判卷」：规范 IR 的权威来源是
    config（`extract.py` 不看参数名），解析器则不看 config 的规模字段 —— 真正的互校
    发生在 `canonical.py` 的 `canonicalize` 里，以及 `data/zoo/_adversarial/` 那些
    渲染器**产不出来**的手写清单上。
    """
    if layout not in NAME_RULES:
        raise ManifestError(f"未知布局 {layout!r}")
    rules = NAME_RULES[layout]
    concrete = expand_paths(tensors, block_groups=block_groups, n_experts=n_experts)
    by_path = {t["path"]: t for t in concrete}
    emitted: dict[str, dict[str, Any]] = {}
    tied: dict[str, dict[str, str]] = {}
    fused_buckets: dict[tuple[str, str], tuple[NameRule, list[dict[str, Any]]]] = {}

    def rule_for(canon_path: str) -> tuple[NameRule, dict[str, str]]:
        """先按一对一规则找，再按融合规则找（把分量嵌回融合模板）。"""
        for rule in rules:
            if rule.split is not None:
                continue
            m = rule.canonical_regex().match(canon_path)
            if m:
                return rule, {k: v for k, v in m.groupdict().items() if v is not None}
        for rule in rules:
            if rule.split is None:
                continue
            m = _component_pattern(rule).match(canon_path)
            if m and m.group("part") in _SPLIT_PARTS[rule.split]:
                subst = {k: v for k, v in m.groupdict().items() if k != "part" and v is not None}
                return rule, subst
        raise ManifestError(f"{model}: 规范路径 {canon_path!r} 在布局 {layout} 里没有对应名字")

    def out_shape(rule: NameRule, canonical: Sequence[int]) -> list[int]:
        """规范形状 → 该方言清单里实际落盘的形状。

        `transpose=True` 的规则（GPT-2 的 `Conv1D`）在清单里是 `(in, out)`，而规范 IR
        一律是 `(out, in)`。**渲染时必须转回去**，否则渲染→解析这一圈会把非方阵
        （`c_fc` 是 3072×768）悄悄转错，而方阵（GPT-2 的注意力）恰好掩盖这个错误。
        """
        return list(reversed(canonical)) if rule.transpose else list(canonical)

    for t in concrete:
        rule, subst = rule_for(t["path"])
        name = _fill(rule.source, subst)
        if rule.split is not None:
            bucket = fused_buckets.setdefault((name, rule.split), (rule, []))
            bucket[1].append(t)
            continue
        if t.get("tied_with"):
            partner = by_path.get(t["tied_with"])
            if partner is None:
                raise ManifestError(f"{model}: {t['path']} 声明共享 {t['tied_with']}，但它不在张量集里")
            prule, psubst = rule_for(partner["path"])
            tied[name] = {"name": name, "shares_with": _fill(prule.source, psubst)}
            continue
        emitted[name] = {"name": name, "shape": out_shape(rule, t["shape"])}

    for (name, kind), (rule, members) in fused_buckets.items():
        want = SPLIT_ARITY[kind]
        if len(members) != want:
            raise ManifestError(
                f"{model}: 融合名 {name} 需要 {want} 个分量，只找到 {len(members)}：{[m['path'] for m in members]}"
            )
        members.sort(key=lambda m: m["path"])
        emitted[name] = {"name": name, "shape": out_shape(rule, _fuse(kind, [m["shape"] for m in members]))}

    return {
        "model": model,
        "dialect": dialect,
        "layout": layout,
        "tensors": sorted(emitted.values(), key=lambda d: d["name"]),
        "absent_but_tied": sorted(tied.values(), key=lambda d: d["name"]),
    }


def _fuse(kind: str, shapes: Sequence[tuple[int, ...]]) -> tuple[int, ...]:
    """把若干分量张量合成融合形状。"""
    if kind == "qkv":
        return shapes[0][:-1] + (sum(s[-1] for s in shapes),)
    if kind == "w13":
        return (sum(s[0] for s in shapes),) + shapes[0][1:]
    if kind == "gate_up":
        return shapes[0][:-1] + (sum(s[-1] for s in shapes),)
    raise ManifestError(f"未知融合策略 {kind!r}")
