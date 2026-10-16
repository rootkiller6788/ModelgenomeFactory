"""ModelGenomeFactory 的数据模型。

本项目的对象是**结构**，不是张量、不是权重、不是训练过程。整条链路上有四个层次的
对象，每一层都有明确的边界：

```
config.json ─┐
             ├─► RawStructure ─► CanonicalIR ─► Genome ─► Diff / Boundary
manifest ────┘   （键名归一）    （结构归一）   （指纹化）  （两两比较）
```

四个关键设计决定（都不是随意的，见 README 附录 C）：

1. **Genome 是「模式」而不是「展开的实例」**：32 层模型不会展开成 32 份节点，而是
   `blocks[*]` 一份模式 + `topology.block_groups = ((0, 32),)` 一个计数。于是
   「层数 32→40」是一次 `dim` 编辑，而不是 8 份节点增删。
2. **指纹分两级**：`shape_fingerprint` 只吃**定性骨架**（算子种类 / 变体 / 标志 /
   张量角色 / 连接），`fingerprint` 再吃**全部规模数值**。于是
   「层数变」与「算子变」在指纹层面就是可判定的两件事，不需要拍阈值。
3. **每个事实都带来源**：`manifest`（张量清单）/ `config`（配置）/ `both`（两者
   都有，必须互校）。范数类型、激活函数、位置编码类型只存在于 config 里 ——
   张量清单**不足以**确定全部结构，这一点必须显式建模，否则就是静默猜测。
4. **主地址空间与派生量分开**：`primary()` 是 diff / apply 作用的空间；
   `cost` 是它的函数，不进地址空间，但必须能被重新算出来（有测试钉住）。

`primary()` 的键是全项目的公共契约 —— `diff.py` 只认识它，`cli.py` 的
`diff --all` 也只打印它。**新增 Genome 字段时必须同时把它接进 `primary()`**，
否则 `apply(a, diff(a,b)) ≡ b` 会静默失效；`tests/test_genome.py` 有一条
「字段全覆盖」的测试专门盯这件事。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

__all__ = [
    "KINDS",
    "VARIANTS",
    "FLAGS",
    "PROVENANCES",
    "SCALE_EDIT_KINDS",
    "MUTATION_EDIT_KINDS",
    "MUTATION_COORDINATES",
    "SKELETON_TOPOLOGY",
    "DERIVED_GENOME_FIELDS",
    "TUPLE_ADDRS",
    "TUPLE_ADDR_SUFFIXES",
    "rehydrate",
    "ConfigFact",
    "RawStructure",
    "TensorEntry",
    "OpNode",
    "Edge",
    "CanonicalIR",
    "Cost",
    "Genome",
    "Edit",
    "Diff",
    "Boundary",
    "FamilyTimeline",
    "Version",
    "utcnow",
]

# ---------------------------------------------------------------- 词表（封闭集合）

#: 规范算子种类。**新增种类必须同时给 dialects.py 加规则**，否则解析器会拒收。
KINDS = (
    "embedding",
    "position",
    "attention",
    "ffn",
    "norm",
    "router",
    "expert",
    "shared_expert",
    "head",
)

#: 每个种类的**变体**。变体是「定性」的：它变了就是结构突变，不是规模变化。
#:
#: 注意 embedding / head 的变体里**没有** tied / untied —— 参数共享是**张量级**事实
#: （`tensors.head.weight.tied_with`），不是算子变体。把它放两处会让「拆掉权重绑定」
#: 这一个动作产生两条编辑，报告里读起来像改了两件事。
VARIANTS = {
    "embedding": ("token",),
    "position": ("rope", "learned"),
    "attention": ("mha", "gqa", "mla"),
    "ffn": ("mlp", "swiglu", "moe"),
    "norm": ("rmsnorm", "layernorm"),
    "router": ("softmax", "sigmoid"),
    "expert": ("swiglu", "mlp"),
    "shared_expert": ("swiglu", "mlp"),
    "head": ("causal_lm",),
}

#: 定性标志（进骨架）。每一条都必须能被 `flags_of` 从 config/manifest 里推出来。
#:
#: **注意这里没有 fused_qkv / fused_gate_up**：融合是**源清单的布局属性**，不是模型的
#: 结构属性。把它放进 flags 会让同一份结构在 hf 与 vllm 两种方言下得到不同的指纹 ——
#: 而那正好是这个项目要证伪的事情。融合只作为 IR 的 `notes` 记录下来。
FLAGS = (
    "sliding_window",       # 注意力带滑窗掩码
    "attention_bias",       # qkv/o 有偏置
    "mlp_bias",             # ffn 有偏置
    "qk_norm",              # 注意力前有 q/k 归一
    "rope_scaling",         # 位置编码带长上下文缩放（YaRN / LongRoPE）
    "shared_experts",       # MoE 带共享专家分支
    "grouped_routing",      # MoE 分组路由（n_group > 1）
)

#: 事实来源。
PROVENANCES = ("manifest", "config", "both")

#: 只改「规模」的编辑种类 → 量变。
SCALE_EDIT_KINDS = ("attr", "dim", "shape")

#: 增删类编辑的互逆映射。**取反不只是交换 `old` / `new`，种类也要翻过来。**
#: 少了这一步，`flag_del` 取反之后还是 `flag_del`，`apply` 会去删一个本来就不在的
#: 标志位 —— 补丁看上去条条齐全，实际上少恢复了一处。`flag_add` / `flag_del`
#: 尤其容易漏，因为它们的 `old` / `new` 里有一边是 `None`，交换之后「看着挺对」。
EDIT_INVERSE: dict[str, str] = {
    "node_add": "node_del",
    "node_del": "node_add",
    "tensor_add": "tensor_del",
    "tensor_del": "tensor_add",
    "edge_add": "edge_del",
    "edge_del": "edge_add",
    "flag_add": "flag_del",
    "flag_del": "flag_add",
}

#: 改动结构的编辑种类 → 突变。
MUTATION_EDIT_KINDS = (
    "op",          # 算子种类变了
    "variant",     # 算子变体变了
    "flag_add",
    "flag_del",
    "node_add",
    "node_del",
    "tensor_add",
    "tensor_del",
    "tie",
    "edge_add",
    "edge_del",
)

#: 突变的四类坐标。报告里按这个分类给「突变坐标表」。
MUTATION_COORDINATES = {
    "op": "operator",
    "node_add": "operator",
    "node_del": "operator",
    "variant": "variant",
    "flag_add": "variant",
    "flag_del": "variant",
    "tie": "topology",
    "edge_add": "topology",
    "edge_del": "topology",
    "tensor_add": "tensor_schema",
    "tensor_del": "tensor_schema",
}

#: 进**骨架**（因而不含规模）的 topology 键。
#:
#: 判据是**「这个键的值是定性的吗」**，而不是「它看着像不像结构」：
#:
#: - `block_group_shape` 是 `block_groups` 的**去规模版本**：`("all",)` 表示整个堆叠同构，
#:   `("one","many")` 表示「首块稠密、其余 MoE」。带计数的 `block_groups` 只进全指纹 ——
#:   于是「1+26 块」与「1+3 块」是量变，「整堆同构」与「首块例外」才是突变。
#: - `activation` 是字符串（`silu` / `gelu_new`）。它不进骨架的话，
#:   「GELU→SiLU」会被 `dim`/`attr` 归成量变，而它显然是换了个非线性 —— 两套判据会打架。
#:
#: 其余 topology 键（层数 / 宽度 / 词表 / 块组计数 / 最大长度）全是数值，只进全指纹。
#: **凡是重复表达同一件事的键都不放进来**（`tie_embeddings` 由张量的 `tied_with` 表达、
#: `norm_kind` 由范数节点的变体表达、`n_experts` 由 router 节点的 dims 表达）——
#: 冗余事实会让「拆掉权重绑定」这一个动作产生两条编辑，报告读起来像改了两件事。
SKELETON_TOPOLOGY = (
    "block_group_shape",
    "activation",
)

#: **不进**主地址空间的 Genome 字段。新增字段时必须在这里或 `primary()` 里二选一，
#: 否则 `tests/test_genome.py::test_primary_covers_every_field` 会失败。
DERIVED_GENOME_FIELDS = ("name", "identity", "dialect", "cost", "notes")


def utcnow() -> str:
    """产物里的时间戳。**不参与任何指纹**（否则同一份配置跑两次指纹不同）。"""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat(timespec="seconds")


#: 主地址空间里那些**规范值是元组**的地址。JSON 存不下元组，读回来会变成列表 ——
#: 而 `(4096, 4096) != [4096, 4096]`，于是「从产物载入的」与「内存里算出的」
#: 会**看起来一模一样、比不相等**：`same_primary` 说不同、`apply` 还原不出 `b`。
#:
#: 这两张表是**唯一**允许出现元组值的地址，所以 rehydrate 只需要认识它们。
#: 新增元组值地址时必须同步加进来 —— `tests/test_genome.py` 有一条
#: 「产物载入 ≡ 内存构建（逐位）」的测试专门盯这件事。
TUPLE_ADDRS = ("topology.block_groups", "topology.block_group_shape")
TUPLE_ADDR_SUFFIXES = (".shape", ".flags")


def rehydrate(addr: str, v: Any) -> Any:
    """把 JSON 读回来的列表还原成元组。**只对上面那几类地址生效。**

    不做全局递归转换：`topology` 里还有别的值是**列表语义**的（比如
    `block_groups` 之外的东西），一刀切会把它们也改成元组，
    于是「载入的」与「构建的」反过来又不相等了。
    """
    if v is None or not isinstance(v, list):
        return v
    if addr in TUPLE_ADDRS or addr.endswith(TUPLE_ADDR_SUFFIXES):
        return tuple(tuple(x) if isinstance(x, list) else x for x in v)
    return v


# ---------------------------------------------------------------- 抽取层


@dataclass(frozen=True)
class ConfigFact:
    """一条从 config.json 归一化出来的事实。

    `source_key` 是原键名（可能是 `n_layer` 这种另一套命名），`rule` 是命中的规则名 ——
    两者都留着，`explain` 才能回答「这个数字是从哪来的」。
    """

    key: str            # 规范键名
    value: Any
    source_key: str
    rule: str

    def to_dict(self) -> dict[str, Any]:
        return {"key": self.key, "value": self.value, "source_key": self.source_key, "rule": self.rule}


@dataclass(frozen=True)
class RawStructure:
    """config 的归一化视图。**还没看过张量清单。**

    `unknown_keys` 与 `ignored_keys` 是**两件事**，不能合并：

    - `unknown_keys` —— 我们**读不懂**的键。它必须是空的（有测试钉住），
      因为「读不懂但假装读懂了」是这个项目最危险的失败模式；
    - `ignored_keys` —— 我们**有理由不看**的键，每一条带一个理由（dropout /
      transformers 版本 / 分词器 id …）。它不该为空，也不该被当成问题。

    合并成「覆盖率 100%」会把一个真问题和一个设计决定说成同一件事。
    """

    name: str
    facts: tuple[ConfigFact, ...]
    unknown_keys: tuple[str, ...] = ()
    ignored_keys: tuple[tuple[str, str], ...] = ()

    def get(self, key: str, default: Any = None) -> Any:
        for f in self.facts:
            if f.key == key:
                return f.value
        return default

    def as_map(self) -> dict[str, Any]:
        """`{规范键: 值}`。**不要写 `dict(raw.facts)`** —— facts 是 `ConfigFact` 对象，
        那样会得到 `{ConfigFact: ...}` 而不是 `{key: value}`，而它在多数调用点
        只在第一次 `.get()` 时才炸。"""
        return {f.key: f.value for f in self.facts}

    def require(self, key: str) -> Any:
        for f in self.facts:
            if f.key == key:
                return f.value
        raise KeyError(f"{self.name}: config 缺少必需键 {key!r}（已归一化的键：{sorted(f.key for f in self.facts)}）")

    def has(self, key: str) -> bool:
        return any(f.key == key for f in self.facts)

    def keys(self) -> tuple[str, ...]:
        return tuple(f.key for f in self.facts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "facts": [f.to_dict() for f in self.facts],
            "unknown_keys": list(self.unknown_keys),
            "ignored_keys": [{"key": k, "reason": r} for k, r in self.ignored_keys],
        }


# ---------------------------------------------------------------- 规范层


@dataclass(frozen=True)
class TensorEntry:
    """一张张量的规范条目。

    `path` 是**规范路径**（含 `[*]` / `[0]` 这类模式），`source_name` 是它在原始清单
    里的名字 —— 后者只用于审计，**绝不进指纹**（否则换个命名风格指纹就变了）。
    """

    path: str
    role: str
    shape: tuple[int, ...]
    tied_with: str | None = None
    provenance: str = "manifest"
    source_name: str = ""

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "role": self.role,
            "shape": list(self.shape),
            "tied_with": self.tied_with,
            "provenance": self.provenance,
            "source_name": self.source_name,
        }


@dataclass(frozen=True)
class OpNode:
    """计算图上的一个算子节点。

    `dims` 是**规模数值**（只进全指纹），`flags` 是**定性标志**（进骨架）。
    这个二分是本项目判定「量变 / 突变」的唯一依据。
    """

    path: str
    kind: str
    variant: str
    flags: tuple[str, ...] = ()
    dims: tuple[tuple[str, int], ...] = ()
    provenance: str = "both"

    def dim(self, key: str, default: int | None = None) -> int | None:
        for k, v in self.dims:
            if k == key:
                return v
        return default

    def has_flag(self, flag: str) -> bool:
        return flag in self.flags

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind,
            "variant": self.variant,
            "flags": list(self.flags),
            "dims": {k: v for k, v in self.dims},
            "provenance": self.provenance,
        }


@dataclass(frozen=True)
class Edge:
    """计算图上的一条边。`label` 是流过的张量角色。"""

    src: str
    dst: str
    label: str

    def to_dict(self) -> dict[str, Any]:
        return {"src": self.src, "dst": self.dst, "label": self.label}


@dataclass(frozen=True)
class CanonicalIR:
    """Canonical Model IR：与命名无关的结构视图。

    这是 Canonicalizer 的产物，也是整个项目的**中枢**：Genome 由它生成，
    Diff 作用在它派生出的地址空间上，`tree` / `graph` 两个子命令直接打印它。

    `crosschecks` 是「两路互校」的**可见证据**：每一项是
    `(事实名, config 说的, 清单量出的)`。它是报告里最值得看的一张表 ——
    「读不懂但假装读懂了」是这个项目最危险的失败模式，把两条独立路径摆在一起
    是唯一能防住它的办法。
    """

    name: str
    topology: tuple[tuple[str, Any], ...]
    nodes: tuple[OpNode, ...]
    tensors: tuple[TensorEntry, ...]
    edges: tuple[Edge, ...]
    identity: tuple[tuple[str, Any], ...] = ()
    dialect: str = ""
    notes: tuple[str, ...] = ()
    crosschecks: tuple[tuple[str, str, str], ...] = ()

    def topo(self, key: str, default: Any = None) -> Any:
        for k, v in self.topology:
            if k == key:
                return v
        return default

    def node(self, path: str) -> OpNode:
        for n in self.nodes:
            if n.path == path:
                return n
        raise KeyError(f"{self.name}: 没有节点 {path!r}")

    def tensor(self, path: str) -> TensorEntry:
        for t in self.tensors:
            if t.path == path:
                return t
        raise KeyError(f"{self.name}: 没有张量 {path!r}")

    def role_names(self) -> tuple[str, ...]:
        return tuple(sorted({t.role for t in self.tensors}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dialect": self.dialect,
            "identity": {k: v for k, v in self.identity},
            "topology": {k: v for k, v in self.topology},
            "nodes": [n.to_dict() for n in self.nodes],
            "tensors": [t.to_dict() for t in self.tensors],
            "edges": [e.to_dict() for e in self.edges],
            "notes": list(self.notes),
            "crosschecks": [list(c) for c in self.crosschecks],
        }


# ---------------------------------------------------------------- 指纹层


@dataclass(frozen=True)
class Cost:
    """结构代价。**是主地址空间的函数，不是独立事实。**

    三路互校（见 `cost.py`）：张量表求和 / 解析公式 / 公开参数量。

    `seq_len` / `batch` / `dtype_bytes` 是代价的**口径**，必须一起存 ——
    「FLOPs 是多少」在没有「@seq=2048」的情况下没有意义，而且
    `diff.apply` 重建 Genome 时要用同一套口径复算，否则代价对不上。
    """

    params: int
    params_formula: int
    flops_per_token: int
    kv_cache_bytes: int
    weight_bytes: int
    breakdown: tuple[tuple[str, int], ...] = ()
    params_published_b: float | None = None
    seq_len: int = 2048
    batch: int = 1
    dtype_bytes: int = 2

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "params_formula": self.params_formula,
            "params_published_b": self.params_published_b,
            "flops_per_token": self.flops_per_token,
            "kv_cache_bytes": self.kv_cache_bytes,
            "weight_bytes": self.weight_bytes,
            "breakdown": {k: v for k, v in self.breakdown},
            "seq_len": self.seq_len,
            "batch": self.batch,
            "dtype_bytes": self.dtype_bytes,
        }


@dataclass(frozen=True)
class Genome:
    """结构基因。`fingerprint` 与 `shape_fingerprint` 由 `genome.py` 计算。"""

    name: str
    topology: tuple[tuple[str, Any], ...]
    nodes: tuple[OpNode, ...]
    tensors: tuple[TensorEntry, ...]
    edges: tuple[Edge, ...]
    cost: Cost
    identity: tuple[tuple[str, Any], ...] = ()
    dialect: str = "hf"
    notes: tuple[str, ...] = ()

    # -- 主地址空间 ---------------------------------------------------------

    def primary(self) -> dict[str, Any]:
        """diff / apply 作用的**唯一**空间。

        键的形状：
        - `topology.<k>`
        - `nodes.<path>.kind` / `.variant` / `.flags` / `.dims.<k>`
        - `tensors.<path>.shape` / `.tied_with`
        - `edges.<src>-><dst>:<label>`

        **不含** `name` / `identity` / `dialect`（标签）、`cost`（派生量），
        也**不含 `tensors.<path>.role`** —— 角色是路径的纯函数，放进来只会让每条
        张量编辑多出一份冗余。凡是「由别的事实唯一决定」的东西都不进这里。
        """
        out: dict[str, Any] = {}
        for k, v in self.topology:
            out[f"topology.{k}"] = v
        for n in self.nodes:
            out[f"nodes.{n.path}.kind"] = n.kind
            out[f"nodes.{n.path}.variant"] = n.variant
            out[f"nodes.{n.path}.flags"] = tuple(n.flags)
            for k, v in n.dims:
                out[f"nodes.{n.path}.dims.{k}"] = v
        for t in self.tensors:
            out[f"tensors.{t.path}.shape"] = tuple(t.shape)
            out[f"tensors.{t.path}.tied_with"] = t.tied_with
        for e in self.edges:
            out[f"edges.{e.src}->{e.dst}:{e.label}"] = True
        return out

    # -- 便捷访问 -----------------------------------------------------------

    def topo(self, key: str, default: Any = None) -> Any:
        for k, v in self.topology:
            if k == key:
                return v
        return default

    def node(self, path: str) -> OpNode:
        for n in self.nodes:
            if n.path == path:
                return n
        raise KeyError(f"{self.name}: 没有节点 {path!r}")

    def tensor(self, path: str) -> TensorEntry:
        for t in self.tensors:
            if t.path == path:
                return t
        raise KeyError(f"{self.name}: 没有张量 {path!r}")

    def identity_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.identity}

    def role_names(self) -> tuple[str, ...]:
        """出现过的张量角色。与 `CanonicalIR.role_names` 同名同义 —— 两个层次的
        对象共用一套访问器，`cli` 与 `report` 才不需要按类型分叉。"""
        return tuple(sorted({t.role for t in self.tensors}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dialect": self.dialect,
            "identity": {k: v for k, v in self.identity},
            "topology": {k: v for k, v in self.topology},
            "nodes": [n.to_dict() for n in self.nodes],
            "tensors": [t.to_dict() for t in self.tensors],
            "edges": [e.to_dict() for e in self.edges],
            "cost": self.cost.to_dict(),
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------- 比较层


@dataclass(frozen=True)
class Edit:
    """一条最小结构编辑。

    `addr` 是主地址空间里的键；`old` / `new` 是它的值。**只记真正变了的地址** ——
    于是「最小性」是构造出来的性质，但仍然有测试逐条验证（见 `test_diff.py`）。
    """

    kind: str
    addr: str
    old: Any
    new: Any

    @property
    def is_scale(self) -> bool:
        return self.kind in SCALE_EDIT_KINDS

    @property
    def coordinate(self) -> str | None:
        return MUTATION_COORDINATES.get(self.kind)

    def inverted(self) -> "Edit":
        return Edit(EDIT_INVERSE.get(self.kind, self.kind), self.addr, self.new, self.old)

    def render(self) -> str:
        return f"{self.kind:10s} {self.addr} : {_short(self.old)} → {_short(self.new)}"

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "addr": self.addr, "old": self.old, "new": self.new}


def _short(v: Any, limit: int = 40) -> str:
    s = repr(v)
    return s if len(s) <= limit else s[: limit - 1] + "…"


@dataclass(frozen=True)
class Diff:
    """两个 Genome 之间的结构差异。**是一个可以施加回去的补丁。**"""

    left: str
    right: str
    edits: tuple[Edit, ...]
    left_fingerprint: str
    right_fingerprint: str
    left_shape_fingerprint: str
    right_shape_fingerprint: str

    @property
    def is_empty(self) -> bool:
        return not self.edits

    @property
    def is_scale_only(self) -> bool:
        return all(e.is_scale for e in self.edits)

    @property
    def is_mutation(self) -> bool:
        """骨架变了吗 —— 这是**唯一**的突变判据，不看编辑条数。"""
        return self.left_shape_fingerprint != self.right_shape_fingerprint

    @property
    def coordinates(self) -> tuple[str, ...]:
        coords = sorted({e.coordinate for e in self.edits if e.coordinate})
        return tuple(coords)

    def scale_edits(self) -> tuple[Edit, ...]:
        return tuple(e for e in self.edits if e.is_scale)

    def structural_edits(self) -> tuple[Edit, ...]:
        return tuple(e for e in self.edits if not e.is_scale)

    def inverted(self) -> "Diff":
        """取反。**保持顺序，不 reverse** —— `diff` 的输出按「类别 + 地址」排序，
        反方向算出来的序列也是同一个顺序，两者必须逐位相同。早先这里 `reversed()` 了，
        于是 `invert(diff(a,b))` 与 `diff(b,a)` 只在**碰巧回文**的时候相等。
        """
        return Diff(
            left=self.right,
            right=self.left,
            edits=tuple(e.inverted() for e in self.edits),
            left_fingerprint=self.right_fingerprint,
            right_fingerprint=self.left_fingerprint,
            left_shape_fingerprint=self.right_shape_fingerprint,
            right_shape_fingerprint=self.left_shape_fingerprint,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "left": self.left,
            "right": self.right,
            "n_edits": len(self.edits),
            "is_mutation": self.is_mutation,
            "coordinates": list(self.coordinates),
            "left_fingerprint": self.left_fingerprint,
            "right_fingerprint": self.right_fingerprint,
            "left_shape_fingerprint": self.left_shape_fingerprint,
            "right_shape_fingerprint": self.right_shape_fingerprint,
            "edits": [e.to_dict() for e in self.edits],
        }


@dataclass(frozen=True)
class Boundary:
    """家族演化链上的一个**版本边界**（相邻两个版本之间）。"""

    family: str
    left: str
    right: str
    index: int                 # 1-based，链上的第几个边界
    kind: str                  # "same" | "scale" | "mutation"
    coordinates: tuple[str, ...]
    n_edits: int
    n_scale: int
    n_structural: int
    fingerprint_changed: bool
    shape_fingerprint_changed: bool
    headline: tuple[str, ...] = ()

    @property
    def is_mutation_point(self) -> bool:
        return self.shape_fingerprint_changed

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "left": self.left,
            "right": self.right,
            "index": self.index,
            "kind": self.kind,
            "coordinates": list(self.coordinates),
            "n_edits": self.n_edits,
            "n_scale": self.n_scale,
            "n_structural": self.n_structural,
            "fingerprint_changed": self.fingerprint_changed,
            "shape_fingerprint_changed": self.shape_fingerprint_changed,
            "headline": list(self.headline),
        }


@dataclass(frozen=True)
class FamilyTimeline:
    """一个家族的版本演化时间线。"""

    family: str
    members: tuple[str, ...]
    boundaries: tuple[Boundary, ...]
    note: str = ""

    @property
    def mutation_points(self) -> tuple[Boundary, ...]:
        return tuple(b for b in self.boundaries if b.is_mutation_point)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "members": list(self.members),
            "n_boundaries": len(self.boundaries),
            "mutation_points": [b.index for b in self.mutation_points],
            "boundaries": [b.to_dict() for b in self.boundaries],
            "note": self.note,
        }


@dataclass(frozen=True)
class Version:
    id: str
    digest: str
    created_at: str
    protocol_digest: str
    counts: tuple[tuple[str, int], ...] = ()
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "digest": self.digest,
            "created_at": self.created_at,
            "protocol_digest": self.protocol_digest,
            "counts": {k: v for k, v in self.counts},
            "note": self.note,
        }
