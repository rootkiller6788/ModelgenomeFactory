"""共用夹具。

**测试要一开始就快**，所以分层：

- `tiny`：手工构造的极小 Genome 集合 —— 给「代数性质」这类与 zoo 无关的测试用，毫秒级；
- `result`：**一次**完整流水线（13 个模型 × 3 方言 + 78 个无序对），session 级共享；
- `out_dir` / `loaded`：把产物写进 `tmp_path` 再读回来，于是测试不依赖仓库里 `out/` 的状态
  ——「产物 ≡ 内存」这条断言必须测的是**本次运行**的产物。

`is_minimal` 这种 O(编辑数) 次 apply 的检查只在一条测试里跑一次（全部 78 对），
不拆成 78 个测试 —— 拆开会让每次收集都重算一遍流水线。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from modelgenomefactory import config as config_mod  # noqa: E402
from modelgenomefactory import pipeline  # noqa: E402
from modelgenomefactory.extract import load_zoo  # noqa: E402
from modelgenomefactory.genome import build as build_genome  # noqa: E402
from modelgenomefactory.schema import (  # noqa: E402
    CanonicalIR,
    Edge,
    Genome,
    OpNode,
    RawStructure,
    TensorEntry,
)


@pytest.fixture(scope="session")
def root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def cfg(root: Path) -> dict:
    return config_mod.load_config("configs/default.json", root=root)


@pytest.fixture(scope="session")
def cost_cfg(cfg: dict) -> dict:
    return dict(cfg["cost"])


@pytest.fixture(scope="session")
def zoo(root: Path) -> tuple:
    return load_zoo(root)


@pytest.fixture(scope="session")
def result(cfg: dict, root: Path):
    """**整个会话只跑一次**完整流水线。"""
    return pipeline.run_pipeline(cfg, root=root)


@pytest.fixture(scope="session")
def out_dir(tmp_path_factory, result, cfg: dict) -> Path:
    d = tmp_path_factory.mktemp("out")
    pipeline.write_outputs(result, d, cfg=cfg)
    return d


@pytest.fixture(scope="session")
def loaded(out_dir: Path):
    r = pipeline.load_result(out_dir)
    assert r is not None, "产物写出来却读不回来 —— load_result 与 write_outputs 不同步了"
    return r


# ------------------------------------------------------------------ tiny zoo


def _ir(
    name: str,
    *,
    layers: int = 2,
    hidden: int = 32,
    heads: int = 4,
    kv_heads: int | None = None,
    intermediate: int = 64,
    ffn: str = "swiglu",
    norm: str = "rmsnorm",
    position: str = "rope",
    activation: str = "silu",
    tie: bool = True,
    extra_flag: str | None = None,
    ffn_kind: str = "ffn",
) -> CanonicalIR:
    """手工搭一个极小 IR。**与 zoo 无关**，所以代数测试不依赖 13 个真实模型。

    张量的 `role` 一律走 `dialects.role_of(path)` —— 与 `canonicalize` 同一条路。
    早先这里手写成「路径去掉 `blocks[*].`」，于是 `blocks[*].norm1.weight` 拿到
    `"norm1.weight"`，而 `role_of` 给它 `"norm"`；`rebuild(g, primary(g))` 于是
    算出**不同的指纹**（`role` 进指纹），而 `primary_diff_keys` 是空的 ——
    看起来像「往返丢了东西」，其实是 fixture 自己造了第二份 role 定义。
    """
    from modelgenomefactory.dialects import role_of

    kv = heads if kv_heads is None else kv_heads
    hd = hidden // heads
    # 变体由头数比决定 —— 与 `canonicalize` 同一条规则。**不能手写死**：
    # 手写死会让 `gqa` 变体拿到 `mha` 的骨架指纹，于是「MHA→GQA 是突变」这条
    # 真实结论在小模型上复现不出来（这个坑是我自己在 fixture 里踩的）。
    attn_variant = "mha" if kv == heads else "gqa"
    nodes = [
        OpNode("embed", "embedding", "token", (), (("vocab", 128), ("hidden", hidden))),
        OpNode("position", "position", position, (), (("theta", 10000),)),
        OpNode(
            "blocks[*].attn",
            "attention",
            attn_variant,
            ("sliding_window",) if extra_flag == "sliding_window" else (),
            (("n_heads", heads), ("n_kv_heads", kv), ("head_dim", hd)),
        ),
        OpNode(f"blocks[*].ffn", ffn_kind, ffn, (), (("intermediate", intermediate),)),
        OpNode("blocks[*].norm1", "norm", norm, (), (("eps", 1000000),)),
        OpNode("blocks[*].norm2", "norm", norm, (), (("eps", 1000000),)),
        OpNode("final_norm", "norm", norm, (), (("eps", 1000000),)),
        OpNode("head", "head", "causal_lm", (), (("vocab", 128), ("hidden", hidden))),
    ]
    tensors = [
        TensorEntry(p, role_of(p), sh, tied_with=tw)
        for p, sh, tw in (
            ("embed.weight", (128, hidden), None),
            ("head.weight", (128, hidden), "embed.weight" if tie else None),
            ("final_norm.weight", (hidden,), None),
            ("blocks[*].attn.q_proj.weight", (heads * hd, hidden), None),
            ("blocks[*].attn.k_proj.weight", (kv * hd, hidden), None),
            ("blocks[*].attn.v_proj.weight", (kv * hd, hidden), None),
            ("blocks[*].attn.o_proj.weight", (hidden, heads * hd), None),
            ("blocks[*].ffn.gate_proj.weight", (hidden, intermediate), None),
            ("blocks[*].ffn.up_proj.weight", (hidden, intermediate), None),
            ("blocks[*].ffn.down_proj.weight", (intermediate, hidden), None),
            ("blocks[*].norm1.weight", (hidden,), None),
            ("blocks[*].norm2.weight", (hidden,), None),
        )
    ]
    edges = [
        Edge("embed", "blocks[*].attn", "hidden"),
        Edge("blocks[*].attn", "blocks[*].ffn", "hidden"),
        Edge("blocks[*].ffn", "final_norm", "hidden"),
        Edge("final_norm", "head", "hidden"),
    ]
    topo = (
        ("n_layers", layers),
        ("hidden", hidden),
        ("vocab", 128),
        ("max_positions", 512),
        ("block_groups", ((0, layers),)),
        ("block_group_shape", ("all",)),
        ("activation", activation),
    )
    return CanonicalIR(
        name=name,
        topology=topo,
        nodes=tuple(nodes),
        tensors=tuple(tensors),
        edges=tuple(edges),
        identity=(("source_model_type", "tiny"),),
        dialect="hf",
    )


def _facts(**kw: object) -> RawStructure:
    """给 tiny IR 配一份**自洽的** config 事实表。

    必须自洽，因为 `build` 会同时算两条路：`params_from_schema`（从张量表）与
    `params_from_formula`（从 config 标量）。对不上就会在 tiny 上复现真实项目里
    那种「公式写错了」的失败 —— 而这正是我们想要的：小模型也要走同一条互校。
    """
    from modelgenomefactory.schema import ConfigFact

    rows = [(k, v) for k, v in kw.items() if v is not None]
    return RawStructure(
        name=str(kw["name"]),
        facts=tuple(ConfigFact(key=k, value=v, source_key=k, rule="test:fixture") for k, v in rows),
    )


def _genome(ir: CanonicalIR, *, seq_len: int = 128, batch: int = 1, dtype_bytes: int = 2) -> Genome:
    facts = {k: v for k, v in ir.topology}
    for n in ir.nodes:
        for k, v in n.dims:
            facts.setdefault(k, v)
    # 范数类型与位置编码类型**不是 dims**，而是节点的变体 —— 它们只能从 config 推，
    # 所以这里显式补进事实表（真实项目里由 `dialects.parse_config` 填）。
    facts["name"] = ir.name
    facts["norm_kind"] = next(n.variant for n in ir.nodes if n.kind == "norm")
    facts["position_kind"] = next(n.variant for n in ir.nodes if n.kind == "position")
    facts["n_experts"] = 0
    facts["n_shared_experts"] = 0
    facts["tie_embeddings"] = any(t.tied_with for t in ir.tensors)
    return build_genome(ir, _facts(**facts), seq_len=seq_len, batch=batch, dtype_bytes=dtype_bytes)


def with_facts(raw: RawStructure, **overrides: object) -> RawStructure:
    """复制一份 RawStructure，把指定事实换成探针值（来源标成 `test:probe`）。

    两个测试模块都要用它（方言探针 / 互校探针），所以放在 conftest 里 ——
    测试模块之间**不能**互相 import：`tests/` 没有 `__init__.py`，
    相对导入会直接报 `ImportError`。
    """
    from modelgenomefactory.schema import ConfigFact

    by = {f.key: (f.value, f.source_key, f.rule) for f in raw.facts}
    for k, v in overrides.items():
        by[k] = (v, "", "test:probe")
    return RawStructure(
        name=raw.name,
        facts=tuple(
            ConfigFact(key=k, value=v, source_key=s, rule=r) for k, (v, s, r) in sorted(by.items())
        ),
        unknown_keys=raw.unknown_keys,
        ignored_keys=raw.ignored_keys,
    )


@pytest.fixture(scope="session")
def facts_of():
    """`facts_of(raw, **overrides) -> RawStructure` —— 翻动某个 config 事实的探针入口。

    以夹具而不是模块函数的形式暴露：`tests/` 没有 `__init__.py`，测试模块之间
    互相 `import` 会踩 `ImportError`；而 `import conftest` 又依赖 pytest 的
    `prepend` 导入模式，换个 `importmode` 就散架。
    """
    return with_facts


@pytest.fixture(scope="session")
def tiny_of():
    """`(name="base", *, seq_len, batch, dtype_bytes, **ir_kw) -> Genome`。

    给「换一个代价口径重算」这类测试用 —— `tiny` 夹具的口径是固定的 128/1/2，
    而 FLOPs 与 KV cache 的定义域就是那个口径。
    """

    def _make(
        name: str = "base",
        *,
        seq_len: int = 128,
        batch: int = 1,
        dtype_bytes: int = 2,
        **ir_kw: object,
    ) -> Genome:
        return _genome(_ir(name, **ir_kw), seq_len=seq_len, batch=batch, dtype_bytes=dtype_bytes)

    return _make


@pytest.fixture(scope="session")
def tiny() -> dict[str, Genome]:
    """一个基线 + 七个「只改一处」的变体，覆盖**全部 14 类编辑**：

    - **只有规模**：`wide`（宽度与 FFN 都变）—— 骨架指纹必须**不变**；
    - **换算子**：`moe`（MLP→MoE，走 `variant`）、`layernorm`（RMSNorm→LayerNorm）；
    - **换算子种类**：`op_swap`（`blocks[*].ffn` 的 `kind` 由 `ffn` 改成 `expert`）；
    - **加标志**：`window`（加滑窗）；
    - **拓扑/张量**：`untied`（拆掉权重绑定）、`gqa`（MHA→GQA，改了张量契约）。

    基线刻意用 `tie=True` —— 这样「拆绑定」才是一个真实存在的变体，
    否则基线与 `untied` 会是同一份结构（这个坑我踩过：两个变体指纹一模一样）。

    **`op_swap` 是刻意合成的，不对应任何真实模型。** 它唯一的职责是把
    `op`（算子种类变了）这一条分支走通：13 个真实模型里没有任何一对在**同一个
    路径**上放了**不同种类**的算子（MoE 是 `ffn` 的 `variant`，不是换 `kind`），
    所以 `op` 在 zoo 上是零覆盖的 —— 而「零覆盖的分支」正是这个项目反复在抓的
    那种东西（同 `dialects.CONDITIONAL_RULES`、`ignored_keys` 的纪律）。

    diff 代数与量变/突变判据的单测全跑在这个集合上，**不依赖 zoo 的规模**。
    """
    return {
        "base": _genome(_ir("base")),
        "wide": _genome(_ir("wide", hidden=64, intermediate=128)),
        "moe": _genome(_ir("moe", ffn="moe")),
        "window": _genome(_ir("window", extra_flag="sliding_window")),
        "untied": _genome(_ir("untied", tie=False)),
        "layernorm": _genome(_ir("layernorm", norm="layernorm")),
        "gqa": _genome(_ir("gqa", kv_heads=2)),
        "op_swap": _genome(_ir("op_swap", ffn_kind="expert")),
    }
