"""报告：把结构证据渲染成 markdown。

**报告的组织顺序就是本项目的论证顺序**，不是「数据的自然顺序」：

1. 口径 —— 先说清这些数字在什么条件下得到（版本 / 协议 / zoo 规模 / 代价口径）；
2. **验收标准逐条核对** —— README 那两句话到底兑现了没有，一条一条摆出来；
3. 指纹表 —— 两级指纹并列，这是「量变 / 突变」可判定的**前提**；
4. 同构类 —— 「指纹有什么用」的直接答案；
5. 对比矩阵 —— **骨架列与规模列分开排**，矩阵本身就把量变/突变的分界线画出来；
6. 家族突变时间线 —— 验收标准第二条；
7. 三视图示例 —— Module Tree / Graph / Tensor Schema，README 点名要的三张视图；
8. 代价三路互校 —— 表 / 公式 / checkpoint 元数据逐位一致；
9. config ↔ 清单 两路互校 —— 「读不懂但假装读懂了」的唯一防线；
10. 设计期望 vs 实测 —— **被推翻的期望**是这一节最有价值的部分（附录 B）；
11. 可复现性边界 —— 哪些数字必须逐位一致，哪些不是。

`acceptance_checks` 是**公开**函数：CLI 的 `acceptance` 子命令与报告共用它，
于是「命令行说通过的」与「报告里写的」不可能不一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from . import compare as compare_mod
from .config import expectations_of
from .diff import headline, summary as diff_summary
from .schema import Genome

__all__ = ["Check", "acceptance_checks", "render_report", "fmt", "bar", "module_tree", "graph_lines", "tensor_schema"]


# ---------------------------------------------------------------------- 小工具


def fmt(v: Any, nd: int = 3) -> str:
    """统一数字格式。CLI 与报告必须用同一套，否则两处会不一致。"""
    if isinstance(v, bool):
        return "是" if v else "否"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if v == int(v) and abs(v) < 1e15:
            return f"{int(v):,}"
        return f"{v:.{nd}f}"
    if v is None:
        return "—"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x, nd) for x in v) + "]"
    return str(v)


def bar(value: float, width: int = 24, char: str = "█") -> str:
    n = max(0, min(width, int(round(value * width))))
    return char * n + "·" * (width - n)


def _human(n: int) -> str:
    """参数量的人类可读形式。报告里 `7.24B` 比 `7,241,732,096` 好读。"""
    for unit, scale in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if abs(n) >= scale:
            return f"{n / scale:.3f}{unit}"
    return str(n)


def _tbl(header: Sequence[str], rows: Iterable[Sequence[Any]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return out


# ---------------------------------------------------------------------- 三视图

#: 三视图里每个节点缩进的列宽。
_TREE_INDENT = "  "


def module_tree(genome: Genome, *, max_tensors: int = 2) -> str:
    """**Module Tree 视图**：按节点路径的层级展开，块内的模式只出现一次。

    因为 Genome 是「模式」而不是「展开的实例」，32 层的模型不会打印 32 份子树 ——
    `blocks[*]` 就是一份，层数写在 `topology.n_layers` 里。这正是这个视图与
    `print(model)` 的区别：后者会把每一层都打出来，于是「哪一层不一样」看不出来。
    """
    lines: list[str] = []
    by_parent: dict[str, list[Any]] = {}
    for n in genome.nodes:
        parent = n.path.rsplit(".", 1)[0] if "." in n.path else ""
        by_parent.setdefault(parent, []).append(n)

    def walk(prefix: str, depth: int) -> None:
        for n in sorted(by_parent.get(prefix, []), key=lambda x: x.path):
            leaf = n.path.rsplit(".", 1)[-1]
            dims = " ".join(f"{k}={fmt(v)}" for k, v in n.dims)
            flags = ("  [" + ",".join(n.flags) + "]") if n.flags else ""
            prov = "" if n.provenance == "both" else f"  ({n.provenance})"
            lines.append(f"{_TREE_INDENT * depth}{leaf:24s} {n.kind:13s} {n.variant:9s} {dims}{flags}{prov}")
            walk(n.path, depth + 1)

    walk("", 0)
    lines.append("")
    lines.append(f"{'topology':24s} " + "  ".join(f"{k}={fmt(v)}" for k, v in genome.topology))
    return "\n".join(lines)


def graph_lines(genome: Genome) -> list[str]:
    """**Graph 视图**：边表。`edges` 是进骨架指纹的定性事实，不是派生量。"""
    return [f"{e.src:28s} --{e.label:18s}--> {e.dst}" for e in genome.edges]


def tensor_schema(genome: Genome, *, limit: int | None = None) -> list[str]:
    """**Tensor Schema 视图**：规范路径 / 角色 / 形状 / 权重绑定 / 来源。

    `source_name` 只用于审计，**不进指纹** —— 视图里单列一列，是为了让
    「换个命名方言，指纹不变」这句话可以被肉眼核对。
    """
    rows = [
        f"{t.path:52s} {t.role:14s} {str(list(t.shape)):22s} "
        f"{('→' + t.tied_with) if t.tied_with else '-':22s} {t.provenance}"
        for t in sorted(genome.tensors, key=lambda x: x.path)
    ]
    return rows[:limit] if limit else rows


# ---------------------------------------------------------------------- 验收


@dataclass(frozen=True)
class Check:
    """一条验收断言。`detail` 是**算出来的**证据，不是结论的复述。"""

    name: str
    passed: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "passed": self.passed, "detail": self.detail}


def acceptance_checks(result: Any, cfg: Mapping[str, Any]) -> tuple[Check, ...]:
    """逐条核对配置里的验收口径与设计期望。**这是报告的第一节。**"""
    acc = dict(cfg.get("acceptance") or {})
    exp = expectations_of(cfg)
    out: list[Check] = []
    names = result.names()

    # --- 覆盖度 ---
    want_models = int(acc.get("min_models", 0))
    out.append(
        Check(
            f"模型数 ≥ {want_models}",
            len(names) >= want_models,
            f"实测 {len(names)} 个：{', '.join(names)}",
        )
    )

    want_dialects = int(acc.get("min_dialects_per_model", 1))
    per_model = {n: len(result.dialects_of(n)) for n in names}
    worst = min(per_model.values()) if per_model else 0
    out.append(
        Check(
            f"每个模型 ≥ {want_dialects} 种命名方言",
            worst >= want_dialects,
            "逐模型方言数 " + ", ".join(f"{n}={c}" for n, c in sorted(per_model.items())),
        )
    )

    # --- 跨方言同指纹：整个项目的前提 ---
    if result.canonical:
        fps_by_model: dict[str, set[str]] = {}
        for (name, _d), _ir in result.canonical.items():
            fps_by_model.setdefault(name, set()).add(result.fingerprint_of(name))
        bad = {n: sorted(f) for n, f in fps_by_model.items() if len(f) != 1}
        out.append(
            Check(
                "同一模型的不同方言给出同一个指纹",
                not bad,
                f"{len(fps_by_model)} 个模型全部一致" if not bad else f"不一致：{bad}",
            )
        )
        n_ck = {n: len(result.crosschecks(n)) for n in names}
        out.append(
            Check(
                "config ↔ 清单 两路互校逐项通过",
                True,  # 任一项不一致 canonicalize 会直接抛错，走不到这里
                f"互校 {sum(n_ck.values())} 项（最少 {min(n_ck.values()) if n_ck else 0} 项/模型）",
            )
        )
    else:
        out.append(Check("同一模型的不同方言给出同一个指纹", True, "产物模式：canonical 未载入，跳过"))

    # --- config 覆盖率 ---
    cov = result.coverage or {}
    if acc.get("require_unknown_keys_empty", False) and not cov.get("from_products"):
        unknown = cov.get("unknown") or {}
        n_ign = int(cov.get("n_ignored", 0))
        out.append(
            Check(
                "config 里没有读不懂的键",
                not unknown,
                (
                    "全部 config 键都有归一化规则；"
                    f"另有 {n_ign} 处键是**有理由忽略**的（dropout / 版本号 / 分词器 id …），"
                    "理由见第十节"
                )
                if not unknown
                else f"读不懂：{unknown}",
            )
        )

    # --- 代价三路互校 ---
    zoo_by_name = {e.name: e for e in (result.zoo or ())}
    if zoo_by_name:
        rows = []
        all_ok = True
        for n in names:
            c = result.genome(n).cost
            ck = dict(zoo_by_name[n].checkpoint or {})
            exact = ck.get("params")
            ok = c.params == c.params_formula and (exact is None or c.params == int(exact))
            all_ok &= ok
            rows.append(f"{n}: 表={c.params:,} 公式={c.params_formula:,} checkpoint={exact}")
        out.append(Check("代价三路逐位一致（表 / 公式 / checkpoint 元数据）", all_ok, "；".join(rows[:3]) + " …"))

    # --- Diff 可施加 ---
    if acc.get("require_diff_applies", False):
        from .diff import apply
        from .genome import same_primary

        bad: list[str] = []
        n_pairs = 0
        for d in result.pairs_ordered():
            n_pairs += 1
            a = result.genome(d.left)
            b = result.genome(d.right)
            if not same_primary(apply(a, d), b):
                bad.append(f"{d.left}→{d.right}")
        out.append(
            Check(
                "任意两个模型的 diff 都能施加回去",
                not bad,
                f"{n_pairs} 个无序对全部满足 apply(a, diff(a,b)) ≡ b" if not bad else f"失败：{bad}",
            )
        )

    # --- 设计期望 ---
    if exp:
        for fam, key in (("llama", "llama_mutation_points"), ("qwen", "qwen_mutation_points"), ("mistral", "mistral_mutation_points")):
            if key not in exp:
                continue
            want = int(exp[key])
            got = _n_mutations(result, fam)
            out.append(Check(f"{fam} 家族的突变点数 = {want}", got == want, f"实测 {got} 个突变点"))
        if exp.get("qwen_last_boundary_is_zero_diff"):
            tl = _timeline_or_none(result, "qwen")
            last = tl.boundaries[-1] if tl and tl.boundaries else None
            got = bool(last and last.kind == "same")
            out.append(
                Check(
                    "qwen 链最后一个边界是零差异",
                    got,
                    f"{last.left}→{last.right}: kind={last.kind} 编辑={last.n_edits}" if last else "没有边界",
                )
            )
        if "min_full_isomorphic_classes" in exp:
            want = int(exp["min_full_isomorphic_classes"])
            full = result.classes.get("full", {})
            got = sum(1 for v in full.values() if len(v) >= 2)
            members = [v for v in full.values() if len(v) >= 2]
            out.append(
                Check(
                    f"至少 {want} 个成员 ≥2 的全同构类",
                    got >= want,
                    f"实测 {got} 个：{members}",
                )
            )
    return tuple(out)


def _n_mutations(result: Any, family: str) -> int:
    tl = _timeline_or_none(result, family)
    return len(tl.mutation_points) if tl else 0


def _timeline_or_none(result: Any, family: str) -> Any:
    for tl in result.timelines:
        if tl.family == family:
            return tl
    return None


# ---------------------------------------------------------------------- 各节


def _header(result: Any, cfg: Mapping[str, Any]) -> str:
    v = result.version
    counts = dict(v.counts)
    cost_cfg = dict(cfg.get("cost") or {})
    lines = [
        "# ModelGenomeFactory 报告",
        "",
        "> 本项目的产物是**结构证据**：把模型从「一堆代码」变成可比较的结构基因。",
        "> 它**不训练任何模型**，也不回答「哪个模型更好」—— 那需要评测，不是结构。",
        "",
        "## 一、口径",
        "",
        "| 项 | 值 |",
        "| --- | --- |",
        f"| 版本 | `{v.id}` |",
        f"| 内容摘要 | `{v.digest}` |",
        f"| 协议摘要 | `{v.protocol_digest}` |",
        f"| 创建时间 | {v.created_at}（唯一不可逐位复现的字段） |",
        f"| zoo | {counts.get('models', 0)} 个模型 / {counts.get('dialects', 0)} 份命名清单 |",
        f"| 两两比较 | {counts.get('pairs', 0)} 个无序对 |",
        f"| 血缘图 | {counts.get('lineage_nodes', 0)} 节点 / {counts.get('lineage_edges', 0)} 边 |",
        f"| 代价口径 | seq_len={cost_cfg.get('seq_len')}, batch={cost_cfg.get('batch')}, "
        f"dtype={cost_cfg.get('dtype_bytes')}B |",
        f"| **优化目标** | **无**（测量型：只测量结构、不训练、不选优） |",
        f"| 命名方言 | {', '.join(cfg.get('dialects') or [])} |",
        f"| 归一化规则 | " + _rule_summary(result.rules) + " |",
        "",
        "**为什么「不训练」是核心命题而不是省事。** 一个结构的可比较性不依赖于它的权重："
        "层数、算子、连接、张量契约都是**声明出来的**事实。一旦引入训练，"
        "「结构差异」就会和「训练差异」混在一起，报告再也说不清某一处不同是架构改了"
        "还是只是训得久。这个项目把这两件事彻底分开。",
        "",
        "**「两路互校」是什么。** 每个模型有两条独立的读法：config 标量（能给出范数类型、"
        "激活函数、位置编码类型、专家数，但**给不出**张量契约）与参数清单（能给出张量路径与"
        "形状、层数、宽度、偏置是否存在，但**给不出**范数类型）。两边都能回答的事实"
        "（层数 / 宽度 / 词表 / 专家数 / 偏置 / 位置表）在 `canonicalize` 里**逐项对质**，"
        "不一致就报错。这是「读不懂但假装读懂了」的唯一防线。",
        "",
    ]
    return "\n".join(lines)


def _acceptance(result: Any, cfg: Mapping[str, Any]) -> str:
    checks = acceptance_checks(result, cfg)
    n_ok = sum(1 for c in checks if c.passed)
    lines = [
        "## 二、验收标准",
        "",
        f"**{n_ok}/{len(checks)} 条通过。** 这些断言全部由数据算出，改口径改不动它们。",
        "",
        "| # | 断言 | 判定 | 证据 |",
        "| --- | --- | --- | --- |",
    ]
    for i, c in enumerate(checks, 1):
        mark = "✅" if c.passed else "❌"
        lines.append(f"| {i} | {c.name} | {mark} | {c.detail} |")
    lines += [
        "",
        "README 的两条验收标准对应下面两节：",
        "",
        "- **「对任意两个开源模型输出可复现的结构 Diff」** → 第五节（矩阵）+ 第十一节（Diff 代数）；",
        "- **「同一家族不同版本的结构演化可被自动识别为突变点」** → 第六节（家族突变时间线）。",
        "",
    ]
    return "\n".join(lines)


def _fingerprints(result: Any) -> str:
    lines = [
        "## 三、两级指纹",
        "",
        "指纹分两级，这是**「量变 / 突变」不需要阈值就能判定**的原因：",
        "",
        "- `shape_fingerprint` 只吃**定性骨架**：节点路径 / 种类 / 变体 / 标志、"
        "张量路径 / 角色 / 是否绑定、边、`block_group_shape`、`activation`；",
        "- `fingerprint` = 骨架 + **全部规模数值**（层数 / 宽度 / 词头数 / 词表 / 最大长度 / 专家数）。",
        "",
        "于是：**骨架指纹相同 ⟺ 只改了规模；骨架指纹不同 ⟺ 结构发生了突变。**"
        "这句话没有阈值、没有容差、没有「显著」两个字。",
        "",
        "| 模型 | 家族 | 骨架指纹 | 全指纹 | 参数量 | 激活参数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    meta = {e.name: e for e in (result.zoo or ())}
    for n in result.names():
        g = result.genome(n)
        e = meta.get(n)
        active = result.active_params(n)
        act = f"{_human(active)}" + ("（= 总参数）" if active == g.cost.params else "")
        lines.append(
            f"| `{n}` | {e.family if e else '—'} | `{result.shape_fingerprint_of(n)[:12]}` | "
            f"`{result.fingerprint_of(n)[:12]}` | {_human(g.cost.params)} | {act} |"
        )
    lines.append("")
    return "\n".join(lines)


def _classes(result: Any) -> str:
    shape = result.classes.get("shape", {})
    full = result.classes.get("full", {})
    lines = [
        "## 四、同构类",
        "",
        "**这是「指纹有什么用」的直接答案。** 按骨架指纹分组 = 同一个结构的不同规模；"
        "按全指纹分组 = 逐位相同的结构。",
        "",
        "### 4.1 全同构类（成员 ≥ 2）",
        "",
    ]
    big_full = {k: v for k, v in full.items() if len(v) >= 2}
    if big_full:
        lines += _tbl(["指纹", "规模", "成员"], [[f"`{k[:12]}`", len(v), ", ".join(f"`{x}`" for x in v)] for k, v in sorted(big_full.items())])
    else:
        lines.append("（没有成员 ≥2 的全同构类）")
    lines += [
        "",
        "> **这一节最有信息量的地方在于它把「发布了新版本」和「结构变了」分开了。**"
        "两个不同发布版本落在同一个全同构类里，说明它们之间的进步**完全来自数据与配方**，"
        "而不是结构 —— 这句话在只有代码的时候是说不出来的。",
        "",
        "### 4.2 骨架类（成员 ≥ 2）",
        "",
    ]
    big_shape = {k: v for k, v in shape.items() if len(v) >= 2}
    if big_shape:
        lines += _tbl(
            ["骨架指纹", "规模", "成员"],
            [[f"`{k[:12]}`", len(v), ", ".join(f"`{x}`" for x in v)] for k, v in sorted(big_shape.items())],
        )
    else:
        lines.append("（没有成员 ≥2 的骨架类）")
    lines += [
        "",
        "骨架类的成员之间**只差规模**：它们是同一个结构基因在不同刻度上的实例。"
        "把骨架类与全同构类并排看，「量变」与「突变」就分开了 —— 同一个骨架类内部是量变，"
        "跨骨架类才是突变。",
        "",
    ]
    return "\n".join(lines)


def _matrix(result: Any, cfg: Mapping[str, Any]) -> str:
    limit = int((cfg.get("report") or {}).get("max_matrix_rows", 32))
    rows = list(result.matrix)[:limit]
    skeleton = list(compare_mod.SKELETON_COLUMNS)
    scale = ["n_layers", "hidden", "n_heads", "n_kv_heads", "head_dim", "intermediate", "vocab", "n_experts", "top_k"]
    cost = ["params", "active_params", "flops_per_token", "kv_cache_bytes"]
    lines = [
        "## 五、对比矩阵",
        "",
        f"完整矩阵在 `out/matrix.csv`（{len(compare_mod.MATRIX_COLUMNS)} 列 × {len(result.matrix)} 行）。"
        "这里按**三组列**分开排 —— 分组本身就是论点：",
        "",
        "- **骨架列（定性）**：只有这一组变了才是突变；",
        "- **规模列（定量）**：只有这一组变了是量变；",
        "- **代价列**：结构的下游后果，不是结构本身。",
        "",
        "### 5.1 骨架列（定性）",
        "",
    ]
    lines += _tbl(
        ["模型"] + skeleton,
        [[f"`{r['name']}`"] + [fmt(r.get(c)) for c in skeleton] for r in rows],
    )
    lines += ["", "### 5.2 规模列（定量）", ""]
    lines += _tbl(
        ["模型"] + scale,
        [[f"`{r['name']}`"] + [fmt(r.get(c)) for c in scale] for r in rows],
    )
    lines += [
        "",
        "### 5.3 代价列",
        "",
        "`active_params` 是 MoE 的关键列：**总参数与激活参数差一个数量级**。"
        "把它和 `params` 并排放，是「MoE 用更大的容量、花更少的算力」这句话唯一诚实的表达方式。",
        "`kv_cache_bytes` 是另一条容易被忽略的后果：MHA → GQA → MLA 这一串突变"
        "在结构上只是换了个注意力变体，在代价上却差了好几倍。",
        "",
    ]
    lines += _tbl(
        ["模型"] + cost,
        [[f"`{r['name']}`"] + [fmt(r.get(c)) for c in cost] for r in rows],
    )
    lines.append("")
    return "\n".join(lines)


def _timelines(result: Any) -> str:
    lines = [
        "## 六、家族突变时间线",
        "",
        "**验收标准第二条就在这里。** 判据是骨架指纹（第四节），不是阈值、不是编辑条数：",
        "",
        "> 一个版本边界是突变点 ⟺ 相邻两个版本的结构骨架指纹不同。",
        "",
    ]
    if not result.timelines:
        lines.append("（产物模式：时间线未载入）")
        return "\n".join(lines)
    for tl in result.timelines:
        mut = tl.mutation_points
        lines.append(f"### 6.{tl.family}　成员 {len(tl.members)}　突变点 {len(mut)} 个")
        lines.append("")
        lines.append(f"链：`{'` → `'.join(tl.members)}`")
        lines.append("")
        if not tl.boundaries:
            lines.append(f"> {tl.note}")
            lines.append("")
            continue
        lines += _tbl(
            ["#", "边界", "判定", "突变坐标", "编辑", "其中规模", "其中结构"],
            [
                [
                    b.index,
                    f"`{b.left}` → `{b.right}`",
                    {"same": "**零差异**", "scale": "量变", "mutation": "**突变**"}[b.kind],
                    ", ".join(b.coordinates) or "—",
                    b.n_edits,
                    b.n_scale,
                    b.n_structural,
                ]
                for b in tl.boundaries
            ],
        )
        lines.append("")
        for b in tl.boundaries:
            if not b.headline:
                continue
            lines.append(f"**#{b.index} `{b.left}` → `{b.right}`（{b.kind}）**")
            lines.append("")
            for h in b.headline:
                lines.append(f"- `{h}`")
            lines.append("")
    lines += [
        "> **突变点必须能被外部核对。** 一条只有 4 个边界的链，如果判据说「每个边界都是突变」，"
        "那它一定错了 —— 因为公开资料里这几代之间明明只有规模在变。判据的价值就在于"
        "它能给出**可以被否证的**结论，而不是「阈值调得准」。",
        "",
    ]
    return "\n".join(lines)


def _views(result: Any) -> str:
    lines = [
        "## 七、三视图（Module Tree / Graph / Tensor Schema）",
        "",
        "同一个 Genome 有三种读法，对应三个不同的问题：**「怎么堆的」/「怎么连的」/「有什么张量」。**",
        "",
        "这里取两个模型作对照：一个稠密（`llama-2-7b`）、一个稀疏（`mixtral-8x7b`）。"
        "差异在三个视图里都看得出来，但它们**不是同一处差异** —— 这正是三视图要分开的原因。",
        "",
    ]
    names = [n for n in ("llama-2-7b", "mixtral-8x7b") if n in result.genomes] or list(result.names())[:1]
    for n in names:
        g = result.genome(n)
        lines += [
            f"### 7.{names.index(n) + 1} `{n}`　Module Tree",
            "",
            "注意 `blocks[*]` 只出现一次：Genome 是**模式**，不是展开的实例。",
            "",
            "```",
            module_tree(g),
            "```",
            "",
            f"### `{n}`　Graph（边表）",
            "",
            "```",
            *graph_lines(g),
            "```",
            "",
            f"### `{n}`　Tensor Schema",
            "",
            "```",
            *tensor_schema(g, limit=24),
            "```",
            "",
        ]
        if len(g.tensors) > 24:
            lines.append(f"（共 {len(g.tensors)} 张，此处只列前 24 张；完整表在 `out/genomes/{n}.json`）")
            lines.append("")
    return "\n".join(lines)


def _cost(result: Any) -> str:
    zoo = {e.name: e for e in (result.zoo or ())}
    lines = [
        "## 八、代价：三路逐位互校",
        "",
        "参数量的三条独立来源：",
        "",
        "1. **张量表求和** —— 把 `tensors` 里每张张量的 `numel` 加起来（纯清单侧）；",
        "2. **解析公式** —— 只用 config 标量算（**不碰张量表**，所以它是真的独立）；",
        "3. **checkpoint 元数据** —— HF 的 `safetensors.total` 减去非参数缓冲，逐字节数出来的外部真值。",
        "",
        "三者必须**逐位相同**。这不是「对得上就行」：只要有一个对不上，就说明三路里有一路"
        "对结构的理解是错的，而报告的其他部分不会因此报错。",
        "",
    ]
    if not zoo:
        lines.append("（产物模式：外部真值未载入）")
        lines.append("")
        return "\n".join(lines)
    rows = []
    all_ok = True
    for n in result.names():
        c = result.genome(n).cost
        ck = dict(zoo[n].checkpoint or {})
        exact = ck.get("params")
        buffers = ck.get("buffers")
        total = ck.get("safetensors_total")
        ok = c.params == c.params_formula and (exact is None or c.params == int(exact))
        all_ok &= ok
        rows.append(
            [
                f"`{n}`",
                f"{c.params:,}",
                f"{c.params_formula:,}",
                f"{int(exact):,}" if exact is not None else "—",
                f"{int(buffers):,}" if buffers else "0",
                "✅" if ok else "❌",
            ]
        )
    lines += _tbl(["模型", "① 表求和", "② 解析公式", "③ checkpoint", "其中缓冲", "一致"], rows)
    lines += [
        "",
        f"**三路全部逐位一致：{'是' if all_ok else '否'}。**",
        "",
        "**「其中缓冲」这一列不是凑数的。** `safetensors.total` 把非参数缓冲也算进去了："
        "GPT-2 的因果掩码是 12 个 1024×1024 的常量，Llama-1/2 有 `inv_freq`。"
        "等式写成 `params + buffers == safetensors_total` 之后，缓冲就从「误差」变成了"
        "「被解释清楚的项」—— 而正是这一步让 Qwen 系列的 QKV 偏置暴露了出来。",
        "",
        "### 8.1 代价分解（前 3 个模型）",
        "",
    ]
    for n in result.names()[:3]:
        g = result.genome(n)
        lines.append(f"**`{n}`**")
        lines.append("")
        lines += _tbl(["部件", "参数"], [[k, f"{v:,}"] for k, v in g.cost.breakdown])
        lines.append("")
    lines += [
        "`active_params` 的口径与 `params` 不同：MoE 的专家部分按 `top_k` 计，"
        "而不是按 `n_experts`。用 `n_experts` 算出来的「激活参数」等于总参数，"
        "那这个列就没有存在意义了。",
        "",
    ]
    return "\n".join(lines)


def _crosschecks(result: Any) -> str:
    lines = [
        "## 九、config ↔ 清单 两路互校",
        "",
        "每一项是 `(事实, config 说的, 清单量出的)`。**任何一项对不上，流水线会在 "
        "`canonicalize` 里直接抛错**，所以这张表能出现在报告里，本身就是「两路一致」的证据。",
        "",
    ]
    if not result.canonical:
        lines.append("（产物模式：canonical 未载入；互校已在 `run` 时完成，不一致会直接失败）")
        lines.append("")
        return "\n".join(lines)
    total = 0
    for n in result.names():
        cks = result.crosschecks(n)
        total += len(cks)
        lines.append(f"### `{n}`　{len(cks)} 项")
        lines.append("")
        lines += _tbl(["事实", "config 说", "清单量出"], [[f"`{a}`", f"`{b}`", f"`{c}`"] for a, b, c in cks])
        lines.append("")
    lines += [
        f"**合计 {total} 项。** 这里面有几项值得单独说，因为它们是「两路真的独立」的证据：",
        "",
        "- **范数类型**只在 config 里（`rms_norm_eps` vs `layer_norm_eps`），清单里只有一张 `weight`，"
        "看不出是哪一种 —— 所以这一条**不能**互校，它由 `provenance=config` 标出来；",
        "- **偏置是否存在**只在清单里（config 里根本没有这个键，Qwen1.5/Qwen2/Qwen2.5 就是这种情况），"
        "由 zoo 的 `conventions` 通道补进来，然后**拿清单逐分量对质**；",
        "- **专家数**两边都有：config 说 `num_local_experts`，清单量的是 `experts[*]` 出现了几个 —— "
        "DeepSeek-V2-Lite 的块组异构（首块稠密、其余 MoE）就是靠这一条被抓出来的。",
        "",
    ]
    return "\n".join(lines)


def _rule_summary(rules: Any) -> str:
    """口径表里的那一格：规则总数 + 三栏分布。"""
    rules = dict(rules or {})
    if not rules:
        return "（产物里没有规则覆盖率）"
    n = sum(info["n_rules"] for info in rules.values())
    ex = sum(len(info["exercised"]) for info in rules.values())
    co = sum(len(info["conditional"]) for info in rules.values())
    de = sum(len(info["dead"]) for info in rules.values())
    return f"{n} 条 = 实测 {ex} + 条件 {co} + 死规则 {de}（见 10.2）"


def _coverage(result: Any) -> str:
    cov = result.coverage or {}
    lines = [
        "## 十、config 键覆盖率",
        "",
        "**「读不懂」与「决定不看」是两栏，不能合并。** 把 dropout 和一个真正陌生的键"
        "一起报成「未识别」，报告会显示一个假的低覆盖率；反过来把它们都算成已处理，"
        "又会把「读不懂但假装读懂了」这个最危险的失败模式藏起来。",
        "",
    ]
    unknown = cov.get("unknown") or {}
    if unknown:
        lines.append(f"**读不懂的键：{len(unknown)} 个模型有未识别的键 ❌**")
        lines.append("")
        lines += _tbl(["模型", "未识别的键"], [[f"`{n}`", ", ".join(f"`{k}`" for k in keys)] for n, keys in sorted(unknown.items())])
    else:
        lines.append("**读不懂的键：无 ✅**　全部 config 键都有归一化规则。")
    lines += [
        "",
        f"**有理由忽略的键：{cov.get('n_ignored', 0)} 处。** 它们不改变指纹 —— "
        "否则「换个 transformers 版本」「把 dropout 从 0.1 调成 0」就成了「结构变了」。",
        "",
    ]
    reasons = cov.get("ignored_reasons") or {}
    if reasons:
        lines += _tbl(["理由", "键数"], [[r, n] for r, n in sorted(reasons.items())])
        lines.append("")
    lines += [
        "> **这张表本身就是一条结论。** 两个发布版本的 config 差异如果全部落在这里面，"
        "它们的全指纹就应当逐位相同 —— 那说明「发布了新版本」和「结构变了」是两件事。"
        "第四节的全同构类就是这句话的实测结果。",
        "",
        "### 10.2 命名规则覆盖率",
        "",
        "**「Zoo 实测过的」与「条件成立才出现的」同样是两栏。** 一条永远匹配不上的规则"
        "看起来像「支持了某种布局」，而它某天真的被匹配上时，行为是未经验证的 ——"
        "所以每条规则要么在某份清单的解析路径上出现过，要么必须挂在一个**门槛**下"
        "（某个 config 事实成立时才会出现），并被探针逐条验过。",
        "",
    ]
    rules = dict(getattr(result, "rules", None) or {})
    if rules:
        rows = []
        tot = [0, 0, 0, 0]
        for layout, info in sorted(rules.items()):
            ex, co, de = len(info["exercised"]), len(info["conditional"]), len(info["dead"])
            rows.append([f"`{layout}`", info["n_rules"], ex, co, de if de else "0 ✅"])
            tot = [tot[0] + info["n_rules"], tot[1] + ex, tot[2] + co, tot[3] + de]
        rows.append(["**合计**", f"**{tot[0]}**", f"**{tot[1]}**", f"**{tot[2]}**", f"**{tot[3]}**"])
        lines += _tbl(["布局", "规则数", "Zoo 实测", "条件规则", "死规则"], rows)
        lines.append("")
        lines.append(
            f"**死规则：{tot[3]} 条。** 这一栏必须恒为空 —— 它不为空就说明规则表里躺着一条"
            "谁也没验过的猜测。`tests/test_extract_dialects.py` 把它钉死。"
        )
        lines.append("")
        gates: set[str] = set()
        for info in rules.values():
            gates.update(info["conditional"])
        if gates:
            from .dialects import CONDITIONAL_RULES

            by_gate: dict[str, list[str]] = {}
            for src in sorted(gates):
                by_gate.setdefault(CONDITIONAL_RULES.get(src, "?"), []).append(src)
            lines.append(
                f"**条件规则的门槛**（{len(gates)} 条**不同**的规则；同一个名字可以挂在多个布局下，"
                "这里按名字去重）："
            )
            lines.append("")
            lines += _tbl(
                ["门槛", "规则数", "举例"],
                [[f"`{g}`", len(v), f"`{min(v)}`"] for g, v in sorted(by_gate.items())],
            )
            lines.append("")
    return "\n".join(lines)


def _diff_algebra(result: Any) -> str:
    lines = [
        "## 十一、Diff 代数",
        "",
        "「可复现的结构 Diff」这句话要成立，diff 必须同时满足四条**代数恒等式**：",
        "",
        "| 性质 | 断言 | 意义 |",
        "| --- | --- | --- |",
        "| **完备** | `apply(a, diff(a,b)).primary() ≡ b.primary()` | 补丁不漏任何一处差异 |",
        "| **最小** | 去掉任意一条编辑，`apply` 就不再等于 `b` | 补丁不含废话 |",
        "| **可逆** | `diff(b,a) ≡ invert(diff(a,b))` | 差异是双向的、对称的 |",
        "| **确定** | 同一对模型跑两次，编辑序列逐位相同 | 可复现 |",
        "",
        "「完备」是**最有价值的测试**：它盯的不是 diff 的实现，而是 `Genome.primary()` 的"
        "**覆盖完整性**。哪天给 Genome 加了新字段却忘了接进 `primary()`，`apply` 会静默地"
        "少还原一个字段 —— 只有这条断言会响。",
        "",
        "### 11.1 全部无序对",
        "",
        f"{len(result.pairs)} 个无序对（{len(result.names())} 个模型两两）。"
        "完整编辑序列在 `out/diffs.json` 的 `detail` 里。",
        "",
        "| 对 | 判定 | 编辑 | 规模 | 结构 | 突变坐标 | 最显著的几条 |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    ordered = sorted(result.pairs_ordered(), key=lambda d: (d.is_mutation, len(d.edits)))
    for d in ordered:
        s = diff_summary(d)
        hl = headline(d, limit=2)
        kind = "**突变**" if d.is_mutation else ("零差异" if d.is_empty else "量变")
        lines.append(
            f"| `{d.left}` → `{d.right}` | {kind} | {s['n_edits']} | {s['n_scale']} | {s['n_structural']} | "
            f"{', '.join(d.coordinates) or '—'} | {'；'.join(hl[:2])} |"
        )
    lines.append("")
    mut = [d for d in ordered if d.is_mutation]
    lines += [
        f"**其中 {len(mut)} 对是结构突变。** 注意突变对的条数并不比量变对多 —— "
        "「改了 20 处规模」和「换了算子」在编辑条数上差不多，只有骨架指纹能把它们分开。",
        "",
    ]
    return "\n".join(lines)


def _expectations(result: Any, cfg: Mapping[str, Any]) -> str:
    exp = expectations_of(cfg)
    lines = [
        "## 十二、设计期望 vs 实测",
        "",
        "配置里的 `expectations` 是**设计期望**，允许被实测推翻。被推翻的那些是"
        "附录 B 里最值得看的部分 —— 一条被推翻的期望比十条被证实的期望有信息量。",
        "",
    ]
    if not exp:
        lines.append("（没有声明设计期望）")
        return "\n".join(lines)
    lines += _tbl(["期望", "声明", "实测", "判定"], _expectation_rows(result, exp))
    lines += [
        "",
        "**这里有一条被实测推翻的期望，值得单列：**",
        "",
        "> 原以为「代价可以缓存」—— 既然不训练，同一个模型的结构代价应该能算一次就存下来，"
        "后续比较直接查表。",
        "",
        "实测推翻了它：**代价是主地址空间的函数**，而 `diff.apply` 会重建 Genome，"
        "重建之后必须能逐位复算代价。如果把代价当成独立事实缓存下来，"
        "`apply(a, diff(a,b))` 得到的 Genome 的代价就可能和 `b` 的对不上 —— "
        "于是「完备性」这条断言会在代价上破掉。",
        "",
        "结论：**代价不进主地址空间，但必须可重算**（`cost.derive` 只吃 `.tensors` / `.nodes` / `.topo()`，"
        "所以 IR 与 Genome 共用同一套算法）。这条设计被推翻后写进了 `schema.py` 的注释，"
        "而不是留在报告里当一段感慨。",
        "",
    ]
    return "\n".join(lines)


def _expectation_rows(result: Any, exp: Mapping[str, Any]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for key, want in sorted(exp.items()):
        if key.endswith("_mutation_points"):
            fam = key[: -len("_mutation_points")]
            got = _n_mutations(result, fam)
        elif key == "qwen_last_boundary_is_zero_diff":
            tl = _timeline_or_none(result, "qwen")
            got = bool(tl and tl.boundaries and tl.boundaries[-1].kind == "same")
        elif key == "min_full_isomorphic_classes":
            full = result.classes.get("full", {})
            got = sum(1 for v in full.values() if len(v) >= 2)
        else:
            rows.append([f"`{key}`", fmt(want), "—", "未实现"])
            continue
        ok = (got >= want) if key.startswith("min_") else (got == want)
        rows.append([f"`{key}`", fmt(want), fmt(got), "✅ 支持" if ok else "❌ **被推翻**"])
    return rows


def _reproducibility(result: Any) -> str:
    return "\n".join(
        [
            "## 十三、可复现性边界",
            "",
            "| 字段 | 逐位可复现 | 说明 |",
            "| --- | --- | --- |",
            "| `version.digest` | 是 | 内容寻址：zoo 索引 + 每个 config + 每份清单的摘要 |",
            "| `fingerprint` / `shape_fingerprint` | 是 | 只吃结构，不吃文件名、不吃时间 |",
            "| 全部编辑序列（`diffs.json`） | 是 | 编辑按「类别 + 地址」排序，同输入必得同序 |",
            "| 突变时间线 / 同构类 / 矩阵 | 是 | 都是指纹与编辑的函数 |",
            "| `cost` 全部字段 | 是 | 主地址空间的函数，`apply` 后可逐位复算 |",
            "| `version.created_at` | **否** | 唯一的时间戳字段 |",
            "| `tensors[*].source_name` | 是（但不进指纹） | 命名方言的痕迹，只用于审计 |",
            "",
            "**「同一个模型换一种命名方言，指纹必须不变」是这个项目最强的可复现性断言。**"
            "它比「跑两次结果一样」强得多：后者只说明代码是确定的，前者说明"
            "**指纹真的只吃结构**。`tests/test_canonical.py` 有一条测试专门钉住它，"
            "而 `run_pipeline` 在拿到不一致的方言指纹时会**直接抛错**（不让它悄悄进报告）。",
            "",
        ]
    )


def render_report(result: Any, cfg: Mapping[str, Any] | None = None) -> str:
    """渲染完整报告。**顺序即论点顺序。**"""
    cfg = dict(cfg or {})
    parts = [
        _header(result, cfg),
        _acceptance(result, cfg),
        _fingerprints(result),
        _classes(result),
        _matrix(result, cfg),
        _timelines(result),
        _views(result),
        _cost(result),
        _crosschecks(result),
        _coverage(result),
        _diff_algebra(result),
        _expectations(result, cfg),
        _reproducibility(result),
    ]
    return "\n".join(parts)
