"""命令行。**读命令不写盘**，写命令只有 `materialize` / `run` / `visualize`。

这不是洁癖：`out/` 里那份结果可能对应另一份配置，用户看一眼报告就会拿它下结论。
如果「看一眼」会顺手改写产物，结论与配置的对应关系就断了。所以：

- 读命令**从产物重建结果对象**（`pipeline.load_result`），绝不重跑流水线；
- 产物缺失、或该命令需要产物里**没有**的东西（zoo 元数据、canonical IR）时，
  读命令在内存里跑一遍 —— 仍然**不落盘**，并明确告诉你「本次是内存重跑」；
- `materialize` 是唯一改**输入数据**的命令（它渲染 `data/zoo/<name>/manifest.<d>.json`），
  所以它是独立子命令而不是 `run` 的一部分。

命令清单（前 18 个只读）：

```
summary / zoo / facts / tree / graph / tensors / genome / fingerprint / classes
/ matrix / diff / mutations / crosscheck / cost / coverage / rules / lineage / acceptance
materialize / run / visualize          ← 写
```

`acceptance` 与报告共用 `report.acceptance_checks` —— 于是「命令行说通过的」
与「报告里写的」不可能不一致。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from . import compare as compare_mod
from .config import DEFAULTS, expectations_of, load_config
from .diff import apply as apply_diff
from .diff import headline, summary as diff_summary
from .extract import extract_model, load_zoo
from .genome import same_primary
from .pipeline import EXTRA_FILES, OUT_FILES, load_result, run_pipeline, write_outputs
from .report import acceptance_checks, fmt, graph_lines, module_tree, tensor_schema

#: 项目根目录（本文件在 `src/modelgenomefactory/cli.py`）。
ROOT = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

DEFAULT_CONFIG = os.path.join("configs", "default.json")


def _out_dir(args: argparse.Namespace) -> Path:
    p = Path(args.out)
    return p if p.is_absolute() else ROOT / p


# ---------------------------------------------------------------------- 取数


def _result(args: argparse.Namespace, *, fresh: bool = False) -> tuple[Any, dict[str, Any], bool]:
    """返回 `(结果, 配置, 是否来自产物)`。**永不落盘。**

    `fresh=True` 用于那些需要产物里没有的东西的命令（zoo 元数据 / canonical IR）。
    与其在产物里塞一份冗余副本，不如**明说这次是重算的** —— 产物就该只存
    「比较的结论」，不该存「比较的输入」。
    """
    cfg = load_config(args.config, root=ROOT)
    if not fresh:
        cached = load_result(_out_dir(args))
        if cached is not None:
            return cached, cfg, True
    return run_pipeline(cfg, root=ROOT), cfg, False


def _note(from_products: bool) -> str:
    return "" if from_products else "  （产物不存在或本命令需要额外数据：本次在内存中重跑，未写盘）"


def _w(s: Any) -> int:
    """终端显示宽度。CJK 与全角标点占两列 —— `str.ljust` 只数字符，
    于是表头里的「模型」会让整张表错位两列。"""
    import unicodedata

    return sum(2 if unicodedata.east_asian_width(c) in "WF" else 1 for c in str(s))


def _pad(s: Any, width: int, *, right: bool = False) -> str:
    gap = " " * max(0, width - _w(s))
    return (gap + str(s)) if right else (str(s) + gap)


def _row(cells: Sequence[tuple[str, int]], *, right: Sequence[int] = ()) -> str:
    """一行表格。列间留一个空格 —— 不留的话「缓冲」和「一致」会粘成「缓冲一致」。"""
    parts = [_pad(v, w, right=i in right) for i, (v, w) in enumerate(cells)]
    return "  " + " ".join(parts).rstrip()


# ------------------------------------------------------------ --json 的出口

#: 真正的 stdout。`--json` 时命令里的 `print` 会被 `_quiet_prose` 吞掉，
#: 只有 `_emit_json` 写到这里。
_REAL_STDOUT = sys.stdout


def _emit_json(payload: Any) -> None:
    """`--json` 的唯一出口：**只**打印一段 JSON，不夹任何说明文字。

    早先的实现是「先打印说明，再追加一段 JSON」—— 于是 `--json` 的输出没法直接
    `| jq`，下游得先自己找到第一个 `{`，而说明里恰好出现一个花括号就会解析错。
    既然声明了机器可读，就只输出机器可读的东西。
    """
    _REAL_STDOUT.write(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    _REAL_STDOUT.flush()


class _quiet_prose:
    """把命令里的 `print` 全部吞掉，只在 `--json` 时使用。

    这样每个命令的写法不用改：说明照常 `print`，JSON 照常 `_emit_json` ——
    但两者不会同时出现在 stdout 上。
    """

    def __enter__(self) -> "_quiet_prose":
        import io

        self._saved = sys.stdout
        sys.stdout = io.StringIO()
        return self

    def __exit__(self, *exc: object) -> bool:
        sys.stdout = self._saved
        return False


def _zoo() -> tuple[Any, ...]:
    return load_zoo(ROOT)


def _entry(name: str) -> Any:
    for e in _zoo():
        if e.name == name:
            return e
    near = [e.name for e in _zoo() if name[:4] in e.name]
    print(f"zoo 里没有模型 {name!r}", file=sys.stderr)
    if near:
        print("你是不是想找：" + ", ".join(near), file=sys.stderr)
    return None


# ---------------------------------------------------------------------- 只读


def cmd_summary(args: argparse.Namespace) -> int:
    r, cfg, cached = _result(args)
    v = r.version
    counts = dict(v.counts)
    print(f"版本        : {v.id}")
    print(f"内容摘要    : {v.digest}")
    print(f"协议摘要    : {v.protocol_digest}")
    print(f"创建时间    : {v.created_at}  （唯一不可逐位复现的字段）")
    print(f"zoo         : {counts.get('models', 0)} 个模型 / {counts.get('dialects', 0)} 份命名清单")
    print(f"两两比较    : {counts.get('pairs', 0)} 个无序对")
    print(f"血缘图      : {counts.get('lineage_nodes', 0)} 节点 / {counts.get('lineage_edges', 0)} 边")
    print(f"命名方言    : {', '.join(cfg.get('dialects') or [])}")
    print("优化目标    : 无（测量型：只测量结构、不训练、不选优）")
    print(f"家族        : {', '.join(r.families()) if r.zoo else ', '.join(t.family for t in r.timelines)}")
    mut = sum(len(t.mutation_points) for t in r.timelines)
    print(f"突变点      : {mut} 个（跨 {len(r.timelines)} 条家族链）")
    print(f"来源        : {'产物' if cached else '内存重跑'}{_note(cached)}")
    return 0


def cmd_zoo(args: argparse.Namespace) -> int:
    print(f"zoo（{len(_zoo())} 个公开模型的结构摘录）\n")
    print(_row([("模型", 20), ("家族", 9), ("链", 3), ("方言", 22), ("公开参数量", 12), ("checkpoint", 18)], right=(2, 4, 5)))
    for e in _zoo():
        ck = dict(e.checkpoint or {})
        total = ck.get("safetensors_total")
        dialects = ",".join(d for d, _ in e.dialects)
        print(
            _row(
                [
                    (e.name, 20),
                    (e.family, 9),
                    (str(e.chain), 3),
                    (dialects, 22),
                    (f"{e.params_b}B" if e.params_b else "—", 12),
                    (f"{int(total):,}" if total else "—", 18),
                ],
                right=(2, 4, 5),
            )
        )
    print("\n  注：本项目只读 config 与参数清单，**不下载权重、不训练**。")
    print("      checkpoint 列是 HF 的 safetensors.total（逐字节数出来的外部真值）。")
    if args.json:
        _emit_json([e.to_dict() for e in _zoo()])
    return 0


def cmd_facts(args: argparse.Namespace) -> int:
    """一个模型的 config 归一化明细：**每个数字是从哪个原键来的。**"""
    e = _entry(args.model)
    if e is None:
        return 1
    raw = extract_model(e)
    print(f"{e.name} 的 config 归一化（{len(raw.facts)} 条事实）\n")
    print(_row([("规范键", 24), ("值", 16), ("原键", 26), ("规则", 0)], right=(1,)))
    for f in raw.facts:
        val = f.value if not isinstance(f.value, float) else f"{f.value:g}"
        print(_row([(f.key, 24), (str(val), 16), (f.source_key, 26), (f.rule, 0)], right=(1,)))
    if raw.unknown_keys:
        print(f"\n  读不懂的键（{len(raw.unknown_keys)} 个）：{list(raw.unknown_keys)}")
    else:
        print("\n  读不懂的键：无 ✅")
    if raw.ignored_keys:
        by_reason: dict[str, list[str]] = {}
        for k, reason in raw.ignored_keys:
            by_reason.setdefault(reason, []).append(k)
        print(f"\n  有理由忽略的键（{len(raw.ignored_keys)} 个）—— 它们不改变指纹：")
        for reason, keys in sorted(by_reason.items()):
            print(f"    {reason}")
            print(f"      {', '.join(sorted(keys))}")
    conv = dict(e.conventions or {})
    if conv:
        print(f"\n  家族约定（config 里没有、官方 checkpoint 里有）：{conv}")
        print("  ⇒ 它们不是猜的默认值：填进去之后 canonicalize 会拿清单逐分量对质。")
    if args.json:
        _emit_json(raw.to_dict())
    return 0


def _pick_genome(args: argparse.Namespace) -> tuple[Any, Any, bool]:
    r, cfg, cached = _result(args)
    if args.model not in r.genomes:
        print(f"没有 {args.model!r} 的 genome", file=sys.stderr)
        print("可用：" + ", ".join(r.names()), file=sys.stderr)
        return None, r, cached
    return r.genome(args.model), r, cached


def cmd_tree(args: argparse.Namespace) -> int:
    g, _r, cached = _pick_genome(args)
    if g is None:
        return 1
    print(f"{g.name} 的 Module Tree{_note(cached)}\n")
    print(module_tree(g))
    return 0


def cmd_graph(args: argparse.Namespace) -> int:
    g, _r, cached = _pick_genome(args)
    if g is None:
        return 1
    if args.dot:
        print("digraph genome {")
        print('  rankdir="LR";')
        for e in g.edges:
            print(f'  "{e.src}" -> "{e.dst}" [label="{e.label}"];')
        print("}")
        return 0
    print(f"{g.name} 的计算图（{len(g.edges)} 条边）{_note(cached)}\n")
    for line in graph_lines(g):
        print("  " + line)
    return 0


def cmd_tensors(args: argparse.Namespace) -> int:
    g, _r, cached = _pick_genome(args)
    if g is None:
        return 1
    print(f"{g.name} 的 Tensor Schema（{len(g.tensors)} 张）{_note(cached)}\n")
    print(_row([("规范路径", 52), ("角色", 14), ("形状", 22), ("绑定", 22), ("来源", 0)]))
    for line in tensor_schema(g):
        print("  " + line)
    print("\n  注：`source_name`（原清单里的名字）**不进指纹** —— 换个命名方言，指纹必须不变。")
    if args.json:
        _emit_json([t.to_dict() for t in g.tensors])
    return 0


def cmd_genome(args: argparse.Namespace) -> int:
    g, r, cached = _pick_genome(args)
    if g is None:
        return 1
    if args.json:
        _emit_json(g.to_dict())
        return 0
    c = g.cost
    print(f"{g.name}{_note(cached)}\n")
    print(f"  身份      : {g.identity_dict()}")
    print(f"  方言      : {g.dialect}（指纹与方言无关）")
    print(f"  topology  : {dict(g.topology)}")
    print(f"  节点      : {len(g.nodes)} 个 {sorted({n.kind for n in g.nodes})}")
    print(f"  张量      : {len(g.tensors)} 张，角色 {list(g.role_names())}")
    print(f"  边        : {len(g.edges)} 条")
    print(f"  骨架指纹  : {r.shape_fingerprint_of(g.name)}")
    print(f"  全指纹    : {r.fingerprint_of(g.name)}")
    print(f"  参数量    : {c.params:,}（表求和）/ {c.params_formula:,}（解析公式）")
    print(f"  激活参数  : {r.active_params(g.name):,}")
    print(f"  FLOPs/tok : {c.flops_per_token:,}  @seq={c.seq_len}")
    print(f"  KV 缓存   : {c.kv_cache_bytes:,} 字节  @seq={c.seq_len} batch={c.batch} dtype={c.dtype_bytes}B")
    print(f"  权重字节  : {c.weight_bytes:,}")
    if g.notes:
        print("\n  备注：")
        for n in g.notes:
            print(f"    - {n}")
    return 0


def cmd_fingerprint(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args)
    print(f"两级指纹{_note(cached)}\n")
    print(_row([("模型", 20), ("骨架指纹", 14), ("全指纹", 14), ("参数量", 16)], right=(3,)))
    for n in r.names():
        print(
            _row(
                [
                    (n, 20),
                    (r.shape_fingerprint_of(n)[:12], 14),
                    (r.fingerprint_of(n)[:12], 14),
                    (f"{r.genome(n).cost.params:,}", 16),
                ],
                right=(3,),
            )
        )
    print("\n  骨架指纹相同 ⟺ 只改了规模；骨架指纹不同 ⟺ 结构突变。没有阈值。")
    if args.json:
        _emit_json(
            {n: {"shape": r.shape_fingerprint_of(n), "full": r.fingerprint_of(n)} for n in r.names()}
        )
    return 0


def cmd_classes(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args)
    print(f"同构类（按指纹分组）{_note(cached)}\n")
    for kind, label in (("full", "全同构（逐位相同）"), ("shape", "骨架类（同一结构、不同规模）")):
        bucket = r.classes.get(kind, {})
        big = {k: v for k, v in bucket.items() if len(v) >= args.min_size}
        print(f"  {label}：{len(bucket)} 类，其中成员 ≥{args.min_size} 的 {len(big)} 类")
        for k, v in sorted(big.items(), key=lambda kv: -len(kv[1])):
            print(f"    `{k[:12]}`  ×{len(v)}  {', '.join(v)}")
        print()
    print("  注：成员为 1 的类没有信息量，默认不展示（用 --min-size 1 看全部）。")
    if args.json:
        _emit_json(r.classes)
    return 0


def cmd_matrix(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args)
    cols = list(compare_mod.SKELETON_COLUMNS) + list(compare_mod.SCALE_COLUMNS) + ["params", "active_params"]
    if args.group == "skeleton":
        cols = list(compare_mod.SKELETON_COLUMNS)
    elif args.group == "scale":
        cols = list(compare_mod.SCALE_COLUMNS)
    elif args.group == "cost":
        cols = list(compare_mod.COST_COLUMNS)
    print(f"对比矩阵（{len(r.matrix)} 行）{_note(cached)}\n")
    print(_row([("模型", 20)] + [(c[:11], 13) for c in cols]))
    for row in r.matrix[: args.limit]:
        print(_row([(row["name"], 20)] + [(fmt(row.get(c))[:11], 13) for c in cols], right=range(1, len(cols) + 1)))
    print(f"\n  完整矩阵（{len(compare_mod.MATRIX_COLUMNS)} 列）在 out/matrix.csv。")
    if args.json:
        _emit_json(list(r.matrix))
    return 0


def cmd_diff(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args)
    if args.all:
        print(f"全部无序对（{len(r.pairs)} 对）{_note(cached)}\n")
        print(_row([("对", 46), ("判定", 6), ("编辑", 4), ("规模", 4), ("结构", 4), ("突变坐标", 0)], right=(2, 3, 4)))
        for d in sorted(r.pairs_ordered(), key=lambda x: (x.is_mutation, len(x.edits))):
            kind = "突变" if d.is_mutation else ("零差异" if d.is_empty else "量变")
            print(
                _row(
                    [
                        (f"{d.left} → {d.right}", 46),
                        (kind, 6),
                        (str(len(d.edits)), 4),
                        (str(len(d.scale_edits())), 4),
                        (str(len(d.structural_edits())), 4),
                        (", ".join(d.coordinates) or "—", 0),
                    ],
                    right=(2, 3, 4),
                )
            )
        print(f"\n  其中 {sum(1 for d in r.pairs_ordered() if d.is_mutation)} 对是结构突变。")
        return 0
    a, b = args.left, args.right
    try:
        d = r.diff(a, b)
    except KeyError as exc:
        print(str(exc), file=sys.stderr)
        print("可用：" + ", ".join(r.names()), file=sys.stderr)
        return 1
    s = diff_summary(d)
    print(f"结构 Diff：{a} → {b}{_note(cached)}\n")
    print(f"  判定      : {'**结构突变**' if d.is_mutation else ('零差异' if d.is_empty else '量变（只有规模变化）')}")
    print(f"  编辑      : {s['n_edits']} 条（规模 {s['n_scale']} / 结构 {s['n_structural']}）")
    print(f"  突变坐标  : {', '.join(d.coordinates) or '—'}")
    print(f"  骨架指纹  : {d.left_shape_fingerprint[:12]} → {d.right_shape_fingerprint[:12]}"
          f"  {'（变了）' if d.is_mutation else '（没变）'}")
    print(f"  全指纹    : {d.left_fingerprint[:12]} → {d.right_fingerprint[:12]}")
    print("\n  最显著的差异：")
    for h in headline(d, limit=args.limit):
        print(f"    - {h}")
    print("\n  按种类聚合：")
    for k, v in s["by_kind"].items():
        print(f"    {k:12s} ×{v}")
    if args.edits:
        print(f"\n  全部 {len(d.edits)} 条编辑：")
        for e in d.edits:
            print(f"    {e.render()}")
    if args.json:
        _emit_json(d.to_dict())
    return 0


def cmd_mutations(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args)
    tls = [t for t in r.timelines if not args.family or t.family == args.family]
    if not tls:
        print(f"没有家族 {args.family!r}", file=sys.stderr)
        return 1
    print(f"家族突变时间线{_note(cached)}\n")
    for tl in tls:
        mut = tl.mutation_points
        print(f"  {tl.family}（{len(tl.members)} 个成员，{len(mut)} 个突变点）")
        print(f"    链：{' → '.join(tl.members)}")
        if not tl.boundaries:
            print(f"    {tl.note}")
            print()
            continue
        print("    " + _row([("#", 3), ("边界", 46), ("判定", 6), ("编辑", 4), ("突变坐标", 0)], right=(0, 3)))
        for b in tl.boundaries:
            print(
                "    "
                + _row(
                    [
                        (str(b.index), 3),
                        (f"{b.left} → {b.right}", 46),
                        (b.kind, 6),
                        (str(b.n_edits), 4),
                        (", ".join(b.coordinates) or "—", 0),
                    ],
                    right=(0, 3),
                )
            )
            for h in b.headline[: args.limit]:
                print(f"        · {h}")
        print()
    print("  判据：相邻版本的结构骨架指纹不同 ⟺ 突变。没有阈值、没有容差。")
    if args.json:
        _emit_json([t.to_dict() for t in tls])
    return 0
    return 0


def cmd_crosscheck(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args, fresh=True)
    names = [args.model] if args.model else list(r.names())
    for n in names:
        if n not in r.genomes:
            print(f"没有 {n!r}", file=sys.stderr)
            return 1
    print(f"config ↔ 清单 两路互校{_note(cached)}\n")
    total = 0
    for n in names:
        cks = r.crosschecks(n)
        total += len(cks)
        print(f"  {n}（{len(cks)} 项）")
        for a, b, c in cks:
            print(f"    {a:34s} config={b:>14s}  清单={c:>14s}")
        print()
    print(f"  合计 {total} 项，全部一致 ✅")
    print("  任一项不一致，canonicalize 会直接抛错 —— 走不到这里。")
    if args.json:
        _emit_json({n: [list(c) for c in r.crosschecks(n)] for n in names})
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args)
    zoo = {e.name: e for e in _zoo()}
    names = [args.model] if args.model else list(r.names())
    print(f"代价三路互校{_note(cached)}\n")
    print(_row([("模型", 20), ("① 表求和", 17), ("② 解析公式", 17), ("③ checkpoint", 17), ("缓冲", 12), ("一致", 0)], right=(1, 2, 3, 4)))
    for n in names:
        if n not in r.genomes:
            print(f"没有 {n!r}", file=sys.stderr)
            return 1
        c = r.genome(n).cost
        ck = dict(zoo[n].checkpoint or {}) if n in zoo else {}
        exact = ck.get("params")
        buffers = ck.get("buffers") or 0
        ok = c.params == c.params_formula and (exact is None or c.params == int(exact))
        print(
            _row(
                [
                    (n, 20),
                    (f"{c.params:,}", 17),
                    (f"{c.params_formula:,}", 17),
                    (f"{int(exact):,}" if exact is not None else "—", 17),
                    (f"{buffers:,}", 12),
                    ("✅" if ok else "❌", 0),
                ],
                right=(1, 2, 3, 4),
            )
        )
    if not args.model:
        print()
        for n in r.names()[:3]:
            g = r.genome(n)
            print(f"  {n} 的分解：" + "，".join(f"{k}={v:,}" for k, v in g.cost.breakdown))
    print("\n  ③ 是 HF 的 safetensors.total（逐字节数出来的外部真值），等式写成 params + buffers == total。")
    print("  ② 只用 config 标量算，**不碰张量表** —— 所以它与 ① 是真的独立。")
    if args.json:
        _emit_json({n: r.genome(n).cost.to_dict() for n in names})
    return 0


def cmd_coverage(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args, fresh=True)
    cov = r.coverage or {}
    print(f"config 键覆盖率{_note(cached)}\n")
    print(f"  模型数      : {cov.get('n_models', 0)}")
    unknown = cov.get("unknown") or {}
    if unknown:
        print(f"  读不懂的键  : {len(unknown)} 个模型有未识别的键 ❌")
        for n, keys in sorted(unknown.items()):
            print(f"    {n:20s} {list(keys)}")
    else:
        print("  读不懂的键  : 无 ✅（全部 config 键都有归一化规则）")
    print(f"  有理由忽略  : {cov.get('n_ignored', 0)} 处（它们不改变指纹）")
    for reason, n in (cov.get("ignored_reasons") or {}).items():
        print(f"    ×{n:<3d} {reason}")
    print("\n  **两栏必须分开看。** 把「读不懂」和「决定不看」合并成一栏，")
    print("  报告会把一个真问题和一个设计决定说成同一件事。")
    if args.json:
        _emit_json(cov)
    return 0


def cmd_rules(args: argparse.Namespace) -> int:
    # 规则覆盖率现在是**产物**（`out/rules.json`），所以这里不 fresh ——
    # 只读命令读产物，不重算。重算只发生在产物缺失/过期时。
    r, _cfg, cached = _result(args)
    rules = dict(r.rules or {})
    print(f"命名规则表{_note(cached)}\n")
    print(_row([("布局", 16), ("规则数", 6), ("实测", 6), ("条件", 6), ("死规则", 6)], right=(1, 2, 3, 4)))
    tot = [0, 0, 0, 0]
    for layout, info in sorted(rules.items()):
        ex, co, de = len(info["exercised"]), len(info["conditional"]), len(info["dead"])
        print(_row([(layout, 16), (info["n_rules"], 6), (ex, 6), (co, 6), (de, 6)], right=(1, 2, 3, 4)))
        tot = [tot[0] + info["n_rules"], tot[1] + ex, tot[2] + co, tot[3] + de]
    print(_row([("合计", 16), (tot[0], 6), (tot[1], 6), (tot[2], 6), (tot[3], 6)], right=(1, 2, 3, 4)))
    print()
    print("  三栏分开是有意的：一条**从没被任何清单匹配过**的规则看起来像「支持了某种布局」，")
    print("  而它某天真的被匹配上时行为是未经验证的。所以每条规则要么在 Zoo 的解析路径上出现过，")
    print("  要么挂在一个明确的门槛下（某 config 事实成立时才出现），并被探针逐条验过。")
    dead = {k: v["dead"] for k, v in sorted(rules.items()) if v["dead"]}
    if dead:
        print(f"\n  ❌ 死规则（必须为空）：{dead}")
    else:
        print("\n  死规则：0 ✅")
    from .dialects import CONDITIONAL_RULES

    gates: set[str] = set()
    for info in rules.values():
        gates.update(info["conditional"])
    if gates:
        by_gate: dict[str, list[str]] = {}
        for src in sorted(gates):
            by_gate.setdefault(CONDITIONAL_RULES.get(src, "?"), []).append(src)
        print(f"\n  条件规则的门槛（{len(gates)} 条不同的规则，按名字去重）：")
        for gate, srcs in sorted(by_gate.items()):
            print(f"    {gate:12s} {len(srcs):2d} 条  例：{sorted(srcs)[0]}")
    if args.json:
        _emit_json({"layouts": rules, "conditional": dict(CONDITIONAL_RULES)})
    return 0


def cmd_lineage(args: argparse.Namespace) -> int:
    r, _cfg, cached = _result(args)
    if args.dot:
        print(r.lineage.to_dot())
        return 0
    if args.extract_only:
        keep = sorted(
            n.id for n in r.lineage.nodes if n.kind in ("source", "config", "manifest", "stage", "genome")
        )
        print(r.lineage.to_dot(include=keep))
        return 0
    print(f"结构血缘 DAG（{len(r.lineage.nodes)} 节点 / {len(r.lineage.edges)} 边）{_note(cached)}\n")
    print(r.lineage.to_ascii(limit=args.limit))
    print("\n  两条别处没有的图不变量：")
    print("    · genome:<name> 的父节点恰好是该模型声明的全部方言的 canonical 节点（跨方言同指纹画进了图里）；")
    print("    · mutations:<family> 的父节点恰好是该家族链上相邻版本的那些 diff（一个不多、一个不少）。")
    return 0


def cmd_acceptance(args: argparse.Namespace) -> int:
    r, cfg, cached = _result(args, fresh=True)
    checks = acceptance_checks(r, cfg)
    n_ok = sum(1 for c in checks if c.passed)
    print(f"验收标准逐条核对（{n_ok}/{len(checks)} 通过）{_note(cached)}\n")
    for i, c in enumerate(checks, 1):
        print(f"  [{'✅' if c.passed else '❌'}] {i:2d}. {c.name}")
        print(f"          {c.detail}")
    print()
    print("  README 的两条验收标准：")
    print("    ① 「对任意两个开源模型输出可复现的结构 Diff」→ diff / diff --all");
    print("    ② 「同一家族不同版本的结构演化可被自动识别为突变点」→ mutations")
    if args.json:
        _emit_json([c.to_dict() for c in checks])
    return 0 if n_ok == len(checks) else 1


def cmd_adversarial(args: argparse.Namespace) -> int:
    """对抗用例：**证明 canonicalize 会报错，而不是猜。** 不需要产物、不写盘。"""
    from .adversarial import render_cases, run_cases

    cfg = load_config(args.config, root=ROOT)
    cost = dict(cfg.get("cost") or {})
    results = run_cases(ROOT, cfg=cost)
    print(render_cases(results))
    if args.json:
        _emit_json([r.to_dict() for r in results])
    return 0 if all(r.ok for r in results) else 1


def cmd_config(args: argparse.Namespace) -> int:
    from .config import protocol_digest

    cfg = load_config(args.config, root=ROOT)
    print(f"生效配置（{cfg.get('_config_path')}）\n")
    for k in sorted(k for k in cfg if not k.startswith("_")):
        print(f"  {k:14s} {json.dumps(cfg[k], ensure_ascii=False)}")
    print(f"\n  协议摘要  : {protocol_digest(cfg)}")
    print("  优化目标  : 无（本项目不优化任何东西）")
    exp = expectations_of(cfg)
    if exp:
        print("  设计期望  :（允许被实测推翻）")
        for k, v in sorted(exp.items()):
            print(f"    {k:34s} {json.dumps(v, ensure_ascii=False)}")
    print("\n  默认值（DEFAULTS）:")
    for k, v in sorted(DEFAULTS.items()):
        print(f"    {k:14s} {json.dumps(v, ensure_ascii=False)}")
    return 0


# ---------------------------------------------------------------------- 写命令


def cmd_materialize(args: argparse.Namespace) -> int:
    """把每个模型按每个声明的方言渲染成清单。**这是唯一改输入数据的命令。**"""
    from .pipeline import materialize, manifest_path

    cfg = load_config(args.config, root=ROOT)
    zoo = load_zoo(ROOT)
    dialects = tuple(cfg.get("dialects") or ())
    written = materialize(zoo, root=ROOT, dialects=dialects)
    if not args.quiet:
        print(f"渲染了 {len(written)} 份清单 → data/zoo/<模型>/manifest.<方言>.json")
        for e in zoo:
            names = ", ".join(
                manifest_path(ROOT, e.name, d).name for d, _ in e.dialects if d in dialects
            )
            print(f"  {e.name:20s} {names}")
        print("\n  注：清单是从 config 标量渲染出来的，而 run 读清单时**不走那条路** ——")
        print("      两边互校发生在 canonicalize 里。渲染与解析共用一个函数就等于自问自答。")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load_config(args.config, root=ROOT)
    out = _out_dir(args)
    result = run_pipeline(cfg, root=ROOT)
    written = write_outputs(result, out, cfg=cfg)
    if not args.quiet:
        print(f"写入 {out}/：")
        for f in OUT_FILES:
            print(f"  {f}")
        print(f"  genomes/*.json（{len(result.genomes)} 个模型）")
        print(f"\n版本 {result.version.id}")
        mut = sum(len(t.mutation_points) for t in result.timelines)
        print(f"{len(result.names())} 个模型 / {len(result.pairs)} 个无序对 / {mut} 个突变点")
        print(f"产物共 {len(written)} 个文件。下一步：python main.py visualize 生成 report.html")
    return 0


def cmd_visualize(args: argparse.Namespace) -> int:
    from .visualize import render_html

    out = _out_dir(args)
    result = load_result(out)
    if result is None:
        print("产物不存在或不完整，无法生成 report.html（先跑 run）", file=sys.stderr)
        return 1
    path = out / EXTRA_FILES[0]
    path.write_text(render_html(result), encoding="utf-8")
    if not args.quiet:
        print(f"写入 {path}（自包含、无外部引用，可离线打开）")
    return 0


# ---------------------------------------------------------------------- 解析


#: 项目根：`src/{pkg}/cli.py` → 上三级。**所有相对路径默认值都锚到这里。**
_PKG_ROOT = Path(__file__).resolve().parent.parent.parent


def _anchor_args(args: argparse.Namespace) -> None:
    """把 `--config` / `--out` / `--root` 的**相对路径**锚定到项目根。

    契约是 `cd <项目目录> && python main.py ...`，从项目目录里跑时锚不锚一个样
    —— 缺陷因此**长期隐形**。一旦从别处调用（例如跨项目的批量核对
    `python L5-xxx/main.py check`），`"configs/default.json"` 就指向了 cwd 下一个
    不存在的位置：严格 `load_config` 直接崩，宽松的会**静默回退 DEFAULTS**
    （更糟 —— 不报错，只是悄悄换了一套口径）。

    **在 `main()` 里紧跟 `parse_args` 锚一次**，比在每个调用点各锚一次可靠：
    漏掉一个调用点，就是一条只在某种调用方式下才现形的路径。

    绝对路径原样使用。语义与 `pytest` 的 rootdir / `cargo` / `npm --prefix` 一致。
    """
    for name in ("config", "out", "root"):
        value = getattr(args, name, None)
        if isinstance(value, str) and value:
            p = Path(value)
            if not p.is_absolute():
                setattr(args, name, str(_PKG_ROOT / p))

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="modelgenomefactory",
        description="ModelGenomeFactory —— 把模型从「一堆代码」变成可比较的结构基因（不训练）",
    )
    p.add_argument("--config", default=DEFAULT_CONFIG, help="配置文件路径；相对路径按项目根解析")
    p.add_argument("--out", default="out", help="产物目录（只有 run / visualize 会写它）；相对路径按项目根解析")
    p.add_argument("--quiet", action="store_true", help="安静模式")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("summary", help="版本 / 计数 / 口径概览").set_defaults(func=cmd_summary)

    z = sub.add_parser("zoo", help="列出 zoo 里的公开模型")
    z.add_argument("--json", action="store_true")
    z.set_defaults(func=cmd_zoo)

    f = sub.add_parser("facts", help="一个模型的 config 归一化明细（每个数字从哪来）")
    f.add_argument("model")
    f.add_argument("--json", action="store_true")
    f.set_defaults(func=cmd_facts)

    t = sub.add_parser("tree", help="Module Tree 视图")
    t.add_argument("model")
    t.set_defaults(func=cmd_tree)

    g = sub.add_parser("graph", help="Graph 视图（边表 / DOT）")
    g.add_argument("model")
    g.add_argument("--dot", action="store_true")
    g.set_defaults(func=cmd_graph)

    s = sub.add_parser("tensors", help="Tensor Schema 视图")
    s.add_argument("model")
    s.add_argument("--json", action="store_true")
    s.set_defaults(func=cmd_tensors)

    gn = sub.add_parser("genome", help="一个模型的 Genome 全貌")
    gn.add_argument("model")
    gn.add_argument("--json", action="store_true")
    gn.set_defaults(func=cmd_genome)

    fp = sub.add_parser("fingerprint", help="两级指纹表")
    fp.add_argument("--json", action="store_true")
    fp.set_defaults(func=cmd_fingerprint)

    c = sub.add_parser("classes", help="同构类（骨架类 / 全同构类）")
    c.add_argument("--min-size", type=int, default=2)
    c.add_argument("--json", action="store_true")
    c.set_defaults(func=cmd_classes)

    m = sub.add_parser("matrix", help="对比矩阵")
    m.add_argument("--group", choices=["all", "skeleton", "scale", "cost"], default="all")
    m.add_argument("--limit", type=int, default=20)
    m.add_argument("--json", action="store_true")
    m.set_defaults(func=cmd_matrix)

    d = sub.add_parser("diff", help="两个模型的结构 Diff（或全部无序对）")
    d.add_argument("left", nargs="?", help="左模型")
    d.add_argument("right", nargs="?", help="右模型")
    d.add_argument("--all", action="store_true", help="打印全部无序对的判定")
    d.add_argument("--edits", action="store_true", help="打印全部编辑")
    d.add_argument("--limit", type=int, default=8)
    d.add_argument("--json", action="store_true")
    d.set_defaults(func=cmd_diff)

    mu = sub.add_parser("mutations", help="家族突变时间线（验收标准第二条）")
    mu.add_argument("family", nargs="?")
    mu.add_argument("--limit", type=int, default=4)
    mu.add_argument("--json", action="store_true")
    mu.set_defaults(func=cmd_mutations)

    ck = sub.add_parser("crosscheck", help="config ↔ 清单 两路互校")
    ck.add_argument("model", nargs="?")
    ck.add_argument("--json", action="store_true")
    ck.set_defaults(func=cmd_crosscheck)

    co = sub.add_parser("cost", help="代价三路互校")
    co.add_argument("model", nargs="?")
    co.add_argument("--json", action="store_true")
    co.set_defaults(func=cmd_cost)

    cv = sub.add_parser("coverage", help="config 键覆盖率（读不懂的键）")
    cv.add_argument("--json", action="store_true")
    cv.set_defaults(func=cmd_coverage)

    ru = sub.add_parser("rules", help="归一化规则表规模")
    ru.add_argument("--json", action="store_true")
    ru.set_defaults(func=cmd_rules)

    ln = sub.add_parser("lineage", help="结构血缘 DAG（ASCII / DOT）")
    ln.add_argument("--dot", action="store_true")
    ln.add_argument("--extract-only", action="store_true")
    ln.add_argument("--limit", type=int, default=30)
    ln.set_defaults(func=cmd_lineage)

    ac = sub.add_parser("acceptance", help="验收标准逐条核对")
    ac.add_argument("--json", action="store_true")
    ac.set_defaults(func=cmd_acceptance)

    ad = sub.add_parser("adversarial", help="对抗用例：证明 canonicalize 会报错而不是猜")
    ad.add_argument("--json", action="store_true")
    ad.set_defaults(func=cmd_adversarial)

    sub.add_parser("config", help="生效配置与设计期望").set_defaults(func=cmd_config)

    sub.add_parser("materialize", help="渲染命名清单到 data/zoo/（写输入数据）").set_defaults(func=cmd_materialize)
    sub.add_parser("run", help="跑完整流水线并落盘").set_defaults(func=cmd_run)
    sub.add_parser("visualize", help="生成自包含 report.html").set_defaults(func=cmd_visualize)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _anchor_args(args)
    try:
        if getattr(args, "json", False):
            # 命令内部照常 print 说明、照常 _emit_json 出数据；这里只负责
            # 把说明从 stdout 上摘掉，保证 `--json` 的输出能被 `jq` 直接吃。
            with _quiet_prose():
                return int(args.func(args))
        return int(args.func(args))
    except KeyboardInterrupt:  # pragma: no cover
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
