"""编排与产物读写。

流水线本身很短 —— 这个项目的复杂度全在「归一化」与「比较」两处：

```
zoo.json ──► 每个模型的 config ──► RawStructure（键名归一）
                    │
                    └─► 每个方言的清单 ──► CanonicalIR（参数名归一）
                                              │
                                    ┌─────────┴─────────┐
                                    ▼                   ▼
                              Genome + 指纹       config↔清单 互校
                                    │
                          ┌─────────┴──────────┐
                          ▼                    ▼
                   N² 两两 Diff           同构类 / 对比矩阵
                          │
                          ▼
                    家族突变时间线
```

## 只读命令从产物重建，绝不重跑

`load_result` 是只读子命令的唯一入口。它从 `out/` 读回 genome / diff / 时间线 / 矩阵，
**不碰 data/、不写盘**。这条纪律有测试用 md5 指纹钉住：跑完所有只读命令后
`out/` 的每个文件都必须逐位不变。

`report.html` 不在 `OUT_FILES` 里：它由 `visualize` 单独写。把它算进 `run` 的产物
会让「`run` 之后 report.html 也是新的」变成一个隐式约定，而 `run` 并不需要它。
"""

from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import compare as compare_mod
from . import cost as cost_mod
from .canonical import CanonicalError, canonicalize
from .config import expectations_of, protocol_digest, version_digest
from .dialects import CONDITIONAL_RULES, rule_coverage
from .diff import structural_diff as diff_pair
from .diff import headline, summary as diff_summary
from .extract import ZooEntry, coverage_report, extract_model, load_config_file, load_zoo
from .genome import build, dump as dump_genome, fingerprint, load as load_genome, shape_fingerprint, write as write_genome
from .lineage import (
    COLLECT_NODE_ID,
    Lineage,
    add_diff_nodes,
    attach_mutation_parents,
    build_lineage,
    dumps as dumps_lineage,
    diff_node,
    pair_key,
)
from .mutations import build_timelines, classify_diff
from .schema import CanonicalIR, Diff, FamilyTimeline, Genome, RawStructure, Version, utcnow

__all__ = [
    "OUT_FILES",
    "EXTRA_FILES",
    "MANIFEST_DIR",
    "PipelineResult",
    "run_pipeline",
    "materialize",
    "write_outputs",
    "load_result",
    "missing_manifests",
]

#: `run` 写出的产物。
OUT_FILES = (
    "report.md",
    "matrix.csv",
    "classes.json",
    "diffs.json",
    "mutations.json",
    "rules.json",
    "lineage.json",
    "lineage.dot",
    "lineage_extract.dot",
    "version.json",
)

#: `visualize` 单独写的产物。
EXTRA_FILES = ("report.html",)

#: `genomes/` 子目录：每个模型一份可重新载入的 Genome（本项目的可交付物）。
GENOME_DIR = "genomes"

MANIFEST_DIR = "data/zoo"


# ---------------------------------------------------------------- 结果对象


@dataclass(frozen=True)
class PipelineResult:
    """一次完整分析的全部结果。**只读命令全部从这里取数。**"""

    zoo: tuple[ZooEntry, ...]
    raws: Mapping[str, RawStructure]
    canonical: Mapping[tuple[str, str], CanonicalIR]
    genomes: Mapping[str, Genome]
    pairs: Mapping[str, Diff]
    timelines: tuple[FamilyTimeline, ...]
    classes: Mapping[str, Mapping[str, list[str]]]
    matrix: tuple[dict[str, Any], ...]
    lineage: Lineage
    version: Version
    coverage: Mapping[str, Any] = field(default_factory=dict)
    rules: Mapping[str, Any] = field(default_factory=dict)
    expectations: Mapping[str, Any] = field(default_factory=dict)

    # -- 查询 --------------------------------------------------------------

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self.genomes))

    def entry(self, name: str) -> ZooEntry:
        for e in self.zoo:
            if e.name == name:
                return e
        raise KeyError(f"zoo 里没有 {name!r}")

    def genome(self, name: str) -> Genome:
        if name not in self.genomes:
            raise KeyError(f"没有 genome {name!r}")
        return self.genomes[name]

    def ir(self, name: str, dialect: str) -> CanonicalIR:
        key = (name, dialect)
        if key not in self.canonical:
            raise KeyError(f"没有 {name!r} 的 {dialect!r} 方言 IR")
        return self.canonical[key]

    def dialects_of(self, name: str) -> tuple[str, ...]:
        return tuple(d for (n, d) in sorted(self.canonical) if n == name)

    def diff(self, a: str, b: str) -> Diff:
        """任意有序对的结构 Diff。

        产物里每个**无序对**只存一份（`pair_key` 排序后的方向）；反方向靠 `invert`
        得到。于是「可逆性」不是一条测试，而是**只读命令依赖的性质** ——
        它要是坏了，`diff b a` 立刻报错，而不是悄悄给出一个不同的答案。
        """
        if a == b:
            g = self.genome(a)
            return diff_pair(g, g)
        key = pair_key(a, b)
        if key not in self.pairs:
            raise KeyError(f"产物里没有 {key!r} 这一对")
        d = self.pairs[key]
        return d if d.left == a else d.inverted()

    def pairs_ordered(self) -> tuple[Diff, ...]:
        return tuple(self.pairs[k] for k in sorted(self.pairs))

    def families(self) -> tuple[str, ...]:
        return tuple(sorted({e.family for e in self.zoo}))

    def timeline(self, family: str) -> FamilyTimeline:
        for tl in self.timelines:
            if tl.family == family:
                return tl
        raise KeyError(f"没有家族 {family!r}")

    def crosschecks(self, name: str) -> tuple[tuple[str, str, str], ...]:
        out: list[tuple[str, str, str]] = []
        for (n, _d), ir in sorted(self.canonical.items()):
            if n == name:
                out.extend(ir.crosschecks)
        return tuple(out)

    def fingerprint_of(self, name: str) -> str:
        return fingerprint(self.genome(name))

    def shape_fingerprint_of(self, name: str) -> str:
        return shape_fingerprint(self.genome(name))

    def matrix_row(self, name: str) -> dict[str, Any]:
        for r in self.matrix:
            if r["name"] == name:
                return r
        raise KeyError(name)

    def active_params(self, name: str) -> int:
        return cost_mod.active_params(self.genome(name))


# ---------------------------------------------------------------- 清单


def manifest_path(root: Path | str, name: str, dialect: str) -> Path:
    return Path(root) / MANIFEST_DIR / name / f"manifest.{dialect}.json"


def missing_manifests(root: Path | str, zoo: Sequence[ZooEntry], dialects: Sequence[str]) -> list[str]:
    """缺哪些清单。缺了就**明说**，不要让 `run` 跑到一半才炸。"""
    out: list[str] = []
    for e in zoo:
        for d, _layout in e.dialects:
            if d not in dialects:
                continue
            p = manifest_path(root, e.name, d)
            if not p.exists():
                out.append(str(p))
    return out


def materialize(
    zoo: Sequence[ZooEntry],
    *,
    root: Path | str,
    dialects: Sequence[str],
) -> list[Path]:
    """把每个模型按每个声明的方言渲染成清单，写到 `data/zoo/<name>/manifest.<d>.json`。

    这是**写命令**（它改的是输入数据，不是产物），所以它是独立子命令而不是 `run`
    的一部分。渲染用的结构来自 config 标量（`extract.expected_tensors`），
    而 `run` 读清单时**不调用**那条路 —— 两边互校发生在 `canonicalize` 里。
    """
    from .dialects import render_manifest
    from .extract import block_groups_for, expected_tensors, position_kind_of

    written: list[Path] = []
    for e in zoo:
        raw = extract_model(e)
        cfg_raw = load_config_file(e.config_path)
        tensors = expected_tensors(raw, position_kind=position_kind_of(cfg_raw))
        groups = block_groups_for(raw)
        n_experts = int(raw.get("n_experts") or 0)
        for d, layout in e.dialects:
            if d not in dialects:
                continue
            manifest = render_manifest(
                tensors,
                layout=layout,
                model=e.name,
                dialect=d,
                block_groups=groups,
                n_experts=n_experts,
            )
            p = manifest_path(root, e.name, d)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            written.append(p)
    return written


# ---------------------------------------------------------------- 跑


def run_pipeline(cfg: Mapping[str, Any], *, root: Path | str) -> PipelineResult:
    root = Path(root)
    zoo = load_zoo(root)
    dialects = tuple(cfg.get("dialects") or ())
    cost_cfg = cfg.get("cost") or {}
    seq_len = int(cost_cfg["seq_len"])
    batch = int(cost_cfg["batch"])
    dtype_bytes = int(cost_cfg["dtype_bytes"])

    missing = missing_manifests(root, zoo, dialects)
    if missing:
        raise FileNotFoundError(
            "缺清单：\n  - " + "\n  - ".join(missing[:5]) + ("\n  …" if len(missing) > 5 else "")
            + "\n先跑 `python main.py materialize`。"
        )

    raws: dict[str, RawStructure] = {}
    canonical: dict[tuple[str, str], CanonicalIR] = {}
    genomes: dict[str, Genome] = {}
    manifests: list[Mapping[str, Any]] = []
    inputs: list[tuple[str, Any]] = [("zoo.json", json.loads((root / MANIFEST_DIR / "zoo.json").read_text(encoding="utf-8")))]

    for entry in zoo:
        raw = extract_model(entry)
        raws[entry.name] = raw
        inputs.append((f"config:{entry.name}", load_config_file(entry.config_path)))
        fps: dict[str, str] = {}
        for dialect, _layout in entry.dialects:
            if dialect not in dialects:
                continue
            manifest = json.loads(manifest_path(root, entry.name, dialect).read_text(encoding="utf-8"))
            inputs.append((f"manifest:{entry.name}:{dialect}", manifest))
            manifests.append(manifest)
            ir = canonicalize(raw, manifest, name=entry.name, dialect=dialect)
            canonical[(entry.name, dialect)] = ir
            g = build(
                ir,
                raw,
                seq_len=seq_len,
                batch=batch,
                dtype_bytes=dtype_bytes,
                params_published_b=entry.params_b,
            )
            fp = fingerprint(g)
            fps[dialect] = fp
            genomes[entry.name] = g
        if len(set(fps.values())) > 1:
            # 这一条要是能触发，整个项目的前提就塌了。
            raise CanonicalError(
                f"{entry.name}: 不同命名方言给出的指纹不一致 —— {fps}。"
                "归一化漏了某条路径，或某条规则把结构读错了。"
            )
        if len(fps) < int((cfg.get("acceptance") or {}).get("min_dialects_per_model", 2)):
            raise CanonicalError(f"{entry.name}: 只有 {len(fps)} 种方言，少于要求的下限")

    # 两两 diff（无序对只存一份）
    pairs: dict[str, Diff] = {}
    names = sorted(genomes)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            pairs[pair_key(a, b)] = diff_pair(genomes[a], genomes[b])

    members_by_family: dict[str, list[Genome]] = {}
    for entry in sorted(zoo, key=lambda e: (e.family, e.chain)):
        members_by_family.setdefault(entry.family, []).append(genomes[entry.name])
    timelines = build_timelines(members_by_family)

    classes = compare_mod.isomorphism_classes([genomes[n] for n in names])
    meta = {e.name: e.to_dict() for e in zoo}
    matrix = compare_mod.matrix_rows([genomes[n] for n in names], meta)

    lineage = build_lineage(
        zoo_names=names,
        dialects_by_model={e.name: [d for d, _ in e.dialects if d in dialects] for e in zoo},
        families={tl.family: tl.members for tl in timelines},
        n_pairs=len(pairs),
        n_models=len(names),
    )
    add_diff_nodes(lineage, [(d.left, d.right) for d in pairs.values()], fingerprint_of={})
    for tl in timelines:
        attach_mutation_parents(
            lineage, tl.family, [(b.left, b.right) for b in tl.boundaries]
        )

    digest = version_digest(cfg, inputs)
    version = Version(
        id=f"v1-{digest[:12]}",
        digest=digest,
        created_at=utcnow(),
        protocol_digest=protocol_digest(cfg),
        counts=(
            ("models", len(names)),
            ("dialects", sum(len(result_dialects(genomes, e, dialects)) for e in zoo)),
            ("pairs", len(pairs)),
            ("genomes", len(genomes)),
            ("lineage_nodes", len(lineage.nodes)),
            ("lineage_edges", len(lineage.edges)),
        ),
        note="本项目不训练、不优化：产物是结构证据，不是模型。",
    )

    return PipelineResult(
        zoo=zoo,
        raws=raws,
        canonical=canonical,
        genomes=genomes,
        pairs=pairs,
        timelines=timelines,
        classes=classes,
        matrix=tuple(matrix),
        lineage=lineage,
        version=version,
        coverage=coverage_report(raws.values()),
        rules=rule_coverage(manifests),
        expectations=expectations_of(cfg),
    )


def result_dialects(genomes: Mapping[str, Genome], entry: ZooEntry, dialects: Sequence[str]) -> tuple[str, ...]:
    return tuple(d for d, _ in entry.dialects if d in dialects)


# ---------------------------------------------------------------- 产物


def _csv(rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(list(columns))
    for r in rows:
        w.writerow([_cell(r.get(c)) for c in columns])
    return buf.getvalue()


def _cell(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, float):
        return f"{v:.6g}"
    return "" if v is None else str(v)


def write_outputs(
    result: PipelineResult,
    out_dir: Path | str,
    *,
    cfg: Mapping[str, Any] | None = None,
) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name in result.names():
        written.append(write_genome(result.genome(name), out / GENOME_DIR / f"{name}.json"))

    (out / "matrix.csv").write_text(_csv(result.matrix, compare_mod.MATRIX_COLUMNS), encoding="utf-8")
    written.append(out / "matrix.csv")

    classes = {
        "shape": {k: v for k, v in sorted(result.classes["shape"].items())},
        "full": {k: v for k, v in sorted(result.classes["full"].items())},
        "table": compare_mod.class_table([result.genome(n) for n in result.names()], min_size=2),
    }
    _write_json(out / "classes.json", classes)
    written.append(out / "classes.json")

    diffs = {
        "pairs": [
            {**diff_summary(d), "headline": list(headline(d, limit=6))}
            for d in result.pairs_ordered()
        ],
        "detail": {k: result.pairs[k].to_dict() for k in sorted(result.pairs)},
    }
    _write_json(out / "diffs.json", diffs)
    written.append(out / "diffs.json")

    _write_json(
        out / "mutations.json",
        {
            "timelines": [tl.to_dict() for tl in result.timelines],
            "crosscheck": {
                "families": len(result.timelines),
                "boundaries": sum(len(tl.boundaries) for tl in result.timelines),
                "mutation_points": sum(len(tl.mutation_points) for tl in result.timelines),
            },
        },
    )
    written.append(out / "mutations.json")

    # 命名规则的覆盖率：**实测 / 条件 / 死规则**三栏。死规则恒为空 —— 它不为空就说明
    # 规则表里有一条谁也没验过的猜测，报告会把它点名。
    _write_json(out / "rules.json", {"layouts": dict(result.rules), "conditional": dict(CONDITIONAL_RULES)})
    written.append(out / "rules.json")

    (out / "lineage.json").write_text(dumps_lineage(result.lineage), encoding="utf-8")
    written.append(out / "lineage.json")
    (out / "lineage.dot").write_text(result.lineage.to_dot(), encoding="utf-8")
    written.append(out / "lineage.dot")
    extract_only = {
        n.id
        for n in result.lineage.nodes
        if n.kind in ("source", "config", "manifest", "stage", "genome")
    }
    (out / "lineage_extract.dot").write_text(
        result.lineage.to_dot(include=sorted(extract_only)), encoding="utf-8"
    )
    written.append(out / "lineage_extract.dot")

    _write_json(out / "version.json", result.version.to_dict())
    written.append(out / "version.json")

    from .report import render_report

    (out / "report.md").write_text(render_report(result, cfg), encoding="utf-8")
    written.append(out / "report.md")
    return written


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_result(out_dir: Path | str) -> PipelineResult | None:
    """从 `out/` 重建结果。缺任何一个产物都返回 `None`（**不抛异常、不重跑**）。

    返回 `None` 而不是抛异常，是为了让 `cli` 能干净地决定「退回在内存里跑一遍」。
    一个只读命令因为「少了一个文件」而崩掉，用户会以为工具坏了。
    """
    out = Path(out_dir)
    if not out.is_dir():
        return None
    needed = (
        "matrix.csv",
        "classes.json",
        "diffs.json",
        "mutations.json",
        "rules.json",
        "lineage.json",
        "version.json",
    )
    for f in needed:
        if not (out / f).is_file():
            return None
    gdir = out / GENOME_DIR
    if not gdir.is_dir() or not any(gdir.glob("*.json")):
        return None

    genomes: dict[str, Genome] = {}
    for p in sorted(gdir.glob("*.json")):
        genomes[p.stem] = load_genome(json.loads(p.read_text(encoding="utf-8")))
    if not genomes:
        return None

    classes = json.loads((out / "classes.json").read_text(encoding="utf-8"))
    diffs_raw = json.loads((out / "diffs.json").read_text(encoding="utf-8"))
    mutations = json.loads((out / "mutations.json").read_text(encoding="utf-8"))
    version = json.loads((out / "version.json").read_text(encoding="utf-8"))
    rules = json.loads((out / "rules.json").read_text(encoding="utf-8"))
    matrix = _read_matrix(out / "matrix.csv")

    pairs = {k: _diff_from_dict(v) for k, v in diffs_raw.get("detail", {}).items()}
    timelines = tuple(_timeline_from_dict(t) for t in mutations.get("timelines", []))

    from .lineage import Lineage

    lineage = Lineage.from_dict(json.loads((out / "lineage.json").read_text(encoding="utf-8")))

    return PipelineResult(
        zoo=(),
        raws={},
        canonical={},
        genomes=genomes,
        pairs=pairs,
        timelines=timelines,
        classes={k: v for k, v in classes.items() if k in ("shape", "full")},
        matrix=tuple(matrix),
        lineage=lineage,
        version=Version(
            id=version["id"],
            digest=version["digest"],
            created_at=version["created_at"],
            protocol_digest=version["protocol_digest"],
            counts=tuple(sorted(version.get("counts", {}).items())),
            note=version.get("note", ""),
        ),
        coverage={"unknown": {}, "clean": True, "n_models": len(genomes), "from_products": True},
        rules=rules.get("layouts", {}),
        expectations={},
    )


#: CSV 里的列类型。**不还原类型的话，「从产物读的矩阵」与「内存里算的矩阵」
#: 会处处不相等** —— 而它们本该逐位相同。`csv` 模块没有类型，所以类型表必须显式写出来；
#: 漏一列就会让那一列静默地退化成字符串（`"4096" != 4096`），
#: 而 `tests/test_pipeline_cli.py` 的「产物 ≡ 内存」测试会立刻抓到。
_INT_COLUMNS = frozenset(compare_mod.SCALE_COLUMNS) | frozenset(compare_mod.COST_COLUMNS) | {"chain"}
_BOOL_COLUMNS = frozenset({"tie_embeddings", "has_moe", "has_shared_experts"})


def _cell_from_str(col: str, v: str) -> Any:
    if col in _INT_COLUMNS:
        return int(v) if v != "" else 0
    if col in _BOOL_COLUMNS:
        return v == "true"
    return v


def _read_matrix(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return [
            {k: _cell_from_str(k, v) for k, v in row.items()}
            for row in csv.DictReader(fh)
        ]


def _diff_from_dict(d: Mapping[str, Any]) -> Diff:
    from .diff import rehydrate_payload
    from .schema import Edit, rehydrate

    edits = tuple(
        Edit(
            e["kind"],
            e["addr"],
            rehydrate(e["addr"], rehydrate_payload(e["old"])),
            rehydrate(e["addr"], rehydrate_payload(e["new"])),
        )
        for e in d["edits"]
    )
    return Diff(
        left=d["left"],
        right=d["right"],
        edits=edits,
        left_fingerprint=d["left_fingerprint"],
        right_fingerprint=d["right_fingerprint"],
        left_shape_fingerprint=d["left_shape_fingerprint"],
        right_shape_fingerprint=d["right_shape_fingerprint"],
    )


def _timeline_from_dict(d: Mapping[str, Any]) -> FamilyTimeline:
    from .schema import Boundary

    bs = tuple(
        Boundary(
            family=b["family"],
            left=b["left"],
            right=b["right"],
            index=b["index"],
            kind=b["kind"],
            coordinates=tuple(b["coordinates"]),
            n_edits=b["n_edits"],
            n_scale=b["n_scale"],
            n_structural=b["n_structural"],
            fingerprint_changed=b["fingerprint_changed"],
            shape_fingerprint_changed=b["shape_fingerprint_changed"],
            headline=tuple(b.get("headline", ())),
        )
        for b in d["boundaries"]
    )
    return FamilyTimeline(
        family=d["family"],
        members=tuple(d["members"]),
        boundaries=bs,
        note=d.get("note", ""),
    )
