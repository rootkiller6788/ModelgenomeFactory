"""ModelGenomeFactory —— 把模型从「一堆代码」变成可比较的结构基因。

本项目**不训练、不优化**。它是一条结构测量流水线：读公开模型的 config 与参数清单，
归一到 Canonical Model IR，生成可指纹化的 Genome，再把 N 个模型两两比较成
「结构 Diff + 突变时间线」。

```
config.json ─┐
             ├─► RawStructure ─► CanonicalIR ─► Genome ─► Diff / 突变时间线
manifest ────┘   键名归一         结构归一       指纹化      两两比较
```

三个对外可用的入口：

- `pipeline.run_pipeline(cfg, root=...)` —— 跑完整分析，拿到 `PipelineResult`
- `pipeline.load_result(out_dir)` —— 从 `out/` 重建（只读命令走这条）
- `diff.structural_diff(a, b)` / `diff.apply(a, patch)` —— 结构差异与补丁施加

另有三个「交付物」模块：

- `report.render_report(result, cfg)` —— markdown 报告（`run` 写 `out/report.md`）
- `visualize.render_html(result)` —— 自包含 HTML（`visualize` 写 `out/report.html`）
- `adversarial.run_cases(root)` —— 对抗用例（`adversarial` 子命令；证明会报错而不是猜）

README 附录 A 有完整的 CLI 契约与产物清单，附录 B 是实测结果（含被实测**推翻**的
设计期望），附录 C 是踩过的坑。
"""

from __future__ import annotations

from .adversarial import Case, CaseResult, load_cases, run_cases
from .canonical import CanonicalError, canonicalize
from .compare import MATRIX_COLUMNS, isomorphism_classes, matrix_row, matrix_rows
from .config import DEFAULTS, load_config, objectives_of, protocol_digest, version_digest
from .cost import active_params, compute, kv_cache_bytes, params_from_formula, params_from_schema
from .dialects import (
    CONDITIONAL_RULES,
    LAYOUTS,
    NAME_RULES,
    ManifestError,
    parse_manifest,
    render_manifest,
    role_of,
    rule_coverage,
)
from .diff import apply, headline, invert, is_minimal, structural_diff, summary
from .extract import ZooEntry, extract_model, load_zoo
from .genome import build, fingerprint, load as load_genome, rebuild, shape_fingerprint
from .lineage import Lineage
from .mutations import build_timelines, classify_diff
from .pipeline import OUT_FILES, PipelineResult, load_result, run_pipeline
from .report import acceptance_checks, render_report
from .schema import (
    Boundary,
    CanonicalIR,
    Cost,
    Diff,
    Edit,
    FamilyTimeline,
    Genome,
    OpNode,
    RawStructure,
    TensorEntry,
    Version,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # 抽取
    "ZooEntry",
    "load_zoo",
    "extract_model",
    "RawStructure",
    # 归一
    "canonicalize",
    "CanonicalError",
    "ManifestError",
    "parse_manifest",
    "render_manifest",
    "NAME_RULES",
    "LAYOUTS",
    "CONDITIONAL_RULES",
    "rule_coverage",
    "role_of",
    # 指纹
    "build",
    "rebuild",
    "fingerprint",
    "shape_fingerprint",
    "Genome",
    "CanonicalIR",
    "OpNode",
    "TensorEntry",
    "Cost",
    # 代价
    "compute",
    "params_from_schema",
    "params_from_formula",
    "kv_cache_bytes",
    "active_params",
    # 比较
    "structural_diff",
    "apply",
    "invert",
    "is_minimal",
    "summary",
    "headline",
    "Diff",
    "Edit",
    "Boundary",
    "FamilyTimeline",
    "build_timelines",
    "classify_diff",
    "matrix_row",
    "matrix_rows",
    "MATRIX_COLUMNS",
    "isomorphism_classes",
    # 编排
    "run_pipeline",
    "load_result",
    "PipelineResult",
    "OUT_FILES",
    "load_config",
    "DEFAULTS",
    "objectives_of",
    "protocol_digest",
    "version_digest",
    "Lineage",
    "Version",
    "load_genome",
    # 交付物
    "render_report",
    "acceptance_checks",
    "run_cases",
    "load_cases",
    "Case",
    "CaseResult",
]
