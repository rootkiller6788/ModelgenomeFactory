"""对抗用例：**证明 `canonicalize` 会报错，而不是猜。**

这个项目最危险的失败模式不是「报错」，而是「读不懂但假装读懂了」——
一个宽度对不上的张量被当成对的读进去，然后报告里出现一个没人会去核对的数字。
`canonical.py` 里有一整套宽度互校，但它们只有在**被喂进坏输入**时才被验证过。

`data/zoo/_adversarial/` 里放两类东西：

- **合法但渲染器产不出来**（`gpt2.shuffled` / `mixtral-8x7b.vllm_shuffled`）——
  张量顺序完全倒序。渲染器永远按名字排序，所以这两份清单不可能是它写的。
  它们**必须被接受**，而且指纹必须与正式清单**逐位相同** ——
  这一条盯的是「解析器是不是只认得自己渲染出来的那一种写法」。
- **必须报错**（7 个）—— 每一个都对准一条具体的互校，`cases.json` 里写了
  **期望的错误信息片段**。只断言「抛了异常」是不够的：一个 `KeyError` 也能让测试变绿，
  而它可能来自任何地方。

用例表在 `data/zoo/_adversarial/cases.json` 里，CLI（`adversarial` 子命令）与
`tests/test_adversarial.py` 共用它 —— 于是「命令行说全过的」与「测试说全过的」
不可能是两套结论。

## 已知边界（写出来，别装作没有）

本项目把**融合张量**写成「沿最后一维拼接」：vLLM 的 `qkv_proj.weight` 是
`(hidden, (n_heads+2·n_kv)·head_dim)`。真实 vLLM checkpoint 里是沿第 0 维
（`(6144, 4096)` 而非 `(4096, 6144)`）。所以把真实布局喂进来会**报错**——
`llama-3-8b.qkv_axis` 这条用例钉的就是它。这是「拒绝而不是猜」的正确行为，
但它同时是一条**已知的接不上真实 checkpoint 的缝**，附录 B 里记着。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .canonical import CanonicalError, canonicalize
from .dialects import ManifestError
from .extract import extract_model, load_zoo
from .genome import build, fingerprint
from .schema import Genome

__all__ = ["Case", "CaseResult", "ADVERSARIAL_DIR", "load_cases", "run_cases", "render_cases"]

#: 对抗清单目录（相对项目根）。
ADVERSARIAL_DIR = "data/zoo/_adversarial"

#: 合法用例用来对照的正式清单：`(模型, 方言)`。
_OFFICIAL = {
    "gpt2.shuffled.json": ("gpt2", "hf"),
    "mixtral-8x7b.vllm_shuffled.json": ("mixtral-8x7b", "vllm"),
}


@dataclass(frozen=True)
class Case:
    file: str
    model: str
    verdict: str          # "accept" | "reject"
    why: str
    expect: str = ""      # verdict == "reject" 时：错误信息里必须出现的片段

    def to_dict(self) -> dict[str, Any]:
        return {"file": self.file, "model": self.model, "verdict": self.verdict,
                "why": self.why, "expect": self.expect}


@dataclass(frozen=True)
class CaseResult:
    case: Case
    ok: bool
    detail: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.case.to_dict(), "ok": self.ok, "detail": self.detail}


def load_cases(root: Path | str) -> tuple[Case, ...]:
    path = Path(root) / ADVERSARIAL_DIR / "cases.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(Case(**c) for c in raw["cases"])


def _genome_of(entry: Any, manifest: dict[str, Any], dialect: str, cfg: dict[str, Any]) -> Genome:
    raw = extract_model(entry)
    ir = canonicalize(raw, manifest, name=entry.name, dialect=dialect)
    return build(
        ir,
        raw,
        seq_len=int(cfg["seq_len"]),
        batch=int(cfg["batch"]),
        dtype_bytes=int(cfg["dtype_bytes"]),
        params_published_b=entry.params_b,
    )


def run_cases(root: Path | str, cfg: dict[str, Any] | None = None) -> tuple[CaseResult, ...]:
    """跑全部对抗用例。**不写盘、不改任何输入。**"""
    root = Path(root)
    cfg = dict(cfg or {"seq_len": 2048, "batch": 1, "dtype_bytes": 2})
    entries = {e.name: e for e in load_zoo(root)}
    out: list[CaseResult] = []

    for case in load_cases(root):
        manifest = json.loads((root / ADVERSARIAL_DIR / case.file).read_text(encoding="utf-8"))
        entry = entries[case.model]
        dialect = manifest.get("dialect", "adversarial")
        try:
            g = _genome_of(entry, manifest, dialect, cfg)
        except (CanonicalError, ManifestError, ValueError, KeyError) as exc:
            msg = str(exc)
            if case.verdict == "reject":
                hit = case.expect in msg
                out.append(
                    CaseResult(
                        case,
                        hit,
                        f"按预期报错，信息命中 {case.expect!r}：{msg[:160]}"
                        if hit
                        else f"**报错了，但原因不对**：期望信息含 {case.expect!r}，实际是 {msg[:160]}",
                    )
                )
            else:
                out.append(CaseResult(case, False, f"**本该被接受，却报错**：{msg[:200]}"))
            continue

        if case.verdict == "reject":
            out.append(CaseResult(case, False, "**本该报错，却被静默接受了**（这正是最危险的失败模式）"))
            continue

        # 合法用例：指纹必须与正式清单逐位相同
        official_model, official_dialect = _OFFICIAL[case.file]
        official = json.loads(
            (root / "data" / "zoo" / official_model / f"manifest.{official_dialect}.json").read_text(encoding="utf-8")
        )
        g_ref = _genome_of(entries[official_model], official, official_dialect, cfg)
        same_fp = fingerprint(g) == fingerprint(g_ref)
        same_shape = g_ref.primary().get("topology.n_layers") == g.primary().get("topology.n_layers")
        out.append(
            CaseResult(
                case,
                same_fp and same_shape,
                f"接受；指纹 {fingerprint(g)[:12]} vs 正式清单 {fingerprint(g_ref)[:12]} —— "
                f"{'逐位相同' if same_fp else '**不同**'}（层数 {g.primary().get('topology.n_layers')}）",
            )
        )
    return tuple(out)


def render_cases(results: tuple[CaseResult, ...]) -> str:
    lines = [
        f"对抗用例（{sum(1 for r in results if r.ok)}/{len(results)} 通过）",
        "",
        "  合法但渲染器产不出来（必须被接受，且指纹与正式清单逐位相同）：",
    ]
    for r in results:
        if r.case.verdict == "accept":
            lines.append(f"    [{'✅' if r.ok else '❌'}] {r.case.file}")
            lines.append(f"          {r.case.why}")
            lines.append(f"          {r.detail}")
    lines.append("")
    lines.append("  必须报错（每一个都对准一条具体的互校）：")
    for r in results:
        if r.case.verdict == "reject":
            lines.append(f"    [{'✅' if r.ok else '❌'}] {r.case.file}　期望含 {r.case.expect!r}")
            lines.append(f"          {r.case.why}")
            lines.append(f"          {r.detail}")
    lines.append("")
    lines.append("  注：只断言「抛了异常」是不够的 —— 一个 KeyError 也能让测试变绿，")
    lines.append("      而它可能来自任何地方。所以每条用例都写了期望的错误信息片段。")
    return "\n".join(lines)
