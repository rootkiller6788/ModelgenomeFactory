"""自包含 HTML：内联 SVG 的**结构差异矩阵** + **家族突变时间线**。**不引用任何外部资源。**

**为什么是自包含的。** 报告要能当文件发出去、离线打开、几年后还能看。任何
``http://`` 引用都会在某一天变成坏链，让「证据」失效。所以图全部由代码生成内联 SVG，
字体只用系统栈。

两张图各回答一个问题：

- **结构差异矩阵**（N×N）—— 验收标准第一条的可视化：任意两个模型之间是零差异 / 量变 / 突变，
  一眼看全。对角线是同一个模型，右上与左下必然对称（可逆性），这一点在图上直接可见。
- **家族突变时间线** —— 验收标准第二条的可视化：链上每个版本边界的判定，
  红点就是自动识别出来的突变点。

配色是固定的深色系（不跟随 IDE 主题）：**报告文件要能脱离主题独立存在**。
"""

from __future__ import annotations

import html
from typing import Any, Sequence

_BG = "#0f1419"
_PANEL = "#161d26"
_FG = "#e6edf3"
_DIM = "#8b98a5"
_ACCENT = "#4ea1ff"
_WARN = "#ff7b72"
_OK = "#3fb950"
_MUT = "#f778ba"
_GRID = "#2a3441"


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _human(n: int) -> str:
    for unit, scale in (("T", 10**12), ("B", 10**9), ("M", 10**6), ("K", 10**3)):
        if abs(n) >= scale:
            return f"{n / scale:.2f}{unit}"
    return str(n)


# ---------------------------------------------------------------------- 差异矩阵


def diff_matrix_svg(result: Any, *, cell: int = 34, label: int = 7) -> str:
    """N×N 结构差异矩阵。**验收标准第一条的可视化。**

    配色：绿 = 零差异、蓝 = 只有规模变化、粉 = 结构突变、深灰 = 对角线。
    蓝格的**深浅**按编辑条数取对数，所以「只差 eps」和「改了 20 处规模」不会看起来一样。
    """
    names = list(result.names())
    n = len(names)
    if n == 0:
        return "<svg/>"
    pad_l, pad_t = 74, 56
    width = pad_l + n * cell + 10
    height = pad_t + n * cell + 10
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">'
    ]
    for j, name in enumerate(names):
        x = pad_l + j * cell + cell / 2
        parts.append(
            f'<text x="{x:.1f}" y="{pad_t - 8}" fill="{_DIM}" font-size="8" '
            f'text-anchor="start" transform="rotate(-60 {x:.1f} {pad_t - 8})">{_esc(name)}</text>'
        )
    for i, name in enumerate(names):
        y = pad_t + i * cell + cell / 2 + 3
        parts.append(
            f'<text x="{pad_l - 6}" y="{y:.1f}" fill="{_DIM}" font-size="8" '
            f'text-anchor="end">{_esc(name)}</text>'
        )
    max_edits = max((len(d.edits) for d in result.pairs_ordered()), default=1) or 1
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            x, y = pad_l + j * cell, pad_t + i * cell
            if i == j:
                fill, text, tcol = _PANEL, "·", _DIM
            else:
                d = result.diff(a, b)
                if d.is_empty:
                    fill, text, tcol = _OK, "0", "#0f1419"
                elif d.is_mutation:
                    fill, text, tcol = _MUT, str(len(d.edits)), "#0f1419"
                else:
                    # 量变：按编辑条数的对数取深浅，避免「差 eps」与「差 20 处」同色。
                    import math

                    t = math.log1p(len(d.edits)) / math.log1p(max_edits)
                    fill = f"rgba(78,161,255,{0.20 + 0.65 * t:.2f})"
                    text, tcol = str(len(d.edits)), _FG
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" fill="{fill}" rx="3"/>'
            )
            parts.append(
                f'<text x="{x + (cell - 2) / 2:.1f}" y="{y + (cell - 2) / 2 + 2.6:.1f}" '
                f'fill="{tcol}" font-size="{label}" text-anchor="middle">{text}</text>'
            )
    # 图例
    lx, ly = pad_l, height - 6
    for color, text in ((_OK, "零差异"), (_ACCENT, "量变（仅规模）"), (_MUT, "突变（骨架变了）")):
        parts.append(f'<rect x="{lx}" y="{ly - 8}" width="9" height="9" fill="{color}" rx="2"/>')
        parts.append(f'<text x="{lx + 13}" y="{ly}" fill="{_DIM}" font-size="8">{text}</text>')
        lx += 88
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------- 时间线


def timeline_svg(timeline: Any, *, width: int = 620, row: int = 54) -> str:
    """一条家族链的演化时间线。**红点 = 自动识别出的突变点。**"""
    members = list(timeline.members)
    if not members:
        return "<svg/>"
    pad = 30
    height = row + 34
    n = len(members)
    xs = [pad + i * (width - 2 * pad) / max(1, n - 1) for i in range(n)]
    y = row / 2 + 8
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px" '
        f'xmlns="http://www.w3.org/2000/svg" font-family="ui-monospace,monospace">',
        f'<line x1="{pad}" y1="{y}" x2="{width - pad}" y2="{y}" stroke="{_GRID}" stroke-width="2"/>',
    ]
    for i, m in enumerate(members):
        x = xs[i]
        parts.append(f'<circle cx="{x:.1f}" cy="{y}" r="6" fill="{_ACCENT}"/>')
        parts.append(
            f'<text x="{x:.1f}" y="{y + 22}" fill="{_FG}" font-size="9" text-anchor="middle">{_esc(m)}</text>'
        )
    for b in timeline.boundaries:
        i = b.index - 1
        if i + 1 >= len(xs):
            continue
        x0, x1 = xs[i], xs[i + 1]
        xm = (x0 + x1) / 2
        if b.kind == "mutation":
            color, r = _MUT, 6
        elif b.kind == "scale":
            color, r = _ACCENT, 3.5
        else:
            color, r = _OK, 3
        parts.append(f'<circle cx="{xm:.1f}" cy="{y}" r="{r}" fill="{color}"/>')
        mark = {"mutation": "突变", "scale": "量变", "same": "零差异"}[b.kind]
        parts.append(
            f'<text x="{xm:.1f}" y="{y - 12}" fill="{color}" font-size="8" '
            f'text-anchor="middle">{mark} {b.n_edits}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------- 页面


def _card(title: str, body: str, ok: bool | None = None) -> str:
    border = _OK if ok else (_WARN if ok is False else _DIM)
    badge = "" if ok is None else f'<span style="color:{border}">{"通过" if ok else "未通过"}</span>'
    return (
        f'<div style="background:{_PANEL};border-left:3px solid {border};padding:10px 14px;'
        f'margin:8px 0;border-radius:4px">'
        f'<div style="font-weight:600;margin-bottom:4px">{_esc(title)} {badge}</div>'
        f'<div style="color:{_DIM};font-size:13px;line-height:1.6">{body}</div></div>'
    )


def _table(header: Sequence[str], rows: Sequence[Sequence[Any]], *, size: int = 12) -> str:
    th = "".join(f'<th style="padding:3px 8px">{_esc(h)}</th>' for h in header)
    trs = []
    for r in rows:
        tds = "".join(f'<td style="padding:3px 8px">{c if str(c).startswith("<") else _esc(c)}</td>' for c in r)
        trs.append(f"<tr>{tds}</tr>")
    return (
        f'<table style="width:100%;border-collapse:collapse;font-size:{size}px">'
        f'<thead><tr style="color:{_DIM};text-align:left">{th}</tr></thead>'
        f'<tbody>{"".join(trs)}</tbody></table>'
    )


def render_html(result: Any) -> str:
    """整页 HTML。**内联 SVG、无外部引用**（测试会断言这一点）。"""
    v = result.version
    counts = dict(v.counts)
    names = list(result.names())

    # 指纹表
    fp_rows = []
    for n in names:
        g = result.genome(n)
        active = result.active_params(n)
        fp_rows.append(
            [
                f"<code>{_esc(n)}</code>",
                f"<code>{result.shape_fingerprint_of(n)[:12]}</code>",
                f"<code>{result.fingerprint_of(n)[:12]}</code>",
                _human(g.cost.params),
                _human(active) + ("　=总" if active == g.cost.params else ""),
            ]
        )

    # 同构类
    full = {k: x for k, x in result.classes.get("full", {}).items() if len(x) >= 2}
    shape = {k: x for k, x in result.classes.get("shape", {}).items() if len(x) >= 2}
    cls_rows = [
        [f"`{k[:12]}`", str(len(x)), ", ".join(f"<code>{_esc(y)}</code>" for y in x)]
        for k, x in sorted(full.items())
    ]

    # 时间线
    timelines = "".join(
        f'<div style="background:{_PANEL};padding:10px 12px;border-radius:6px;margin:8px 0">'
        f'<div style="font-size:13px;margin-bottom:2px"><code>{_esc(tl.family)}</code>'
        f'　{len(tl.members)} 个成员　<b style="color:{_MUT}">{len(tl.mutation_points)}</b> 个突变点</div>'
        f"{timeline_svg(tl)}</div>"
        for tl in result.timelines
    )

    # 突变表
    mut_rows = [
        [
            f"<code>{_esc(tl.family)}</code>",
            f"#{b.index}",
            f"<code>{_esc(b.left)}</code> → <code>{_esc(b.right)}</code>",
            f'<span style="color:{ {"mutation": _MUT, "scale": _ACCENT, "same": _OK}[b.kind] }">'
            f'{ {"mutation": "突变", "scale": "量变", "same": "零差异"}[b.kind] }</span>',
            str(b.n_edits),
            ", ".join(b.coordinates) or "—",
        ]
        for tl in result.timelines
        for b in tl.boundaries
    ]

    # 矩阵
    mrows = [
        [
            f"<code>{_esc(r.get('name'))}</code>",
            _esc(r.get("family", "")),
            _esc(r.get("attn_variant", "")),
            _esc(r.get("ffn_variant", "")),
            _esc(r.get("norm_kind", "")),
            _esc(r.get("position", "")),
            _esc(r.get("n_layers", "")),
            _esc(r.get("hidden", "")),
            _esc(r.get("n_experts", "")),
            _esc(r.get("params", "")),
            _esc(r.get("active_params", "")),
            _esc(r.get("kv_cache_bytes", "")),
        ]
        for r in result.matrix
    ]

    return f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ModelGenomeFactory 报告 {_esc(v.id)}</title></head>
<body style="margin:0;background:{_BG};color:{_FG};
font-family:system-ui,-apple-system,'Segoe UI',sans-serif">
<div style="max-width:960px;margin:0 auto;padding:28px 20px 60px">
<h1 style="font-size:22px;margin:0 0 4px">ModelGenomeFactory · 结构基因报告</h1>
<div style="color:{_DIM};font-size:13px;margin-bottom:18px">
版本 <code>{_esc(v.id)}</code>　协议 <code>{_esc(v.protocol_digest)[:12]}</code>　
{counts.get('models', 0)} 个模型　{counts.get('pairs', 0)} 个无序对　
创建于 {_esc(v.created_at)}
</div>

{_card(
    "本项目不训练模型",
    "产物是结构证据：把模型从「一堆代码」变成可比较的结构基因。"
    "它不回答「哪个模型更好」——那需要评测，不是结构。",
)}

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">
验收标准一 · 任意两个模型的结构 Diff</h2>
<div style="color:{_DIM};font-size:13px">格子里的数字是编辑条数；右上与左下必然对称（可逆性）。</div>
<div style="background:{_PANEL};padding:12px;border-radius:6px;margin-top:8px;overflow:auto">
{diff_matrix_svg(result)}</div>

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">
验收标准二 · 家族突变时间线</h2>
<div style="color:{_DIM};font-size:13px">
判据：相邻版本的结构<b>骨架指纹</b>不同 ⟺ 突变。没有阈值、没有容差。</div>
{timelines}

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">
版本边界明细</h2>
{_table(["家族", "#", "边界", "判定", "编辑", "突变坐标"], mut_rows)}

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">
两级指纹</h2>
<div style="color:{_DIM};font-size:13px">
骨架指纹相同 ⟺ 只改了规模；骨架指纹不同 ⟺ 结构突变。</div>
{_table(["模型", "骨架指纹", "全指纹", "参数量", "激活参数"], fp_rows)}

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">
全同构类（成员 ≥ 2）</h2>
<div style="color:{_DIM};font-size:13px">
同一类的成员之间<b>逐位没有结构差异</b> —— 进步来自数据与配方，不是结构。</div>
{_table(["指纹", "规模", "成员"], cls_rows) if cls_rows else f'<div style="color:{_DIM}">（没有成员 ≥2 的全同构类）</div>'}

<h2 style="font-size:16px;border-bottom:1px solid {_PANEL};padding-bottom:6px;margin-top:24px">
对比矩阵（节选）</h2>
{_table(["模型", "家族", "注意力", "FFN", "范数", "位置", "层", "宽度", "专家", "参数", "激活", "KV 缓存"], mrows)}

<div style="color:{_DIM};font-size:12px;margin-top:24px">
本页由 <code>python main.py visualize</code> 生成，自包含、无外部引用，可离线打开。
完整矩阵（{len(result.matrix)} 行 × 全部列）与逐对编辑序列在 <code>out/</code> 里。
</div>
</div></body></html>
"""
