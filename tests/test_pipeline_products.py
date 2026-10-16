"""产物、只读契约、CLI。

这个项目有一条**硬约定**：`run` 是唯一写产物的命令，其余 20 个只读命令**不许**
碰到 `out/`。这条约定如果不测，某天一个「顺手缓存一下」的实现就会让它悄悄失效 ——
而它的后果是「同一份产物跑两次结论不同」，正是本项目要证明不会发生的事。

`products ≡ memory` 也是同一条思路的另一面：产物必须**逐位**等于内存里的结论。
只要有一处不同，「可复现」就成了「跑一遍看起来一样」。

测试把产物写进 `tmp_path`（`out_dir` 夹具），不依赖仓库里 `out/` 的状态 ——
否则「产物 ≡ 内存」测的会是**上一次**运行的产物。
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from modelgenomefactory import pipeline
from modelgenomefactory.pipeline import EXTRA_FILES, GENOME_DIR, OUT_FILES


def _md5_dir(d: Path) -> dict[str, str]:
    out = {}
    for p in sorted(d.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(d))] = hashlib.md5(p.read_bytes()).hexdigest()
    return out


# ------------------------------------------------------------------ 产物清单


def test_write_outputs_writes_exactly_the_declared_files(out_dir, result) -> None:
    """`OUT_FILES` 是**声明**，产物是**事实**，两者必须对得上。

    多写了文件 = 有东西没被登记（只读命令读不到它，也没人知道它存在）；
    少写了文件 = 有声明是空的。两种都要拦。
    """
    got = {p.name for p in out_dir.iterdir() if p.is_file()}
    assert got == set(OUT_FILES), sorted(got ^ set(OUT_FILES))
    genomes = sorted(p.name for p in (out_dir / GENOME_DIR).glob("*.json"))
    assert genomes == sorted(f"{n}.json" for n in result.names())


def test_extra_files_are_written_only_by_visualize(out_dir) -> None:
    """`report.html` 属于 `visualize`，**不属于** `run`。

    它 50KB 起步，塞进 `run` 会让「跑一遍」从 4 秒变成 4 秒 + 一堆没看的字节。
    """
    for f in EXTRA_FILES:
        assert f not in {p.name for p in out_dir.iterdir()}
        assert f not in OUT_FILES


def test_version_json_records_counts_and_note(out_dir) -> None:
    v = json.loads((out_dir / "version.json").read_text(encoding="utf-8"))
    counts = v["counts"]
    assert counts["models"] == 13
    assert counts["pairs"] == 78
    assert counts["genomes"] == 13
    assert "不训练" in v["note"]
    assert v["id"].startswith("v1-")


def test_lineage_dot_is_valid_and_has_both_views(out_dir) -> None:
    full = (out_dir / "lineage.dot").read_text(encoding="utf-8")
    extract = (out_dir / "lineage_extract.dot").read_text(encoding="utf-8")
    for text in (full, extract):
        assert text.startswith("digraph")
        assert text.rstrip().endswith("}")
        assert text.count("{") == text.count("}")
    assert len(extract) < len(full), "抽取视图应当是全图的一个子集"


# ------------------------------------------------------------------ 产物 ≡ 内存


def test_products_are_bit_identical_to_memory(out_dir, loaded, result) -> None:
    """**产物 ≡ 内存。** 逐位比：指纹、矩阵、Diff、时间线、同构类。

    这条比「跑两次结果一样」强得多：它说的是「从产物重建出来的结论」与
    「算出来的结论」是同一份东西 —— 所以只读命令可以放心走产物那条路。
    """
    assert loaded.names() == result.names()
    for name in result.names():
        assert loaded.fingerprint_of(name) == result.fingerprint_of(name), name
        assert loaded.shape_fingerprint_of(name) == result.shape_fingerprint_of(name), name
        assert loaded.genome(name).cost == result.genome(name).cost, name

    assert dict(loaded.matrix[0]) == dict(result.matrix[0])
    assert loaded.matrix == result.matrix
    assert {k: dict(v) for k, v in loaded.classes.items()} == {k: dict(v) for k, v in result.classes.items()}

    assert sorted(loaded.pairs) == sorted(result.pairs)
    for key, d in result.pairs.items():
        assert loaded.pairs[key] == d, key

    assert [tl.to_dict() for tl in loaded.timelines] == [tl.to_dict() for tl in result.timelines]
    assert loaded.version.id == result.version.id
    assert loaded.rules == result.rules


def test_every_ordered_diff_direction_matches_memory(loaded, result) -> None:
    """**78 个无序对 × 2 个方向 = 156 个有向查询**都必须一致。

    产物里只存一个方向，反方向靠 `invert` —— 所以这条同时验了
    「只存一份」这个决定是对的。
    """
    names = result.names()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            assert loaded.diff(a, b) == result.diff(a, b), (a, b)
            assert loaded.diff(b, a) == result.diff(b, a), (b, a)


def test_load_result_returns_none_when_a_product_is_missing(out_dir, tmp_path) -> None:
    """缺产物**返回 None，不抛异常** —— 让 `cli` 能干净地退回在内存里跑一遍。

    一个只读命令因为「少了一个文件」而崩掉，用户会以为工具坏了。
    """
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "matrix.csv").write_text("x", encoding="utf-8")
    assert pipeline.load_result(partial) is None
    assert pipeline.load_result(tmp_path / "does-not-exist") is None
    assert pipeline.load_result(out_dir) is not None


# ------------------------------------------------------------------ 幂等与只读


def test_materialize_is_idempotent(root) -> None:
    """`materialize` 重跑必须**逐位相同** —— 否则每次跑都会把「输入」改一遍。"""
    from modelgenomefactory.extract import load_zoo

    zoo = load_zoo(root)
    dialects = ("hf", "vllm", "custom")
    before = {
        p: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted((root / pipeline.MANIFEST_DIR).rglob("manifest.*.json"))
    }
    pipeline.materialize(zoo, root=root, dialects=dialects)
    after = {
        p: hashlib.md5(p.read_bytes()).hexdigest()
        for p in sorted((root / pipeline.MANIFEST_DIR).rglob("manifest.*.json"))
    }
    assert before == after, "materialize 不是幂等的"


def test_run_pipeline_is_deterministic(result, cfg, root) -> None:
    """**除了 `created_at`，两次运行逐位相同。**

    `created_at` 是唯一不可逐位复现的字段 —— 它是元数据，不进指纹、不进 diff。
    如果哪天它混进了产物里参与比较，这条测试会立刻抓到。
    """
    again = pipeline.run_pipeline(cfg, root=root)
    assert again.version.digest == result.version.digest
    assert again.version.id == result.version.id
    for name in result.names():
        assert again.fingerprint_of(name) == result.fingerprint_of(name), name
    for key, d in result.pairs.items():
        assert again.pairs[key] == d, key
    assert [tl.to_dict() for tl in again.timelines] == [tl.to_dict() for tl in result.timelines]


# ------------------------------------------------------------------ CLI


def _cli(*argv: str, root: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(root / "main.py"), *argv],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


READ_ONLY_COMMANDS = (
    ("summary",),
    ("zoo",),
    ("classes",),
    ("matrix",),
    ("diff", "--all"),
    ("mutations",),
    ("crosscheck",),
    ("cost",),
    ("coverage",),
    ("rules",),
    ("lineage",),
    ("acceptance",),
    ("adversarial",),
    ("config",),
)

#: 逐模型的三个视图 + 两个「单模型」命令。`tree/graph/tensors/genome` 走 Genome，
#: `facts` 走 config 归一化 —— 它们是最容易「顺手写个缓存」的地方。
PER_MODEL_COMMANDS = ("tree", "graph", "tensors", "genome", "facts")
PROBE_MODELS = ("llama-2-7b", "deepseek-v2-lite")


def test_read_only_commands_exit_zero_and_do_not_touch_out(out_dir, root) -> None:
    """**只读契约：14 条只读命令 + 2 个模型 × 5 个视图，共 24 次调用跑一遍，
    `out/` 逐位不变，且每次都 exit 0、都有输出。**

    三件事合在一条测试里，是因为它们必须**在同一个 `out/` 上、同一次调用里**成立：
    拆成两条测试就要把 24 个子进程跑两遍 —— 而 `crosscheck`/`coverage`/`acceptance`
    这三个命令拿不到产物里的数据（互校项与覆盖率不是「比较的结论」，没有落盘），
    每条都要在内存里重跑整条流水线，约 4.3 秒。跑两遍就是白扔 17 秒。

    产物写在 `tmp_path` 里（`out_dir` 夹具）而不是仓库的 `out/`，于是这条测试
    **不依赖仓库里有没有产物**，也不会因为别人正在跑 `run` 而假红。

    用 md5 钉住而不是「没写文件」：写回**同样的字节**仍然是错的 —— 时间戳会变，
    而且它说明命令确实在写。
    """
    before = _md5_dir(out_dir)
    calls = [*READ_ONLY_COMMANDS, *( (cmd, m) for m in PROBE_MODELS for cmd in PER_MODEL_COMMANDS )]
    for argv in calls:
        r = _cli("--out", str(out_dir), *argv, root=root)
        assert r.returncode == 0, f"{argv}\nstdout:\n{r.stdout[-2000:]}\nstderr:\n{r.stderr[-2000:]}"
        assert r.stdout.strip(), f"{argv} 什么都没输出"
    assert _md5_dir(out_dir) == before, "只读命令改动了 out/"


def test_cli_reports_usage_error_for_a_missing_model(root) -> None:
    r = _cli("tree", root=root)
    assert r.returncode != 0
    assert "usage" in (r.stderr or "").lower() or "required" in (r.stderr or "").lower()


def test_cli_diff_requires_two_models(root) -> None:
    r = _cli("diff", "llama-2-7b", root=root)
    assert r.returncode != 0


def test_cli_json_flag_is_machine_readable(root) -> None:
    """`--json` 的 stdout 必须**整段**就是 JSON —— 不许夹任何说明文字。

    早先的实现是「先 print 说明，再追加一段 JSON」，于是下游得先自己找到第一个
    `{` 才能解析。这个测试当时也因此写成 `raw_decode` 从第一个 `{` 起切 —— 而它
    恰好掩盖了一个真实的坑：`zoo` 的载荷是**列表**，切片之后第一个 `{` 落在列表
    元素里，`raw_decode` 返回一个 dict，断言照样通过。

    现在 `--json` 由 `_quiet_prose` 把说明从 stdout 上摘掉，只留 `_emit_json`
    写出的那一段。所以这里直接用 `json.loads` 解析**整段** stdout，并逐个钉住
    顶层类型 —— 这比 `raw_decode` 强，因为它同时证明了「没有夹带」。
    """
    expected = {"zoo": list, "classes": dict, "cost": dict, "rules": dict, "matrix": list}
    for cmd, kind in expected.items():
        r = _cli(cmd, "--json", root=root)
        assert r.returncode == 0, (cmd, r.stderr[-500:])
        payload = json.loads(r.stdout)  # 整段就是 JSON，多余一个字符都会失败
        assert isinstance(payload, kind), (cmd, type(payload).__name__)


def test_visualize_writes_a_self_contained_html(root, tmp_path) -> None:
    """HTML **不许**引用任何外部资源 —— 它要能离线打开、能直接发给人。

    唯一的例外是 SVG 的 `xmlns`（那是个命名空间 URI，不是请求）。
    """
    r = _cli("visualize", root=root)
    assert r.returncode == 0, r.stderr[-2000:]
    html = (root / "out" / "report.html").read_text(encoding="utf-8")
    assert html.lstrip().startswith("<!DOCTYPE html>")
    for bad in ("<script src=", "<link ", "@import", "url(http"):
        assert bad not in html, f"HTML 引用了外部资源：{bad}"
    assert "http://www.w3.org/2000/svg" in html  # xmlns 是唯一允许的 URL
