# ModelGenomeFactory

> 来源：factory-everything/factory-everything-项目总表.md ｜ 板块：factory-everything · L2 ｜ 落地约束：4060 8G 单机可跑

## 一、条目信息（原始字段，完整保留）

| 字段 | 内容 |
| --- | --- |
| 项目 | ModelGenomeFactory |
| 类型 | Genome |
| 真正有意思的核心 | 模型结构 → 计算图 → 架构 Diff |

---

## 二、核心命题

**这个项目不要训练模型。**

它应该成为整个 L2 的**结构描述 / 比较基础设施**。核心是：

> 把模型从"一堆代码"变成可比较的结构基因。

## 三、整体架构

```
Model A                     Model B
  │                           │
  ▼                           ▼
Extractor                  Extractor
  │                           │
  └───────────┬───────────────┘
              ▼
         Canonicalizer
              │
              ▼
          Model IR
              │
       ┌──────┼──────┐
       ▼      ▼      ▼
    Module   Graph  Tensor
     Tree    Graph  Schema
       │      │      │
       └──────┼──────┘
              ▼
         Model Genome
              │
       ┌──────┼────────┐
       ▼      ▼        ▼
      Diff   Search  Visualize
```

## 四、Genome 结构

```
ModelGenomeFactory
│
├── Identity
│   ├── family
│   └── version
│
├── Topology
│   ├── layers = 32
│   ├── hidden = 4096
│   └── heads = 32
│
├── Operators
│   ├── Attention
│   ├── RMSNorm
│   └── SwiGLU
│
├── Connectivity
│   └── Compute DAG
│
├── Tensor Schema
│
└── Cost
    ├── Parameters
    ├── FLOPs
    └── Memory
```

## 五、主流水线

```
Checkpoint / Config / Code
 ↓
Extract
 ↓
Normalize
 ↓
Canonical Model IR
 ↓
Generate Genome
 ↓
Fingerprint
 ↓
Compare
 ↓
Structural Diff
 ↓
Graph Visualization
```

## 六、架构 Diff 示例

```
Llama
  │
  │ Architecture Diff
  ▼
Qwen

32 → 40 Layers
GQA → GQA
SwiGLU → SwiGLU
RoPE → Modified RoPE
Dense FFN → MoE
        ↑
      structural mutation
```

**这条 diff 就是"结构突变"的坐标。** 有了它，模型演化史可以从代码考古变成可比较的结构差异表。

## 七、关键设计点

| 点 | 说明 |
| --- | --- |
| 不训练 | 纯静态分析，从 Checkpoint / Config / 代码抽取结构 |
| Canonicalize | 不同框架（HF / vLLM / 自研）的命名与层级先归一到统一 IR |
| Fingerprint | 结构哈希，快速判断"是否同构" |
| 三视图 | Module Tree（层级）、Graph（连接）、Tensor Schema（张量契约） |
| Cost 内建 | 参数量 / FLOPs / 显存随结构一起产出，便于跨模型对比 |

## 八、阶段性落地计划

- **M0｜抽取骨架**：定义 Model IR 与 Genome 的 schema，跑通"读一个 HF 模型 → 输出 Genome"（非自动）。
- **M1｜Canonicalizer**：实现跨框架命名归一与结构提取，产出 Module Tree / Compute DAG / Tensor Schema。
- **M2｜Fingerprint 与 Diff**：实现结构指纹与架构 Diff，支持任意两个模型的结构差异对比。
- **M3｜搜索与可视化**：在 Genome 空间上做结构搜索，输出 Graph Visualization 与对比报告。

## 九、验收标准

能端到端完成「ModelGenomeFactory」，并在其「核心」（模型结构 → 计算图 → 架构 Diff）上给出可复现的量化结果：

- 对任意两个开源模型输出可复现的结构 Diff
- 同一家族不同版本的结构演化可被自动识别为"突变点"

---

_本 README 的「条目信息」字段来自 `factory-everything-项目总表.md`，架构与流水线为 L2 深化设计稿；可按实际进展修订。_

---

# 附录 A｜骨架、快速开始、CLI 契约、产物

## A.1 目录

```
L2-modelgenomefactory/
├── configs/default.json        所有口径与门槛（代码里不硬编码任何阈值）
├── data/zoo/
│   ├── zoo.json                zoo 索引：13 个模型的家族 / 链位置 / 方言 / 外部真值
│   ├── <model>/config.json     真实 HF config（逐字节照抄，未改一个键）
│   ├── <model>/manifest.<方言>.json   参数清单（名字 + 形状），共 38 份
│   └── _adversarial/           10 条对抗用例（`main.py adversarial` 逐条跑）
├── src/modelgenomefactory/
│   ├── rng.py                  确定性随机（sha256 派生种子）
│   ├── schema.py               四层对象链：RawStructure / CanonicalIR / Genome / Edit
│   ├── dialects.py             命名方言表：171 条规则 + 23 条条件规则的门槛
│   ├── extract.py              路径 A：config → 归一化事实（每个数字记住它从哪个原键来）
│   ├── canonical.py            路径 B：清单 → Canonical IR（模式空间）+ 两路互校
│   ├── genome.py               Genome：骨架指纹 / 全指纹 / 三视图
│   ├── cost.py                 代价三路：表求和 / 解析公式 / checkpoint 元数据
│   ├── diff.py                 Diff 代数：完备 / 最小 / 可逆 / 确定
│   ├── compare.py              对比矩阵 + 同构类
│   ├── mutations.py            家族突变时间线（验收标准第二条）
│   ├── lineage.py              结构血缘 DAG
│   ├── adversarial.py          对抗用例：证明 canonicalize 会报错而不是猜
│   ├── config.py               配置加载 + 版本 / 协议指纹
│   ├── pipeline.py             编排 + 落盘 + 只读重建（`load_result`）
│   ├── report.py               markdown 报告（十三节，顺序即论点顺序）
│   ├── visualize.py            自包含 HTML（内联 SVG，零外部引用）
│   └── cli.py                  23 个子命令（20 只读 + 3 写）
├── tests/                      131 条测试，约 48 秒
├── main.py                     入口（等价于 `python -m modelgenomefactory.cli`）
└── out/                        产物（见 A.4）
```

**形状：比较型（Comparator）。** 扇出的是**一组模型**，收拢点是**两两比较**。产物是
指纹表 + 两两结构 Diff + 家族突变时间线 + 对比矩阵。`objectives_of()` 返回空元组 ——
这里不训练、不选优，任何「按指标排序挑最优」的代码都会立刻发现无事可做。

**四层对象链**（本项目的全部内容就是这一条链）：

```
config.json + manifest
        │  键名归一
        ▼
   RawStructure            「每个数字从哪个原键来」
        │  结构归一
        ▼
    CanonicalIR            「结构事实 + 两路互校」（模式空间：blocks[*]）
        │  指纹化
        ▼
      Genome               「骨架指纹 / 全指纹 / 三视图 / 代价」
        │  两两比较
        ▼
   Diff / Boundary         「量变还是突变，以及突变坐标」
```

## A.2 快速开始

```bash
# 1. 先看任务本身：不需要任何产物
python main.py config            # 生效配置与设计期望
python main.py zoo               # 13 个模型 / 38 份命名清单
python main.py facts llama-2-7b  # config 归一化明细：每个数字从哪个原键来

# 2. 跑完整条流水线（本机约 4.5 秒，写 23 个文件）
python main.py run

# 3. 看结论（只读，从产物重建，不会重跑）
python main.py summary
python main.py fingerprint       # 两级指纹表
python main.py classes           # 同构类
python main.py mutations         # 家族突变时间线（验收标准第二条）
python main.py diff llama-2-7b llama-3-8b
python main.py diff --all        # 78 个无序对全部列出
python main.py acceptance        # 12 条验收标准逐条核对
python main.py adversarial       # 10 条对抗用例：证明它会报错而不是猜

# 4. 自包含 HTML（唯一额外的写命令）
python main.py visualize         # → out/report.html

# 5. 测试
python -m pytest                 # 131 passed，约 48 秒
                                 # pyproject.toml 里已经默认 -q，不要再加一个
```

## A.3 CLI 契约

| 类别 | 命令 | 作用 |
| --- | --- | --- |
| 只读 | `summary` | 版本 / 计数 / 家族 / 突变点概览 |
| 只读 | `zoo` | zoo 里的公开模型（含 HF `safetensors.total` 外部真值） |
| 只读 | `facts <model>` | config 归一化明细：每个数字从哪个原键来 |
| 只读 | `tree <model>` | Module Tree 视图 |
| 只读 | `graph <model> [--dot]` | Graph 视图（边表 / 裸 DOT） |
| 只读 | `tensors <model>` | Tensor Schema 视图 |
| 只读 | `genome <model>` | 一个模型的 Genome 全貌 |
| 只读 | `fingerprint` | 两级指纹表 |
| 只读 | `classes [--min-size N]` | 同构类（骨架类 / 全同构类） |
| 只读 | `matrix` | 对比矩阵（骨架列 / 规模列 / 代价列） |
| 只读 | `diff <a> <b>` ／ `diff --all` | 两个模型的结构 Diff（或全部无序对） |
| 只读 | `mutations` | 家族突变时间线 |
| 只读 | `crosscheck [<model>]` | config ↔ 清单 两路互校逐项 |
| 只读 | `cost [<model>]` | 代价三路互校 |
| 只读 | `coverage` | config 键覆盖率（读不懂的键 vs 有理由忽略的键） |
| 只读 | `rules` | 命名规则覆盖率（实测 / 条件 / 死规则 三栏） |
| 只读 | `lineage [--dot]` | 结构血缘 DAG |
| 只读 | `acceptance` | 12 条验收标准逐条核对 |
| 只读 | `adversarial` | 10 条对抗用例 |
| 只读 | `config` | 生效配置与设计期望 |
| 写 | `materialize` | 渲染命名清单到 `data/zoo/`（写**输入数据**，不是产物） |
| 写 | `run` | 跑流水线并落盘 |
| 写 | `visualize` | 写 `out/report.html` |

**只读契约（三条，都由测试钉住）：**

1. 只读命令**从产物重建结果对象**（`load_result`），不重跑流水线。
2. 产物缺失时，只读命令在内存里重跑，但**仍然不落盘**，并打印「未写盘」。
3. `run` / `visualize` / `materialize` 是仅有的三个写命令。

第 1、2 条由 `tests/test_pipeline_products.py` 用 **md5 指纹**验证：14 条只读命令
+ 2 个模型 × 5 个视图（共 24 次调用）跑一遍，`out/` 里每个文件一个字节都不许变。

**`--json` 的输出是纯 JSON。** 这一点单独说，因为它是「机器可读」这个词的全部内容：
`--json` 时命令内部的 `print` 会被 `_quiet_prose` 吞掉，只有 `_emit_json` 写的
那一段落到 stdout 上，于是 `python main.py rules --json | jq .` 直接可用。
（早先是「先打印说明、再追加一段 JSON」，下游得自己找第一个 `{` —— 而说明文字里
恰好出现一个花括号就会解析错。）

## A.4 产物清单

| 文件 | 内容 |
| --- | --- |
| `report.md` | markdown 报告，十三节，顺序即论点顺序 |
| `matrix.csv` | 对比矩阵（一行一个模型，骨架列 / 规模列 / 代价列） |
| `classes.json` | 同构类（按骨架指纹与全指纹分组） |
| `diffs.json` | **全部 78 个无序对**的编辑序列 |
| `mutations.json` | 家族突变时间线（每个边界一条判定） |
| `rules.json` | 命名规则覆盖率（每个布局的实测 / 条件 / 死规则） |
| `lineage.json` | 结构血缘 DAG（202 节点 / 439 边） |
| `lineage.dot` | 同一张图的 graphviz 版本 |
| `lineage_extract.dot` | 抽取层血缘（config 原键 → 归一化键） |
| `version.json` | 内容寻址版本（`v1-d5ba5edfcc6d`） |
| `genomes/*.json` | **13 个模型各一份可重新载入的 Genome**（本项目的可交付物） |
| `report.html` | 自包含 HTML（由 `visualize` 生成，不属于 `run`） |

**`load_result` 缺任何一个必需文件都返回 `None`** —— 不能「读一半」。半个结果对象
比没有结果更危险。

`OUT_FILES` 是**声明**，产物是**事实**，两者由测试逐项对齐：多写了文件 = 有东西
没被登记（只读命令读不到它，也没人知道它存在）；少写了文件 = 有声明是空的。

---

# 附录 B｜实测结果

## B.1 环境与实测开销

- **零第三方依赖**，只用标准库；`requires-python >= 3.12`。
- 本机实测：CPython 3.13.12 / pytest 9.1.1。
- 整条流水线 `run`：**约 4.5 秒**（13 个模型 × 38 份清单 + 78 个无序对 + 13 份 Genome）。
- 测试套件：**131 条，47.98 秒**。最贵的一条是只读契约（24 个子进程，其中
  `crosscheck` / `coverage` / `acceptance` 三条拿不到产物里的数据，每条要在内存里
  重跑整条流水线）。

## B.2 Zoo：13 个模型 / 38 份命名清单

| 模型 | 家族 | 链位置 | 方言 | 公开参数量 | HF `safetensors.total` |
| --- | --- | --- | --- | --- | --- |
| `llama-1-7b` | llama | 1 | custom / hf / vllm | 6.74B | 6,738,417,664 |
| `llama-2-7b` | llama | 2 | custom / hf / vllm | 6.74B | 6,738,417,664 |
| `llama-2-13b` | llama | 3 | custom / hf / vllm | 13.02B | 13,015,866,880 |
| `llama-3-8b` | llama | 4 | custom / hf / vllm | 8.03B | 8,030,261,248 |
| `llama-3-70b` | llama | 5 | custom / hf / vllm | 70.6B | 70,553,706,496 |
| `mistral-7b` | mistral | 1 | custom / hf / vllm | 7.24B | 7,241,732,096 |
| `mixtral-8x7b` | mistral | 2 | custom / hf / vllm | 46.7B | 46,702,792,704 |
| `qwen1.5-7b` | qwen | 1 | custom / hf / vllm | 7.72B | 7,721,324,544 |
| `qwen2-7b` | qwen | 2 | custom / hf / vllm | 7.62B | 7,615,616,512 |
| `qwen2.5-7b` | qwen | 3 | custom / hf / vllm | 7.62B | 7,615,616,512 |
| `phi-3-mini-4k` | phi | 1 | custom / hf / vllm | 3.82B | 3,821,079,552 |
| `deepseek-v2-lite` | deepseek | 1 | custom / hf / vllm | 15.7B | 15,706,484,224 |
| `gpt2` | gpt2 | 1 | custom / hf | 0.124B | 137,022,720 |

**本项目只读 config 与参数清单，不下载权重、不训练。** 最后一列是 HF 的
`safetensors.total` —— 逐字节数出来的**外部真值**，用它来钉住代价（见 B.4）。

三个方言是刻意选的：`hf` 是官方命名，`vllm` 是推理引擎的融合命名（`qkv_proj` /
`w13_weight`），`custom` 是自研命名（`Wq` / `Wk` / `Wv` / `Wgate` / `Wup` / `Wdown`）。
**同一模型的三种方言必须给出同一个指纹** —— 这是本项目最强的可复现性断言。

## B.3 两条验收标准

### B.3.1 ① 对任意两个开源模型输出可复现的结构 Diff

78 个无序对**全部**满足 Diff 代数的四条性质（`tests/test_diff_algebra.py` 逐对验）：

| 性质 | 断言 |
| --- | --- |
| 完备 | `apply(a, diff(a, b)).primary() ≡ b.primary()` |
| 最小 | 删掉任一编辑就不再完备（`is_minimal`） |
| 可逆 | `diff(b, a) ≡ invert(diff(a, b))` |
| 确定 | 同一对输入必得同一份编辑序列（编辑按「类别 + 地址」排序） |

只存**无序对**（`pair_key = "|".join(sorted(...))`），两个方向共用一条记录。

一条真实 Diff 的例子（`llama-2-7b → llama-3-8b`）：

```
  判定      : **结构突变**
  编辑      : 15 条（规模 14 / 结构 1）
  突变坐标  : variant
  骨架指纹  : dbe35821d3ab → dbba5a2b7d7f  （变了）
  全指纹    : 385fa1964304 → ed376eb25707

    - variant    nodes.blocks[*].attn.variant : 'mha' → 'gqa'
    - shape      ×7   tensors.blocks[*].attn.k_proj.weight.shape : (4096, 4096) → (4096, 1024)
    - dim        ×2   nodes.embed.dims.vocab : 32000 → 128256
    - dim        nodes.blocks[*].attn.dims.n_kv_heads : 32 → 8
    - dim        nodes.blocks[*].ffn.dims.intermediate : 11008 → 14336
    - dim        nodes.position.dims.theta : 10000 → 500000
    - attr       topology.max_positions : 4096 → 8192
    - attr       topology.vocab : 32000 → 128256
```

**这就是 README 第六节那张手画示意图的可计算版本。** 它把「32 → 40 层」「GQA →
GQA」这类描述换成了带地址的编辑，而且**判定了哪一条才是突变**：15 条编辑里 14 条
是规模变化，只有 `variant: mha → gqa` 改动了骨架。

### B.3.2 ② 同一家族不同版本的结构演化可被自动识别为「突变点」

判据是**骨架指纹**，不是阈值、不是编辑条数：**相邻版本的骨架指纹不同 ⟺ 突变**。

```
  llama（5 个成员，1 个突变点）
    1 llama-1-7b → llama-2-7b      scale      4 —
    2 llama-2-7b → llama-2-13b     scale     20 —
    3 llama-2-13b → llama-3-8b     mutation  26 variant
    4 llama-3-8b → llama-3-70b     scale     19 —

  qwen（3 个成员，1 个突变点）
    1 qwen1.5-7b → qwen2-7b        mutation  28 variant
    2 qwen2-7b → qwen2.5-7b        same       0 结构零差异：指纹逐位相同

  mistral（2 个成员，1 个突变点）
    1 mistral-7b → mixtral-8x7b    mutation  19 operator, tensor_schema, topology, variant
```

三条结论：

1. **llama 的突变点是 `llama-2-13b → llama-3-8b`**，不是 `llama-2-7b → llama-3-8b`
   （后者是 README 里那条手画示例的写法）。判据给出了**正确的边界**：`llama-2-13b`
   与 `llama-3-8b` 之间才发生了 `mha → gqa`。
2. **`qwen2-7b → qwen2.5-7b` 是零差异**：这两个发布版的结构**逐位相同**，
   连参数量都是同一个数（7,615,616,512）。所以 qwen 家族只有一个突变点，在
   `qwen1.5-7b → qwen2-7b`。
3. **`mistral-7b → mixtral-8x7b` 的突变坐标是四条**（`operator, tensor_schema,
   topology, variant`）—— 它是唯一一个**同时**改了算子、张量契约、拓扑和变体的边界，
   正是 README 第六节说的「Dense FFN → MoE ↑ structural mutation」。

同构类（按指纹分组）也印证了这一点：

```
  全同构（逐位相同）：12 类，其中成员 ≥2 的 1 类
    `282668b6f12e`  ×2  qwen2-7b, qwen2.5-7b

  骨架类（同一结构、不同规模）：9 类，其中成员 ≥2 的 3 类
    `dbe35821d3ab`  ×3  llama-1-7b, llama-2-13b, llama-2-7b
    `000ff7f98c91`  ×2  qwen2-7b, qwen2.5-7b
    `dbba5a2b7d7f`  ×2  llama-3-70b, llama-3-8b
```

`llama-1-7b` / `llama-2-7b` / `llama-2-13b` 落在同一个骨架类里 —— **它们是同一份
结构的三个规模**，这正是「量变」的定义；而 `llama-3-8b` 与 `llama-3-70b` 落到
另一个类里（GQA 骨架），与突变点的位置完全吻合。

## B.4 代价三路逐位互校

三路互校，**13 个模型全部逐位一致**：

| 路 | 来源 | 说明 |
| --- | --- | --- |
| ① | 张量表求和 | 把每个张量的元素数加起来 |
| ② | 解析公式 | 只用 config 标量算，**不碰张量表** |
| ③ | checkpoint 元数据 | HF 的 `safetensors.total`，等式写成 `params + buffers == total` |

```
  模型                          ① 表求和        ② 解析公式      ③ checkpoint         缓冲 一致
  deepseek-v2-lite        15,706,484,224    15,706,484,224    15,706,484,224            0 ✅
  gpt2                       124,439,808       124,439,808       124,439,808   12,582,912 ✅
  llama-1-7b               6,738,415,616     6,738,415,616     6,738,415,616        2,048 ✅
  llama-2-13b             13,015,864,320    13,015,864,320    13,015,864,320        2,560 ✅
  llama-2-7b               6,738,415,616     6,738,415,616     6,738,415,616        2,048 ✅
  llama-3-70b             70,553,706,496    70,553,706,496    70,553,706,496            0 ✅
  llama-3-8b               8,030,261,248     8,030,261,248     8,030,261,248            0 ✅
  mistral-7b               7,241,732,096     7,241,732,096     7,241,732,096            0 ✅
  mixtral-8x7b            46,702,792,704    46,702,792,704    46,702,792,704            0 ✅
  phi-3-mini-4k            3,821,079,552     3,821,079,552     3,821,079,552            0 ✅
  qwen1.5-7b               7,721,324,544     7,721,324,544     7,721,324,544            0 ✅
  qwen2-7b                 7,615,616,512     7,615,616,512     7,615,616,512            0 ✅
  qwen2.5-7b               7,615,616,512     7,615,616,512     7,615,616,512            0 ✅
```

②「不碰张量表」这一点是**刻意的**：如果它读了张量表，它与①就只是同一份数据的
两种写法，互校退化成自证。

**两个不进「参数量」这一栏、但必须单独有一列的量：**

- `active_params`：MoE 的关键列。`mixtral-8x7b` 总参数 46.7B、激活参数 12.9B ——
  **差一个数量级**。把它和 `params` 并排放，是「MoE 用更大的容量、花更少的算力」
  这句话唯一诚实的表达方式。专家按 `top_k` 计入，共享专家**永远在线**。
- `kv_cache_bytes`：MHA → GQA → MLA 这一串突变在**结构上**只是换了个注意力变体，
  在**代价上**差好几倍：

  | 模型 | 注意力 | `n_kv_heads` | `kv_cache_bytes` |
  | --- | --- | --- | --- |
  | `llama-2-7b` | MHA | 32 | 1,073,741,824 |
  | `llama-3-8b` | GQA | 8 | 268,435,456 |
  | `qwen2-7b` | GQA | 4 | 117,440,512 |
  | `deepseek-v2-lite` | MLA | 16 | 63,700,992 |

  `llama-2-7b == 4 × llama-3-8b` 这个等式是**逐位**成立的（`n_kv_heads` 32 → 8），
  有一条测试专门钉住它。

## B.5 编辑种类覆盖：13 / 14，`op` 缺一格

在真实 zoo 的 78 个无序对上统计（共 3074 条编辑）：

| 编辑种类 | 条数 | 类别 |
| --- | --- | --- |
| `dim` | 733 | 规模 |
| `shape` | 709 | 规模 |
| `attr` | 416 | 规模 |
| `tensor_del` | 393 | 突变 |
| `tensor_add` | 217 | 突变 |
| `edge_del` | 184 | 突变 |
| `variant` | 140 | 突变 |
| `node_del` | 90 | 突变 |
| `edge_add` | 72 | 突变 |
| `flag_del` | 50 | 突变 |
| `flag_add` | 44 | 突变 |
| `node_add` | 14 | 突变 |
| `tie` | 12 | 突变 |
| `op` | **0** | 突变 |

**`op` 是 0，这是结构事实，不是漏测。** 13 个模型的 FFN 只有 `mlp` / `swiglu` / `moe`
三种**变体**，注意力只有 `mha` / `gqa` / `mla` 三种**变体** —— 全都在 `variant` 这一格
里。要让 `op` 出现，得有一个模型的 `blocks[*].ffn.kind` 从「一个算子」变成「另一个
算子」，而公开 zoo 里没有这样的边界。

处理方式是**明说，不是假装覆盖**：`tests/test_diff_algebra.py` 分两条互补测试 ——
一条断言真实 zoo 恰好覆盖 13 种（`op` 显式记为 0 覆盖），另一条用一个**刻意合成**的
微型 Genome（`tiny["op_swap"]`）专门走 `op` 这一支。这样「14 种编辑类型的代码路径
全都被执行过」和「真实数据里只有 13 种」是两句可以分别验证的话。

## B.6 命名规则覆盖率：171 = 121 + 50 + 0

```
  布局             规则数   实测   条件 死规则
  custom_v1            37     34      3      0
  deepseek             30     20     10      0
  gpt2                 17     17      0      0
  llama                35     15     20      0
  mixtral              20     13      7      0
  vllm_llama           13     10      3      0
  vllm_mixtral         19     12      7      0
  合计                171    121     50      0
```

**三栏分开是有意的。** 一条**从没被任何清单匹配过**的规则看起来像「支持了某种布局」，
而它某天真的被匹配上时行为是**未经验证**的。所以每条规则要么

- 在 Zoo 的解析路径上**实测出现过**（121 条），要么
- 挂在**一个明确的门槛**下（50 条）—— 某条 config 事实成立时它才会出现，且被探针
  **逐条**验过。

第三栏「**死规则**」必须为空，而它是空的（0 条）。条件规则共 50 条，按名字去重后
是 23 条不同的规则，门槛有 6 个：

| 门槛 | 条数 | 例子 |
| --- | --- | --- |
| `n_experts` | 7 | `model.layers.{i}.mlp.experts[{e}].down_proj.weight` |
| `q_lora_rank` | 6 | `layers.{i}.attn.Nqa` |
| `mlp_bias` | 4 | `model.layers.{i}.mlp.down_proj.bias` |
| `qkv_bias` | 3 | `model.layers.{i}.self_attn.k_proj.bias` |
| `qk_norm` | 2 | `model.layers.{i}.self_attn.k_norm.weight` |
| `o_bias` | 1 | `model.layers.{i}.self_attn.o_proj.bias` |

这一栏的纪律与 `NON_STRUCTURAL_KEYS`（有理由忽略的 config 键）是同一条：**「不知道」
和「决定不看」必须分成两栏**，合并成一栏会让报告把一个真问题和一个设计决定说成
同一件事。config 键覆盖率里 118 处键是**有理由忽略**的（dropout / 版本号 / 分词器 id …），
「读不懂的键」是 0。

## B.7 设计期望 vs 实测

`configs/default.json` 里的 `expectations` 是**设计期望**，允许被实测推翻。
被推翻的那些是本节最值得看的部分 —— 一条被推翻的期望比十条被证实的期望有信息量。

| 期望 | 声明 | 实测 | 判定 |
| --- | --- | --- | --- |
| `llama_mutation_points` | 1 | 1 | ✅ 支持 |
| `qwen_mutation_points` | 1 | 1 | ✅ 支持 |
| `mistral_mutation_points` | 1 | 1 | ✅ 支持 |
| `qwen_last_boundary_is_zero_diff` | 是 | 是 | ✅ 支持 |
| `min_full_isomorphic_classes` | 1 | 1 | ✅ 支持 |

**被推翻的那条：**

> 原以为「代价可以缓存」—— 既然不训练，同一个模型的结构代价应该能算一次就存下来，
> 后续比较直接查表。

实测推翻了它：**代价是主地址空间的函数**，而 `diff.apply` 会重建 Genome，重建之后
必须能**逐位复算**代价。如果把代价当成独立事实缓存下来，`apply(a, diff(a, b))` 得到的
Genome 的代价就可能和 `b` 的对不上 —— 于是「完备性」这条断言会在代价上破掉。

结论：**代价不进主地址空间，但必须可重算**（`cost.derive` 只吃 `.tensors` /
`.nodes` / `.topo()`，所以 IR 与 Genome 共用同一套算法）。这条被推翻的期望写进了
`schema.py` 的注释，而不是留在报告里当一段感慨。

## B.8 可复现性边界

| 字段 | 逐位可复现 | 说明 |
| --- | --- | --- |
| `version.digest` | 是 | 内容寻址：zoo 索引 + 每个 config + 每份清单的摘要 |
| `fingerprint` / `shape_fingerprint` | 是 | 只吃结构，不吃文件名、不吃时间 |
| 全部编辑序列（`diffs.json`） | 是 | 编辑按「类别 + 地址」排序，同输入必得同序 |
| 突变时间线 / 同构类 / 矩阵 | 是 | 都是指纹与编辑的函数 |
| `cost` 全部字段 | 是 | 主地址空间的函数，`apply` 后可逐位复算 |
| `version.created_at` | **否** | 唯一的时间戳字段 |
| `tensors[*].source_name` | 是（但不进指纹） | 命名方言的痕迹，只用于审计 |

**「同一个模型换一种命名方言，指纹必须不变」是这个项目最强的可复现性断言。**
它比「跑两次结果一样」强得多：后者只说明代码是确定的，前者说明**指纹真的只吃结构**。
13 个模型 × 3 种方言全部一致，而 `run_pipeline` 在拿到不一致的方言指纹时会**直接抛错**
（不让它悄悄进报告）。

**一条已知的缝**：`llama-3-8b.qkv_axis` 这条对抗用例记录着 —— vLLM 真实 checkpoint 的
`qkv_proj.weight` 沿**第 0 维**拼接，而本项目沿**最后一维**。宽度检查会**拒绝**这种
清单（`融合 qkv 宽度对不上：清单 4096，按 (32+2·8)·128 = 6144`），所以行为是
「拒绝而不是猜」—— 但**接不上真实 vLLM checkpoint** 这件事是真的，它被记在
对抗用例的说明里，而不是被悄悄绕过。

---

# 附录 C｜坑（75–93）

编号承接同工作区前序项目（1–57 见 L2-architecturefactory，58–74 见 L2-moefactory）。

### 75. `role` 不是路径的纯函数 —— 指纹会静默漂移

**症状**：`test_fingerprint_is_a_function_of_the_address_space` 在 `base` 上失败 ——
`rebuild(g, primary(g))` 出来的 Genome 指纹**不一样**，而 `primary_diff_keys` 是**空的**。
「地址空间完全相同，指纹却不同」是最难查的一类不一致。

**原因**：`conftest._ir` 里把 `role` 定义成「路径去掉 `blocks[*].`」，于是
`blocks[*].norm1.weight` 得到 `"norm1.weight"`，而 `dialects.role_of` 给的是 `"norm"`。
`role` **进指纹**但不进地址空间 —— 于是地址空间一样、指纹不同。

**修法**：夹具改成调 `dialects.role_of`，并加一条
`test_role_is_a_pure_function_of_the_path` 把「`role` 是路径的纯函数」这个不变量钉住。
**一个量只要进了指纹，它就必须是地址空间的函数**；否则 `rebuild` 的保真性会被
一个看不见的字段破坏。

### 76. 模式空间 vs 实例空间：`blocks[*]` 永远不等于 `blocks[0]`

**症状**：`test_render_then_parse_returns_the_same_canonical_paths` 在 `('gpt2','custom')`
上失败，差异里全是 `blocks[*]` 对 `blocks[0]`。

**原因**：IR 存的是**模式**（32 层记成一条 `blocks[*]` + `topology.block_groups = ((0,32),)`），
而 `parse_manifest` 吐出来的是**实例**（`blocks[0]`、`model.layers.7.*`）。两个集合
本来就不可能相等 —— 这是设计，不是 bug。

**修法**：比较前先 `dialects.expand_paths(..., block_groups=groups, n_experts=n_experts)`
把 IR 展开到实例空间。**Genome 是模式，不是展开的实例** —— 这个区别一旦忘记，
就会写出「32 层 = 32 份重复数据」的 Genome，而它的指纹会把层数编码进去，
让「同结构不同规模」再也认不出来。

### 77. 「死规则」把两类完全不同的问题混成了一类

**症状**：`test_every_name_rule_is_used_by_at_least_one_manifest` 报出一大批死规则
（llama 19、vllm_llama 11、mixtral 10、vllm_mixtral 10、deepseek 14、custom_v1 5）。

**原因**：诊断把两种东西混在一起了：

- **过度声明**（真 bug）：MLA 规则挂在 `llama` / `vllm_llama` 上、k/v 规则挂在
  `deepseek` 上、dense-FFN 规则挂在 `mixtral` 上 —— 这些布局**根本不会**出现那些名字。
- **结构性死代码**：`custom_v1` 的 `layers.{i}.attn.Wqkv` 与 `layers.{i}.mlp.W12`。
  `rule_for` 先试 1:1 规则，而 `custom_v1` 有 `Wq`/`Wk`/`Wv` + `Wgate`/`Wup`/`Wdown`，
  所以那两条融合规则**永远不可达**。

**修法**：分开处理。过度声明的**收窄作用域**（k/v 从 `_L + _MM + _D` 收到 `_L + _MM`，
因为 deepseek 用 MLA 的 `kv_a_proj_with_mqa` / `kv_b_proj`，从来没有 `k_proj`/`v_proj`）；
死代码**直接删**。然后引入第三栏纪律（见 B.6）：每条规则必须要么实测、要么挂门槛。

### 78. 过度修正：把 MoE 规则收窄到 `_D`，对抗用例立刻红

**症状**：删掉 7 条「死」的 MoE 规则（`model.layers.{i}.mlp.gate.weight` 等）之后，
`main.py adversarial` 开始失败：`llama-2-7b.wrong_role.json` 期望「FFN 认不出来」，
实际得到「没有规则能解释参数名 `model.layers.0.mlp.gate.weight`」。

**原因**：那条对抗用例**故意**把 `mlp.gate_proj` 改写成 `mlp.gate`，用来证明
「**能匹配上一条规则**」≠「**结构对**」—— 挡住它的是**块级 FFN 校验**，不是名字表。
而 `model.layers.{i}.mlp.gate.weight` **确实是** llama 前缀的合法词汇
（Qwen1.5-MoE / DeepSeek 的 HF 命名就是这样），所以它必须被**认出来**，
然后由块级校验去否决它。删掉规则等于把「名字不认识」和「结构不对」这两件事
合并成了一个错误信息。

**修法**：把 `_L` **恢复**回去，7 条规则全部登记进 `CONDITIONAL_RULES`（门槛 `n_experts`），
并补上渲染方向 + 解析方向的探针。代码注释里留了原话：

> 这条 `_L` 我删过一次：当时死规则测试报了 7 条…**那是过度修正** —— 紧接着对抗用例
> 就红了…正确做法是把它登记进 `CONDITIONAL_RULES` 并配探针，不是删掉。

**「死规则」的测试报警不等于「这条规则该删」。** 它只说明这条规则**当前没被走到**，
而「当前没被走到」有三种成因：过度声明、结构性死代码、**门槛没满足**。只有前两种该动刀。

### 79. 探针拿模板去比对具体渲染出来的名字

**症状**：`test_conditional_rules_are_reachable` 里 `model.layers.{i}.self_attn.q_proj.bias`
在 `manifest["tensors"]` 里找不到。

**原因**：清单里存的是**具体**名字（`model.layers.0.self_attn.q_proj.bias`），
而注册表里存的是**模板**（`{i}` / `{e}`）。

**修法**：比对前先把模板实例化：`concrete = want.replace("{i}", "0").replace("{e}", "0")`。

### 80. 解析方向的探针拿具体源名去查模板键的注册表

**症状**：`dl.CONDITIONAL_RULES.get('model.layers.0.self_attn.q_norm.weight')` 返回 `None`，
于是探针误报「这条条件规则没被登记」。

**原因**：`CONDITIONAL_RULES` 的键是**模板**，而探针传的是**具体**名字 —— 与 79 正好
是同一个坑的反方向。

**修法**：`_PARSE_PROBES` 里存模板，只在构造手写清单时才替换。**模板与实例的边界
必须只有一个地方跨越**，跨越两次就一定会有一处忘了替换。

### 81. `SCALE_EDIT_KINDS | MUTATION_EDIT_KINDS` 报 `TypeError`

**症状**：`TypeError: unsupported operand type(s) for |: 'tuple' and 'tuple'`。

**原因**：两个常量都是 **tuple**（要保证顺序，因为 `EDIT_KINDS` 的迭代顺序决定了
`to_dict()` 的字段顺序），而 `|` 只对 set 有定义。

**修法**：`frozenset(SCALE_EDIT_KINDS) | frozenset(MUTATION_EDIT_KINDS)`。
顺手把这个坑写进 `ALL_KINDS` 的定义处 —— 「顺序重要」和「集合运算」是两个需求，
用同一个类型表达就会撞上。

### 82. 测试伸进了 `tiny["base"]._ir` 这种私有属性

**症状**：`test_flops_scale_with_seq_len` 报 `AttributeError`。

**原因**：`tiny` 夹具直接产出 `Genome`，中间没有留 `_ir` / `_raw` 这样的私有字段，
而测试想改 `seq_len` 重新算一遍 FLOPs。

**修法**：加一个 `tiny_of(name="base", *, seq_len, batch, dtype_bytes, **ir_kw) -> Genome`
夹具，测试写成 `tiny_of(seq_len=128)` vs `tiny_of(seq_len=256)`。
**测试不该为了省一个夹具去摸私有字段** —— 那会把「夹具的构造方式」变成测试的隐含依赖。

### 83. 跨测试模块 import（`tests/` 没有 `__init__.py`）

**症状**：`test_crosscheck_rejects_a_config_that_disagrees_with_the_manifest` 里
`from .test_extract_dialects import _with_facts` —— 会 `ImportError`。

**原因**：`tests/` 不是包，相对导入不成立；即使改成 `import conftest`，也要依赖
pytest 的 `prepend` 导入模式才碰巧能用。

**修法**：把 `with_facts` / `facts_of` 提到 `tests/conftest.py` 里做**会话级夹具**。
**共享的构造助手属于 conftest，不属于某个测试文件** —— 否则测试文件之间就形成了
一个「谁先被收集」的隐式依赖。

### 84. `MISSING = object()` 不可 JSON 序列化 —— 但那是对的

**症状**：`test_missing_sentinel_round_trips_through_json` 报
`TypeError: Object of type object is not JSON serializable`。

**原因**：`MISSING = object()` 是「这个地址不存在」的内部哨兵，本来就**不该**出现在
JSON 里 —— 编辑里存的是 `None`（`edge_add` 的 `None → True` 就是这个约定）。

**修法**：**删掉那条测试**，换成 `test_none_means_absent_except_for_tie`，
断言真实存在的约定（`tie` 是唯一一个 `None` 有别的含义的编辑种类）。
一条断言了错误约定的测试比没有测试更糟 —— 它会逼着实现去迁就一个不存在的需求。

### 85. `EDIT_INVERSE` 只列了 8 对，其余是自逆

**症状**：`test_edit_kind_inverse_table_is_an_involution` 报 `KeyError: 'shape'`。

**原因**：`EDIT_INVERSE` 只登记 add/del 这 8 对（`node_add ↔ node_del` 等），
`attr` / `dim` / `shape` / `variant` / `op` / `tie` 都是自逆的，所以不在表里。

**修法**：`EDIT_INVERSE.get(kind, kind)`，并断言表的**键集合恰好**是那 8 个。
自逆是「默认」，登记是「例外」—— 把默认值写进表里会让表失去「哪些是例外」的信息。

### 86. 我对「MoE 是 `op` 变更」的假设是错的

**症状**：`test_moe_mutation_is_an_operator_change` 与
`test_gqa_mutation_touches_the_tensor_schema` 双双失败。

**原因**：我的假设错了，而且是**两处**都错：

- MLP → MoE 是 `blocks[*].ffn.variant: swiglu → moe`，即 `variant` 变更，**不是 `op`**。
- MHA → GQA 也是 `variant`，KV 变窄表现为 `dim` / `shape`，**不是 tensor_schema**。

**修法**：两条测试改写成断言 `("variant",)`，并把「`op` 在真实 zoo 里不可达」这件事
写进测试文档串（见 B.5）。**先测量再断言** —— 我对「什么算算子变更」的直觉
把一个变体切换当成了算子替换。

### 87. 固定模板的微型 zoo 不可能覆盖全部编辑种类

**症状**：`test_tiny_zoo_covers_every_edit_kind` 列出 7 种没覆盖的种类。

**原因**：`_ir` 模板是固定的（2 层、4 头、swiglu、rmsnorm），所以
`node_add` / `node_del` / `tensor_add` / `tensor_del` / `edge_add` / `edge_del` /
`flag_del` **不可能**出现 —— 微型 zoo 里没有一条边界会增删节点或边。

**修法**：拆成两条互补的测试：一条断言微型 zoo **恰好**覆盖那 7 种（精确集合，
不是「至少」），一条断言真实 zoo **恰好**覆盖 13 种且 `op` 显式为 0。
「至少 N 种」这种断言永远绿，而它什么也没证明。

### 88. `--json` 只是「追加」一段 JSON，于是它根本没法 pipe

**症状**：`python main.py rules --json | jq .` 报解析错。`--json` 的输出是
「一段说明 + 一段 JSON」。

**原因**：`--json` 的实现是「照常打印说明，最后再 `print(json.dumps(...))`」。
声明了机器可读，产出的却是人读的东西 —— 而下游得自己找第一个 `{`。

**修法**：加 `_emit_json`（唯一出口，写到捕获的 `_REAL_STDOUT`）+ `_quiet_prose`
（`--json` 时把 `sys.stdout` 换成 `StringIO`，命令内部的 `print` 全被吞掉）。
这样**每个命令的写法不用改**，说明照常 `print`，两者不会同时出现在 stdout 上。

**这个坑在测试里还藏了第二层**：那条测试当时写成「从第一个 `{` 起 `raw_decode`」，
而 `zoo` 的载荷是**列表** —— 切片之后第一个 `{` 落在列表元素里，`raw_decode` 返回
一个 dict，`assert isinstance(payload, dict)` 照样通过。改成整段 `json.loads`
之后，它才真正在证明「没有夹带」。

### 89. 自己的机械重构把 `cli.py` 改坏了，而这个工作区没有 git

**症状**：`SyntaxError: 193 unmatched ')'`；`def` 从第 119 行之后全部丢掉缩进。

**原因**：我用正则把 `print(json.dumps(...))` 批量替换成 `_emit_json(...)`，
而模式里的 `.*?` 非贪婪 + 里面有一个 `)` 让它**过匹配**：它吞掉了 `json.dumps` 的
收尾 `)`，然后一路吃到几百行之后的**下一个** `, ensure_ascii=False, indent=2`；
替换函数又把中间所有行的缩进抹平了。

而当时 `git status` 报 `fatal: not a git repository` —— **当时没有版本控制兜底**，
一次过匹配就是不可撤销的。
（2026-10-07 注：工作区根**现在已是** git 仓库，本项目也有一批文件在版本控制里，
但**当前改动尚未提交** ⇒ 结论不变。）

**修法**：定位最后一个 column-0 的 `def`（169 行）与第一个被顶掉的 `def`（195 行），
打印中间内容确认**只是缩进损坏、没有内容丢失**，把这段重建到 `_fix_block.txt`，
再用带断言守卫的脚本拼接：

```python
assert src[190].strip() == "if args.json:", src[190]
assert src[439].strip() == ")", src[439]
assert src[440].strip() == "return 0", src[440]
new = src[:190] + block + src[440:]
ast.parse(text)  # → parses OK
```

**教训**：批量文本替换必须**先 `ast.parse` 再落盘**，而且模式要避开「跨行匹配」——
`.*?` 配 `re.DOTALL` 在代码上就是一颗地雷。另外：**先确认工作区有没有版本控制**，
再决定要不要动批量替换。

### 90. `Edit` 把两行合并成一行，得到一个「调用 `None`」的错误

**症状**：`TypeError: 'NoneType' object is not callable` at `pipeline.py:456`。

**原因**：一次 `Edit` 把
`written.append(out / "lineage.json")` 与
`(out / "lineage.dot").write_text(...)` 合并到了一行上，于是
`written.append(...)` 的返回值（`None`）被当成函数调用。

**修法**：把换行补回去。**`Edit` 的 `old_string` 不要只截半行** ——
它会把「半行」当成一个完整单元去替换。

### 91. 两条测试各跑一遍同样的 24 个子进程，白扔 17 秒

**症状**：`test_read_only_commands_exit_zero`（14 条参数化）与
`test_read_only_commands_do_not_touch_out`（24 次调用）覆盖的几乎是同一批命令，
合计约 38 秒。而且后者依赖仓库里存在 `out/`，缺了就直接 `skip`。

**原因**：两个断言被拆成了两条测试，于是每次调用都付两遍。而
`crosscheck` / `coverage` / `acceptance` 这三条**拿不到产物里的数据**
（互校项与覆盖率不是「比较的结论」，没有落盘），每条都要在内存里重跑整条流水线，
约 4.3 秒。

**修法**：合并成一条 —— 24 次调用，同时断言 `exit 0`、有输出、`out/` 逐位不变。
并把产物写在 `tmp_path` 里（用 `--out` 指过去），于是它**不再依赖仓库里有没有产物**，
那个 `skip` 也一起去掉了。套件从 **65.25 秒 / 145 条** 变成 **47.98 秒 / 131 条**。

**教训**：「同一个夹具、同一个前提」的断言应该合在一条测试里。拆开的代价不只是
重复执行，还有一个更隐蔽的后果：拆开之后两条测试的前置条件会**各自演化**，
于是其中一条悄悄加了个 `skip`。

### 92. `cmd_rules` 声明了产物却不读它

**症状**：`rules.json` 已经是 `OUT_FILES` 里的一员，`load_result` 也会把它恢复成
`result.rules`，但 `cmd_rules` 仍然写 `_result(args, fresh=True)` —— 于是它每次都
重算，并且永远打印「产物不存在或本命令需要额外数据」。

**原因**：这条命令比 `rules.json` 这个产物更早写出来（当时规则覆盖率还不在产物里），
加了产物之后忘了回头改。

**修法**：改成 `_result(args)`。**加了产物就要回头检查谁还在重算** ——
否则「只读命令读产物」这条契约会有一个静默的例外，而例外正是契约失效的方式。

### 93. 「设计期望」被实测推翻，而它比被证实的期望更有价值

**症状**：无报错。`cost` 的注释里写着「既然不训练，代价可以缓存」。

**原因**：实测推翻了它 —— **代价是主地址空间的函数**，而 `diff.apply` 会重建 Genome，
重建之后必须能逐位复算代价。把代价当独立事实缓存下来，`apply(a, diff(a, b))` 的
代价就可能和 `b` 对不上，「完备性」会在代价上破掉。

**修法**：代价**不进主地址空间，但必须可重算**（`cost.derive` 只吃 `.tensors` /
`.nodes` / `.topo()`，IR 与 Genome 共用同一套算法）。这条被推翻的期望写进了
`schema.py` 的注释，而不是留在报告里当一段感慨。

**这条是本项目最值得留的一条**：把「设计期望」写进配置、让流水线**逐条判定**
它是否被支持，被推翻的那些就成了最有信息量的输出。而它的修法不是「改期望」，
是**改设计**。

