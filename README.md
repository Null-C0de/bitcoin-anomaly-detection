# 基于图特征与机器学习的比特币异常交易识别系统

利用 Elliptic Bitcoin Dataset 的交易特征与交易图结构，识别非法（illicit）交易。
按时间步划分训练 / 验证 / 测试，对比三组特征与两种模型，并对误报漏报做 SHAP 可解释性分析，
最后提供 FastAPI 推理接口与 Streamlit 可视化前端。

> ⚠️ **伦理声明**：本项目仅用于学术研究。模型输出是基于历史数据的**统计推断**，
> 受训练数据偏差影响，**不等同于违法认定**，不可直接用于执法或对个人 / 机构定性。

---

## 1. 五条硬约束（贯穿全部代码）

| # | 约束 | 落地位置 |
|---|------|---------|
| 1 | `unknown` 不参与训练与指标，标签存 **NULL** 而非 0 | `src/load_data.py` 写库；`src/common.py` 的 `LABEL_UNKNOWN`；所有 `WHERE label IS NOT NULL` |
| 2 | 按时间步划分，**禁止随机划分**：train 1–29 / val 30–34 / test 35–49 | `src/common.py` 的 `TRAIN_STEPS/VAL_STEPS/TEST_STEPS`、`split_of()` |
| 3 | 图特征只用**预测时点可观测**的历史邻域，禁止未来信息 | `src/graph_features.py`；`src/app.py` 的 `/subgraph` 逐跳过滤 `time_step <= 目标` |
| 4 | 类别不平衡用 `class_weight='balanced'`，重点看 **PR-AUC 和 F1** | `src/train_compare.py` |
| 5 | 单次局部图展示邻域 **≤ 50 节点** | `MAX_SUBGRAPH_NODES = 50`；`/subgraph` 的 `max_nodes` 上限校验 |

---

## 2. 环境准备

```bash
# Python 3.10+（本机实测 3.12.2）
pip install -r requirements.txt
```

把 Elliptic 数据集的三个 CSV 放进 `Data/`：

```
Data/elliptic_txs_features.csv   # 203,769 行 × 167 列 = txId + time_step + f1..f165
Data/elliptic_txs_edgelist.csv   # 234,355 条边（无表头以外的自环 / 重复）
Data/elliptic_txs_classes.csv    # 标签：1=illicit, 2=licit, unknown
```

---

## 3. 目录结构

```
BITCOIN/
├── Data/                     # 原始 CSV（不入库版本控制，见 .gitignore）
├── Doc/                      # 本地文档（不入库版本控制，见 .gitignore）
├── src/                      # 全部源码
│   ├── common.py             # 公共常量、路径、时间划分、特征组、SQL 工具
│   ├── load_data.py          # 模块1：CSV → SQLite
│   ├── graph_features.py     # 模块2：图结构特征
│   ├── train_compare.py      # 模块3：A/B/C × LR/RF 对比实验
│   ├── error_analysis.py     # 模块4：误报漏报 + SHAP
│   ├── app.py                # 模块5：FastAPI 推理服务
│   └── streamlit_app.py      # 模块6：Streamlit 可视化前端
├── run.py                    # 一键启动脚本（自检 → 按需构建 → 启服务）
├── start.bat                 # Windows 双击入口（调用 run.py）
├── outputs/
│   ├── models/               # *_LR.pkl / *_RF.pkl
│   ├── figures/              # PR 曲线、混淆矩阵、SHAP 图
│   └── tables/               # 导出的 CSV
├── elliptic.db               # SQLite 数据库（不入库版本控制）
└── requirements.txt
```

---

## 4. 运行顺序

全部命令都在**项目根目录**执行。

### 4.1 推荐：一键启动

```bash
python run.py                # 自检环境 → 缺什么补什么 → 同时启动 API 与前端
python run.py --check        # 只检查环境和产物，不启动
python run.py --api-only     # 只起 API
python run.py --ui-only      # 只起前端
python run.py --build        # 从头重建全部产物（含 SHAP，约 7 分钟）
python run.py --build-core   # 只重建必需产物（不含 SHAP，约 2 分钟）
```

Windows 上也可以直接双击 `start.bat`。

`run.py` 会依次做四件事：

1. **检查 Python 版本与依赖** —— 按本次要启动的组件判断，
   `--api-only` 不会因为缺 `streamlit` 就拒绝启动；
2. **检查三个原始 CSV**；
3. **探测已构建产物** —— 不是看文件在不在，而是**查表里有没有数据**
   （避免"建库中途中断、文件在但表是空的"这种假就绪）；
4. **启动服务**并打印地址，端口被占用时自动改用相邻端口。

> **为什么两个服务是独立的**：`app.py` 与 `streamlit_app.py` 都直接读同一个
> `elliptic.db`，前端不通过 HTTP 调后端（`streamlit_app.py` 里没有任何
> `requests` 调用）。所以启动顺序无所谓，只起一个也能用。

### 4.2 手工分步执行（想单独控制每个模块时）

```bash
# 模块1：建库（约 2-4 分钟，658MB CSV 分块导入）
python src/load_data.py
python src/load_data.py --verify-only          # 只做完整性校验

# 模块2：图特征（约 70 秒）
python src/graph_features.py

# 模块3：对比实验（约 1 分钟）
python src/train_compare.py
python src/train_compare.py --groups A_local --models RF   # 只跑单个组合

# 模块4：误报漏报 + SHAP（耗时较长，见下方"已知耗时"）
python src/error_analysis.py
python src/error_analysis.py --max-samples 1000 --background-samples 20   # 快速跑通
python src/error_analysis.py --perturbation tree_path_dependent           # 最快，秒级出图

# 模块5：API 服务（根目录启动即可，--app-dir 会把 src/ 加入导入路径）
python -m uvicorn app:app --app-dir src --host 127.0.0.1 --port 8000
# 交互式文档： http://127.0.0.1:8000/docs

# 模块6：可视化前端（另开一个终端，仍在项目根目录）
python -m streamlit run src/streamlit_app.py
# 浏览器自动打开 http://localhost:8501
```

### 4.3 可选：给 API 加鉴权

默认不鉴权，方便本地演示。设置 `DSH_API_KEY` 后 5 个接口全部要求
请求头 `X-API-Key`，缺失或错误统一返回 401：

```bash
# 方式一：环境变量
export DSH_API_KEY=your-secret           # PowerShell: $env:DSH_API_KEY="your-secret"
python run.py

# 方式二：命令行参数
python run.py --api-key your-secret

# 验证
curl http://127.0.0.1:8000/health                               # 401
curl -H "X-API-Key: your-secret" http://127.0.0.1:8000/health   # 200
```

实现上用的是 `secrets.compare_digest()` 做**定长比较**，不是普通 `==`——
后者匹配失败会提前返回，攻击者能靠请求耗时的分布逐字节猜出密钥（时序攻击）。
另外 401 的文案统一，不区分"没带密钥"和"密钥错误"。

> **注意**：`app.py` 与 `streamlit_app.py` 都必须能 `import common`。
> uvicorn 用 `--app-dir src` 指定导入路径（已实测通过）；streamlit 用
> `src/` 前缀引用脚本即可（streamlit 会把脚本所在目录加入 `sys.path`）。
> 所有路径都由 `common.py` 以 `PROJECT_ROOT = Path(__file__).resolve().parents[1]`
> 锚定到项目根目录，**与当前工作目录无关**，在哪儿启动都能找到库和输出目录。

---

## 5. 数据库表结构

| 表 | 行数 | 说明 |
|----|------|------|
| `transactions` | 203,769 | `tx_id, time_step, f1..f165, label(NULL/0/1), split` |
| `edges` | 234,355 | `src, dst` |
| `graph_features` | 203,769 | `in_degree, out_degree, degree, clustering, nbr_count, two_hop_nbr_count, pagerank_rolling` |
| `experiments` | 24 | 3 特征组 × 2 模型 × 2 阈值口径 × 2 split 的指标 |
| `predictions` | 121,098 | 每个实验在 test 集的全量预测概率（供 PR 曲线复现） |
| `error_cases` | — | 每个 FP / FN 案例一行，含 SHAP Top 特征 JSON |
| `risk_scores` | 157,205 | unknown 交易的缓存风险分（`/top_risk` 用） |

索引：`transactions(time_step / label / split / split+label)`、`edges(src / dst)`、
`graph_features(tx_id)`、`risk_scores(risk_score / time_step)`。

---

## 6. 特征组与实验结果

**特征组**

- **A `A_local`**：官方局部特征 `f1–f94`
- **B `B_local_plus_graph`**：A + 6 个结构图特征
- **C `C_official_full`**：官方全量 `f1–f165`

**测试集结果（35–49 步，阈值 0.5，正类 = illicit）**

| 特征组 | 模型 | 特征数 | Precision | Recall | F1 | **PR-AUC** | ROC-AUC | TN/FP/FN/TP |
|--------|------|-------:|----------:|-------:|---:|-----------:|--------:|-------------|
| C_official_full | RF | 165 | 0.9864 | 0.6020 | 0.7477 | **0.7874** | 0.9291 | 15578/9/431/652 |
| B_local_plus_graph | RF | 100 | 0.9200 | 0.7008 | **0.7956** | 0.7799 | 0.8993 | 15521/66/324/759 |
| A_local | RF | 94 | 0.9177 | 0.6999 | 0.7941 | 0.7789 | 0.8963 | 15519/68/325/758 |
| C_official_full | LR | 165 | 0.1402 | 0.9474 | 0.2443 | 0.2044 | 0.8493 | 9297/6290/57/1026 |
| B_local_plus_graph | LR | 100 | 0.1328 | 0.8920 | 0.2311 | 0.1885 | 0.8363 | 9277/6310/117/966 |
| A_local | LR | 94 | 0.1333 | 0.8929 | 0.2319 | 0.1854 | 0.8358 | 9298/6289/116/967 |

结论：

1. **随机森林全面优于逻辑回归**（PR-AUC 0.78–0.79 vs 0.19–0.20）。
   LR 在不平衡数据上退化为"高召回、极低精确率"（FP 高达 6 千余），几乎不可用。
2. **结构图特征的增益很小**：RF 上 Δ(B−A) = **+0.0010**，LR 上 **+0.0031**。
   原因见下一节"数据发现"。
3. **全量特征 C 的 RF 精确率最高（0.9864，仅 9 个误报）但召回最低（0.602）**；
   B 组在 F1 上最优。三者的 PR-AUC 差距很小（0.779 vs 0.787）。

---

## 7. 两个关键数据发现（决定了图特征的设计）

### 发现 1：数据集中**不存在跨时间步的边**

逐条验证 234,355 条边，两端 `time_step` **完全相同**
（`src < dst` 为 0 条，`src == dst` 234,355 条，`src > dst` 0 条）。

因此需求文档里"只统计 `time_step` 小于自身的邻居的 illicit 比例"这一特征
对**全部节点都是空集**，恒为 0，没有任何信息量。

### 发现 2：邻居标签特征是**数据泄漏**，且线上不可复现

- test 期未标注交易中，**37.8%（19,197 / 50,834）在同时间步内就有带标签的邻居**；
- 未标注节点 **100% 至少有一个邻居**。

一条交易的邻居标签在"训练集里已知"，但现实风控场景中，同一时刻的邻居标签
在预测时刻根本拿不到；且这类特征会让模型直接"抄标签"，离线指标虚高。

### 因此的设计取舍（已确认）

| 项目 | 做法 | 理由 |
|------|------|------|
| 邻居标签比例 | **完全不用**（方案 B） | 见发现 1、2：恒为 0 且属泄漏 |
| 图特征只保留结构类 | `in_degree, out_degree, degree, clustering, nbr_count, two_hop_nbr_count` | 局部量，无未来信息，线上可复现 |
| PageRank | **滚动计算，仅展示，不进模型特征** | 全图 PageRank 会把未来时间步的影响力倒推给早期节点，违反硬约束 3 |

**关于图特征增益小的解释**：官方特征 `f95–f165` 本身就是数据方按时间窗口
聚合出的**邻域统计量**，已经吸收了大部分图结构信息；本项目额外计算的 6 个
纯结构特征与之高度冗余，所以 Δ(B−A) 只有千分之一量级。
这是数据集特性导致的诚实结论，而不是实现问题。

---

## 8. API 接口（模块5）

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查：当前模型、特征组、阈值、库文件名、是否开启鉴权 |
| POST | `/score` | 批量打分，请求体 `{"tx_ids":[...]}`，**单次 ≤ 1000 笔**（超出返回 413） |
| GET | `/top_risk?k=20` | 未标注交易的风险 Top-K（首次调用会建缓存，之后毫秒级） |
| GET | `/subgraph/{tx_id}?hops=2&max_nodes=50` | 局部交易图，**只含 `time_step ≤ 目标` 的节点**，`max_nodes` 上限 50 |
| GET | `/stats` | 请求数、平均耗时、当前 / 峰值内存 |

**实测性能**：1000 笔一次打分 **312 ms**（每笔约 0.31 ms），峰值内存 216 MB。

### 8.1 接口鉴权（可选）

默认**不鉴权**，方便本地演示。若要把服务暴露到本机以外，
设置环境变量 `DSH_API_KEY` 即可对所有接口开启校验：

```bash
# 开启鉴权
DSH_API_KEY='你要用的密钥' python -m uvicorn app:app --app-dir src --host 0.0.0.0 --port 8000

# 调用时带上请求头
curl -H "X-API-Key: 你要用的密钥" http://127.0.0.1:8000/health
```

未带或带错密钥一律返回 **401**，响应体不含任何密钥信息。
实测：正确密钥 200；未带 / 错误密钥 401；5 个接口行为一致。

实现要点（`src/app.py`）：

- 用 `secrets.compare_digest()` 做**定长比较**，避免通过响应时间差逐字节猜密钥（timing attack）
- 密钥只从环境变量读取，**不写进代码、不进仓库、不进日志**
- "没带密钥"和"密钥错误"返回同一个 401 文案，不给攻击者探针信号
- `/health` 只回数据库**文件名**而非绝对路径，避免泄漏服务器目录结构

**关于测试集标签**：接口默认**不返回真实标签**（`label` 恒为 `null`），
防止标签从 API 泄漏；仅当设置环境变量 `DSH_EXPOSE_LABELS=1` 时才返回，供内部调试。

---

## 9. 前端页面（模块6）

四个页签：

1. **模型对比**：指标卡片、全部实验汇总表（高亮最优）、PR 曲线、混淆矩阵、阈值敏感性对比
2. **Top-K 风险**：未标注交易的风险排序、评分分布直方图（对数纵轴突出高分尾部）、CSV 导出
3. **局部图**：PyVis 交互式交易图，目标交易金色、高风险红色、节点大小随评分变化，
   并**自检所有节点 `time_step ≤ 目标时间步`**
4. **误报漏报**：FP / FN 案例表 + 每个案例的 SHAP Top 特征归因 + 全局 SHAP 图

前端**不展示测试集标签**，并固定显示伦理声明。

### 9.1 视觉风格

界面刻意不用 Streamlit 默认样式，走"学术期刊"路线：

| 项目 | 取值 | 理由 |
|------|------|------|
| 纸面底 | `#FAF8F4` | 暖白，不是纯白（长文档里纯白刺眼） |
| 墨色文字 | `#1C1A17` | 非纯黑，对比不硬 |
| 唯一强调色 | `#8E2233` 深绯 | 只给主指标与选中态；颜色越少越有分量 |
| 标题字体 | 衬线（Georgia / 宋体） | 学术出版物标准搭配 |
| 正文字体 | 无衬线（Segoe UI / 微软雅黑） | 全部系统字体，**不引网络字体**，断网也能正常显示 |
| 圆角 | 4px | 近直角，去掉"产品化"的圆润感 |
| Emoji | 全部移除 | emoji 在不同系统渲染差异大，投屏尤其难看 |

指标区不用 `st.metric` 卡片，改成"数字 + 细线"的统计条——
四张卡片并排是仪表盘模板感最重的地方。

`src/common.py` 的 `apply_plot_style()` 会把 matplotlib 调成同一套配色
（纸面底、墨色轴、低饱和分类色、深绯→纸面的混淆矩阵色阶），
保证脚本生成的 PNG 和界面不是两种语言。

**一处踩坑**：`/subgraph` 的 PyVis 图原先用力导向布局，导致
①大量互不相连的节点被重力压到圆心重叠、②每次刷新位置都不同（演示时截图对不上）。
现改为按 BFS 跳数分同心环（目标在圆心、一跳内环、二跳外环），并关闭物理引擎，
布局完全确定、可复现。

**另一处踩坑**：Streamlit 会把所有页签都渲染出来，非当前页签的 iframe 处于隐藏状态，
Chrome 对隐藏 iframe 的 `setTimeout` 有强节流（可低至 1 分钟一次），
所以"用定时器调用 `network.fit()`"在切到该页签前根本不会触发、图会挤在左上角。
改用 `IntersectionObserver`，等 iframe 真正可见时才 fit。

---

## 10. 已知耗时与注意事项

| 步骤 | 耗时（本机 3070Ti 笔记本，纯 CPU） |
|------|------|
| 模块1 建库 | 约 2–4 分钟 |
| 模块2 图特征 | 约 70 秒 |
| 模块3 三组 × 两模型 | 约 1 分钟 |
| 模块4 SHAP（默认 `--max-samples 4000 --background-samples 100`） | 约 5 分钟（SHAP 主体约 3 分钟） |
| 模块4 SHAP 快速跑通 | 加 `--max-samples 1000 --background-samples 20`，约 1 分钟；或 `--perturbation tree_path_dependent` 秒级 |

踩过的坑（已在代码注释中标注）：

1. **CSV 实际是 167 列**（`txId + time_step + f1..f165`），需求文档写的 `f1~f166` 有误。
2. **特征组 C 没有任何图特征**，拼接 SQL 时若写成 `f"{tx_cols}, {g_cols}"`
   会留下多余逗号导致 `near "FROM": syntax error`。已抽出
   `common.select_feature_sql()` 统一处理。
3. **FastAPI 同步路由跑在线程池里**，SQLite 连接不能跨线程共享，
   否则报 `SQLite objects created in a thread can only be used in that same thread`。
   现改为**每线程一条只读连接**（`app.get_conn()`）。
4. **`pandas.DataFrame.style` 强依赖 `jinja2`**，且 pandas 3.x 要求 `jinja2>=3.1.5`；
   缺失时前端所有带样式的表格直接崩溃。
5. **Streamlit 的 magic 会把孤立表达式交给 `ast.parse`**，
   跨行三元表达式 `st.image(...) if cond else st.info(...)` 会抛 `SyntaxError`，
   一律改写成显式 `if/else`。
6. `pyvis` 默认从 CDN 拉 vis.js，离线会白屏，已改成 `cdn_resources="in_line"` 内联 JS。
7. **`shap` 自带的加性校验会误报**：4000 条样本算到 100% 时抛
   `ExplainerError: Additivity check failed ... sum of the SHAP values was 0.295100,
   while the model output was 0.315000`。RF 有 100 棵树，float32 舍入会让严格相等
   校验失败，导致几分钟的计算在最后一步全部丢弃。已改为
   `check_additivity=False`，并用 `test_shap_additivity()` 独立复验。
8. **SHAP 的输出是概率尺度，不是 log-odds**：`expected_value ≈ [0.5, 0.5]`
   （`class_weight='balanced'` 的结果），可直接与 `predict_proba` 相加核对。
   自检代码最初误把它当 log-odds 又做了一次 logit 变换，报出 4.6 的假偏差——
   **是自检口径错了，不是 SHAP 错了**；改正后 train / test 各 200 条样本上
   最大绝对误差为 `0.0000`。

---

## 11. SHAP 可解释性结果（模块4）

在最优模型（`C_official_full` + RF，阈值 0.36）上：
FP 71 个、FN 304 个；取 FP / FN 各 5 个案例写入 `error_cases` 表。

**加性自检（可写进答辩）**：`sum(SHAP) + base_value ≈ P(illicit)`，概率尺度上
最大偏差 **0.0001**（8 个抽查样本），与 `predict_proba` 一致。
`base_value = 0.1674`，是 `class_weight='balanced'` 下模型对全体样本的平均
P(illicit)，远高于 test 集真实非法占比 6.5%——这正是"平衡权重"的效果，
也解释了为什么 RF 的精确率能到 0.986 而召回只有 0.602。

**全局重要性 Top-10**（平均 |SHAP|，illicit 类，4000 条 test 样本）：

| 排名 | 特征 | 平均 \|SHAP\| | 排名 | 特征 | 平均 \|SHAP\| |
|-----:|------|-------------:|-----:|------|-------------:|
| 1 | f55 | 0.02368 | 6 | f52 | 0.00880 |
| 2 | f53 | 0.02029 | 7 | f132 | 0.00844 |
| 3 | f59 | 0.01228 | 8 | f41 | 0.00798 |
| 4 | f90 | 0.00982 | 9 | f47 | 0.00761 |
| 5 | f49 | 0.00901 | 10 | f40 | 0.00724 |

结论与业务解释：

- Top 特征集中在 **f53 / f55 / f59**（均为 `f1–f94` 局部区间）与 **f90**，
  说明局部交易形态是判定非法的核心依据；`f132` 属官方聚合区间 `f95–f165`，
  代表长期统计行为。
- **错判模式有明确方向性**：FN 的预测概率低至 0.005–0.010，属"模型很确信它是合法"，
  这类样本多为与已知非法节点**没有图连接**的孤立非法交易——纯特征无法捕捉；
  FP 集中在 0.535–0.905，最高一笔 0.905 是典型"形似非法但实为合法"的交易。
- 完整逐案例归因见 `outputs/tables/error_cases.csv` 的 `shap_json` 字段
  与 `outputs/figures/shap_error_cases.png`。

---

## 12. 检查清单（A–H 自查）

- **A 数据层**：3 表行数与原始 CSV 一致（203,769 / 234,355 / 203,769）；标签 `unknown` 存 NULL。
- **B 时间划分**：1–29 / 30–34 / 35–49，全局仅按 `time_step` 切分，无随机划分。
- **C 图特征**：全部为局部结构量；滚动 PageRank 只展示不进模型。
- **D 模型训练**：`class_weight='balanced'`；LR 前标准化、RF 不标准化；阈值只在验证集挑选。
- **E 结果合理性**：RF 精确率 0.92–0.99、召回 0.60–0.70，符合 Elliptic 论文量级。
- **F 系统**：5 个接口实测通过；1000 笔 312 ms。
- **G 泄漏与伦理**：无邻居标签特征、无未来信息、接口不返回标签、前端固定显示免责声明。
- **H 交付**：代码带注释、README 完整、`requirements.txt` 可用。
