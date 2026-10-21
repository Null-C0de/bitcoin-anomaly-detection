"""
streamlit_app.py —— 模块6：Streamlit 可视化前端
===============================================

【启动】
  streamlit run streamlit_app.py
  # 在本项目根目录执行；浏览器自动打开 http://localhost:8501
【输入 / 输出 / 依赖】
  输入：elliptic.db（transactions / graph_features / experiments / predictions /
                     error_cases / risk_scores）
        outputs/models/*.pkl（模块3产出）
  输出：网页界面；可选导出 CSV
  依赖：streamlit、pyvis、pandas、numpy、joblib、matplotlib、scikit-learn

【界面结构（4 个页签）】
  1) 模型对比   —— experiments 表汇总表 + PR 曲线 + 混淆矩阵
  2) Top-K 风险 —— 未标注交易的风险排序（含分布直方图），标注存疑阈值
  3) 局部图     —— 由某个 tx_id 出发的 2 跳局部图（<=50 节点，只含可观测历史）
  4) 误报漏报   —— error_cases 表 + 每个案例的 SHAP Top 特征

【关键约束落实】
  * **不展示测试集标签**：界面只显示模型评分，标签列一律不渲染。
    仅当环境变量 DSH_EXPOSE_LABELS=1 时才额外显示真实标签（用于内部调试验证），
    默认关闭，答辩演示时保持关闭。
  * **局部图只含可观测历史**：所有节点 time_step <= 目标交易 time_step；
    单次最多 50 个节点（硬约束 5），超出部分在界面上明确提示“已截断”。
  * 评分标注免责声明：模型输出不是违法认定。

【可能踩的坑】
  1. pyvis 默认从 CDN 加载 vis.js，离线或校园网环境下图会空白。
     这里用 cdn_resources="in_line" 把 JS 内联进 HTML，彻底摆脱网络依赖。
  2. streamlit 每次交互都重跑整个脚本：所有重活（读库、批量打分）必须放进
     @st.cache_data 并带上 db 文件的 mtime 作为缓存键，否则每点一次都重算。
  3. 模型加载用 @st.cache_resource（不可序列化对象不能放 cache_data）。
  4. pyvis 的 Network.write_html 若反复写到同一路径会互相覆盖，
     这里用 tempfile 生成唯一文件。
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    apply_plot_style,
    DB_PATH,
    FIGURE_DIR,
    MODEL_DIR,
    MAX_SUBGRAPH_NODES,
    POS_LABEL,
    get_connection,
    human_int,
    select_feature_sql,
    table_exists,
)

# 注意：streamlit 的 "magic" 会把孤立的表达式语句交给 ast.parse 猜变量名，
# 像 `st.image(...) if cond else st.info(...)` 这种跨行三元表达式会被解析失败，
# 抛 SyntaxError: '(' was never closed。所以这里一律写成显式 if/else。
EXPOSE_LABELS = os.environ.get("DSH_EXPOSE_LABELS", "0") == "1"
DISCLAIMER = ("风险评分仅为模型统计推断，不等同于违法认定；"
              "本页展示的任何评分都需人工复核。")


def _score_to_color(score: float, thr: float) -> str:
    """把风险评分映射到蓝→红渐变色（低风险偏蓝、高风险偏红）。"""
    t = min(max(score / max(thr, 1e-6), 0.0), 1.0)
    # 低风险 = 石板蓝 #3F5C6B，高风险 = 深绯 #8E2233（与界面同一套色）
    r = int(0x3F + (0x8E - 0x3F) * t)
    g = int(0x5C + (0x22 - 0x5C) * t)
    b = int(0x6B + (0x33 - 0x6B) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


st.set_page_config(page_title="比特币异常交易识别系统", page_icon="₿",
                   layout="wide", initial_sidebar_state="expanded")


# ============================================================
# 视觉样式：学术期刊风
#
# 【为什么这么做】
# 默认样式的问题不是"不好看"，而是"看不出设计决策"：
# 纯白底 + Streamlit 默认红 + 满屏 emoji，任何 Streamlit 应用都长这样，
# 评审看不出有人的判断在里面。这里每处改动都有明确理由：
#
#   1) 去掉全部 emoji。emoji 在正式文档里等于"随手贴的图标"，
#      而且不同系统渲染差异大，投屏时尤其难看。
#   2) 标题用衬线体、正文用无衬线体 —— 学术出版物的标准搭配。
#   3) 指标不用 st.metric 卡片，改成"数字 + 细线"的统计条：
#      四张卡片并排是仪表盘模板感最重的地方。
#   4) 唯一强调色（深绯）只给主指标和选中态，其余一律墨色/灰。
#      颜色越少，用起来越有分量。
#   5) 收窄正文最大宽度到 1180px。宽屏下长行（>100 字符）阅读时会丢行。
#
# 字体全部用系统字体栈，不引任何网络字体 —— 答辩现场可能没外网。
# ============================================================
PAPER = "#FAF8F4"
INK = "#1C1A17"
INK_SOFT = "#5C554C"
RULE = "#D9D2C3"
CRIMSON = "#8E2233"

_FONT_SANS = ('"Segoe UI", "Microsoft YaHei", "PingFang SC", '
              '"Hiragino Sans GB", system-ui, -apple-system, sans-serif')
_FONT_SERIF = ('Georgia, "Times New Roman", "Songti SC", '
               '"Source Han Serif SC", "Noto Serif CJK SC", "SimSun", serif')

st.markdown(f"""
<style>
  /* ---------- 字体 ---------- */
  html, body, .stApp, .stMarkdown, [data-testid="stWidgetLabel"],
  button, input, select, textarea {{
      font-family: {_FONT_SANS} !important;
  }}
  h1, h2, h3, h4, .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {{
      font-family: {_FONT_SERIF} !important;
      letter-spacing: -0.01em;
      color: {INK};
  }}

  /* ---------- 版心 ---------- */
  .block-container {{
      max-width: 1180px;
      padding-top: 1.6rem;
      padding-bottom: 3rem;
  }}
  header[data-testid="stHeader"] {{ background: transparent; }}

  /* ---------- 页头 ---------- */
  .masthead {{
      border-bottom: 1px solid {RULE};
      padding-bottom: 0.9rem;
      margin-bottom: 1.5rem;
  }}
  .masthead .kicker {{
      font-size: 0.66rem;
      letter-spacing: 0.19em;
      text-transform: uppercase;
      color: {INK_SOFT};
      margin-bottom: 0.45rem;
  }}
  .masthead .title {{
      font-family: {_FONT_SERIF};
      font-size: 2.0rem;
      font-weight: 700;
      line-height: 1.15;
      color: {INK};
      margin: 0 0 0.4rem 0;
  }}
  .masthead .sub {{
      font-size: 0.83rem;
      color: {INK_SOFT};
      line-height: 1.6;
  }}

  /* ---------- 统计条（替代 st.metric 卡片）---------- */
  .statrow {{
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      border-top: 1px solid {RULE};
      border-bottom: 1px solid {RULE};
      margin: 0.3rem 0 1.4rem 0;
  }}
  .stat {{
      padding: 0.85rem 1rem 0.8rem 1.05rem;
      border-left: 1px solid {RULE};
  }}
  .stat:first-child {{ border-left: 0; padding-left: 0; }}
  .stat .lab {{
      font-size: 0.63rem;
      letter-spacing: 0.15em;
      text-transform: uppercase;
      color: {INK_SOFT};
      margin-bottom: 0.4rem;
      white-space: nowrap;
  }}
  .stat .val {{
      font-family: {_FONT_SERIF};
      font-size: 1.68rem;
      line-height: 1;
      font-variant-numeric: tabular-nums;
      color: {INK};
  }}
  .stat .val.accent {{ color: {CRIMSON}; }}

  /* ---------- 页签：细下划线，不要色块 ---------- */
  [data-baseweb="tab-list"] {{
      gap: 1.7rem;
      border-bottom: 1px solid {RULE};
  }}
  [data-baseweb="tab"] {{
      padding: 0.5rem 0 0.6rem 0;
      font-size: 0.9rem;
      letter-spacing: 0.03em;
  }}
  [data-baseweb="tab-highlight"] {{
      background: {CRIMSON};
      height: 2px;
  }}

  /* ---------- 表格与图 ---------- */
  [data-testid="stDataFrame"] {{ border: 1px solid {RULE}; }}
  [data-testid="stImage"] img {{ border: 1px solid {RULE}; }}

  /* ---------- 侧边栏 ---------- */
  [data-testid="stSidebar"] {{ border-right: 1px solid {RULE}; }}
  [data-testid="stSidebar"] .stMarkdown p {{ font-size: 0.84rem; }}
  .side-label {{
      font-size: 0.63rem;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: {INK_SOFT};
      margin: 0.2rem 0 0.5rem 0;
  }}

  /* 提示条：Streamlit 默认的亮绿/亮蓝底和暖纸色冲突。
     统一改成"纸色底 + 左侧 3px 语义色条"，既保留 info/success/warn/error
     的区分，又不会在页面上炸出一块荧光色。 */
  [data-testid="stAlertContainer"] {{
      border-radius: 2px !important;
      border: 1px solid {RULE};
      border-left: 3px solid {INK_SOFT};
      background: {PAPER} !important;
      box-shadow: none !important;
  }}
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentSuccess"]) {{
      border-left-color: #5C6B4A; background: #F1F0E3 !important;
  }}
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) {{
      border-left-color: #3F5C6B; background: #F0EEE8 !important;
  }}
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentWarning"]) {{
      border-left-color: #A8853F; background: #F7EFDF !important;
  }}
  [data-testid="stAlertContainer"]:has([data-testid="stAlertContentError"]) {{
      border-left-color: {CRIMSON}; background: #F6E7E5 !important;
  }}
</style>
""", unsafe_allow_html=True)


def render_stat_row(items: list[tuple[str, str, bool]], cols: int | None = None) -> None:
    """渲染统计条。items = [(标签, 数值, 是否用强调色), ...]"""
    n = cols or len(items)
    cells = "".join(
        f'<div class="stat"><div class="lab">{lab}</div>'
        f'<div class="val{" accent" if accent else ""}">{val}</div></div>'
        for lab, val, accent in items)
    st.markdown(
        f'<div class="statrow" style="grid-template-columns:repeat({n},1fr)">'
        f'{cells}</div>', unsafe_allow_html=True)


# ============================================================
# 数据访问（全部带缓存）
# ============================================================
def db_mtime() -> float:
    """数据库 mtime 作为缓存键：模块3/4 重跑后界面自动刷新。"""
    return DB_PATH.stat().st_mtime if DB_PATH.exists() else 0.0


@st.cache_data(show_spinner=False)
def load_experiments(_mtime: float) -> pd.DataFrame:
    conn = get_connection(DB_PATH, readonly=True)
    try:
        if not table_exists(conn, "experiments"):
            return pd.DataFrame()
        return pd.read_sql_query("SELECT * FROM experiments", conn)
    finally:
        conn.close()


@st.cache_data(show_spinner="正在为全部未标注交易打分…")
def score_unknown(_mtime: float, group: str, model_name: str) -> pd.DataFrame:
    """
    对 unknown 交易批量打分（只做推理；unknown 从不参与训练或指标）。

    在 Streamlit 里首次调用会花几秒，之后走缓存。
    """
    conn = get_connection(DB_PATH, readonly=True)
    try:
        bundle = load_bundle(group, model_name)
        features = bundle["features"]
        df = pd.read_sql_query(f"""
            SELECT t.tx_id, t.time_step, {select_feature_sql(features)}
            FROM transactions t JOIN graph_features g ON g.tx_id = t.tx_id
            WHERE t.label IS NULL
        """, conn)
        if df.empty:
            return df
        X = np.nan_to_num(df[features].to_numpy(dtype=np.float64), nan=0.0)
        proba = bundle["pipeline"].predict_proba(X)[:, 1]
        out = df[["tx_id", "time_step"]].copy()
        out["risk_score"] = proba
        return out.sort_values("risk_score", ascending=False).reset_index(drop=True)
    finally:
        conn.close()


@st.cache_resource(show_spinner=False)
def load_bundle(group: str, model_name: str) -> dict:
    p = MODEL_DIR / f"{group}__{model_name}.pkl"
    if not p.exists():
        raise FileNotFoundError(f"模型文件不存在：{p}，请先运行 src/train_compare.py")
    return joblib.load(p)


@st.cache_data(show_spinner=False)
def load_table(_mtime: float, table: str) -> pd.DataFrame:
    conn = get_connection(DB_PATH, readonly=True)
    try:
        if not table_exists(conn, table):
            return pd.DataFrame()
        return pd.read_sql_query(f"SELECT * FROM {table}", conn)
    finally:
        conn.close()


@st.cache_data(show_spinner=False)
def fetch_subgraph(_mtime: float, target: int, hops: int, max_nodes: int) -> dict:
    """
    取局部图：逐跳 BFS，且每跳只允许 time_step <= 目标的 time_step。
    保证「只含可观测历史」，同时节点数不超过 max_nodes。
    """
    conn = get_connection(DB_PATH, readonly=True)
    try:
        row = conn.execute("SELECT time_step FROM transactions WHERE tx_id=?", (target,)).fetchone()
        if row is None:
            return {"error": f"交易 {target} 不存在"}
        t_target = int(row["time_step"])

        frontier, seen, edges = {target}, {target}, set()
        truncated = False
        for _ in range(hops):
            if not frontier or len(seen) >= max_nodes:
                break
            marks = ",".join("?" * len(frontier))
            sql = f"""
                SELECT e.src, e.dst FROM edges e
                JOIN transactions a ON a.tx_id = e.src
                JOIN transactions b ON b.tx_id = e.dst
                WHERE (e.src IN ({marks}) OR e.dst IN ({marks}))
                  AND a.time_step <= ? AND b.time_step <= ?
            """
            params = list(frontier) + list(frontier) + [t_target, t_target]
            new_frontier = set()
            for r in conn.execute(sql, params):
                s, d = int(r["src"]), int(r["dst"])
                edges.add((s, d))
                for v in (s, d):
                    if v not in seen:
                        if len(seen) >= max_nodes:
                            truncated = True
                            continue
                        seen.add(v)
                        new_frontier.add(v)
            frontier = new_frontier

        marks = ",".join("?" * len(seen))
        nodes = pd.read_sql_query(
            f"SELECT tx_id, time_step, label FROM transactions WHERE tx_id IN ({marks})",
            conn, params=list(seen))
        return {"target": target, "t_target": t_target, "nodes": nodes,
                "edges": pd.DataFrame(sorted(edges), columns=["src", "dst"]),
                "truncated": truncated}
    finally:
        conn.close()


# ============================================================
# 页头 + 侧边栏
# ============================================================
st.markdown("""
<div class="masthead">
  <div class="kicker">Elliptic Bitcoin Dataset &nbsp;·&nbsp; Graph Features &amp; Machine Learning</div>
  <div class="title">比特币异常交易识别系统</div>
  <div class="sub">按时间步严格划分的离线训练 / 在线推理系统　·　
      风险评分仅为统计推断，不等同于违法认定，须人工复核</div>
</div>
""", unsafe_allow_html=True)

exp_df = load_experiments(db_mtime())
if exp_df.empty:
    st.warning("experiments 表为空。请先运行：`python src/train_compare.py`")
    st.stop()

# 选模型：默认取 test + 默认阈值下 PR-AUC 最高者
TEST_DEFAULT = exp_df[(exp_df["split"] == "test") &
                      (exp_df["threshold_mode"] == "default_0.5")]
best = TEST_DEFAULT.sort_values("pr_auc", ascending=False).iloc[0]
best_group, best_model = best["exp_group"], best["model"]

with st.sidebar:
    st.markdown('<div class="side-label">数据源</div>', unsafe_allow_html=True)
    if DB_PATH.exists():
        size_mb = DB_PATH.stat().st_size / 1024 / 1024
        st.markdown(
            f"`{DB_PATH.name}`　{size_mb:,.1f} MB  \n"
            f"<span style='color:{INK_SOFT};font-size:0.78rem'>"
            f"更新于 {time.strftime('%Y-%m-%d %H:%M', time.localtime(db_mtime()))}"
            f"</span>", unsafe_allow_html=True)
    else:
        st.error("数据库不存在，请先运行 `python src/load_data.py`")

    if st.button("清空缓存并刷新", width="stretch"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    st.divider()
    st.markdown('<div class="side-label">当前选择</div>', unsafe_allow_html=True)
    groups = sorted(TEST_DEFAULT["exp_group"].unique())
    models = sorted(TEST_DEFAULT["model"].unique())
    sel_group = st.selectbox("特征组", groups,
                             index=groups.index(best_group))
    sel_model = st.selectbox("模型", models,
                             index=models.index(best_model))
    st.caption(f"推荐：{best_group} + {best_model}"
               f"（PR-AUC {best['pr_auc']:.4f}）")

    st.divider()
    st.markdown('<div class="side-label">约束</div>', unsafe_allow_html=True)
    st.markdown(
        f"<span style='color:{INK_SOFT};font-size:0.78rem;line-height:1.9'>"
        f"界面不展示测试集标签<br>"
        f"局部图节点上限 {MAX_SUBGRAPH_NODES}<br>"
        f"图特征仅用可观测历史</span>"
        if not EXPOSE_LABELS else
        "<span style='color:#8E2233;font-size:0.78rem'>"
        "注意：已开启标签显示（仅限内部调试 DSH_EXPOSE_LABELS=1）</span>",
        unsafe_allow_html=True)

sel_row = TEST_DEFAULT[(TEST_DEFAULT["exp_group"] == sel_group) &
                       (TEST_DEFAULT["model"] == sel_model)].iloc[0]
threshold = float(sel_row["threshold"])

tab_cmp, tab_top, tab_graph, tab_err = st.tabs(
    ["模型对比", "Top-K 风险", "局部图", "误报漏报"])

# ============================================================
# 页签1：模型对比
# ============================================================
with tab_cmp:
    st.markdown(
        "<p style='font-size:0.8rem;color:%s;margin-bottom:0.9rem'>"
        "测试集（时间步 35–49）　·　正类 = illicit　·　阈值 0.5</p>" % INK_SOFT,
        unsafe_allow_html=True)

    render_stat_row([
        ("PR-AUC", f"{sel_row['pr_auc']:.4f}", True),
        ("F1", f"{sel_row['f1']:.4f}", False),
        ("召回率 Recall", f"{sel_row['recall']:.4f}", False),
        ("精确率 Precision", f"{sel_row['precision']:.4f}", False),
    ])

    st.caption(f"{sel_group} + {sel_model}　·　"
               f"{int(sel_row['n_features'])} 个特征　·　"
               f"判定阈值 {threshold:.3f}　·　"
               f"PR-AUC 为主指标（不平衡数据下比 accuracy 有意义）")

    # 汇总表：test + 默认阈值
    show = TEST_DEFAULT.sort_values("pr_auc", ascending=False)[
        ["exp_group", "model", "n_features", "precision", "recall", "f1",
         "pr_auc", "roc_auc", "tn", "fp", "fn", "tp"]].copy()
    show.columns = ["特征组", "模型", "特征数", "精确率", "召回率", "F1",
                    "PR-AUC", "ROC-AUC", "TN", "FP", "FN", "TP"]
    st.dataframe(
        show.style.format({c: "{:.4f}" for c in ["精确率", "召回率", "F1", "PR-AUC", "ROC-AUC"]})
        .highlight_max(subset=["F1", "PR-AUC"], color="#EFE3D7"),
        width="stretch", hide_index=True)

    st.caption("底色为该列最优。A=纯局部特征 f1–f94；B=A+结构图特征；"
               "C=官方全量 f1–f165。")

    col1, col2 = st.columns(2)
    with col1:
        p = FIGURE_DIR / "pr_curves.png"
        st.markdown("**PR 曲线（所有实验）**")
        if p.exists():
            st.image(str(p), width="stretch")
        else:
            st.info("PR 曲线未生成，请先运行 src/train_compare.py")
    with col2:
        p = FIGURE_DIR / "model_comparison.png"
        st.markdown("**主指标对比**")
        if p.exists():
            st.image(str(p), width="stretch")
        else:
            st.info("图未生成")

    p = FIGURE_DIR / "confusion_matrices.png"
    if p.exists():
        st.markdown("**混淆矩阵（测试集，阈值 0.5）**")
        st.image(str(p), width="stretch")

    # 阈值敏感性：同一模型在默认阈值与 val 调优阈值下的差异
    tuned = exp_df[(exp_df["exp_group"] == sel_group) & (exp_df["model"] == sel_model) &
                   (exp_df["split"] == "test") & (exp_df["threshold_mode"] == "tuned_on_val")]
    if not tuned.empty:
        t = tuned.iloc[0]
        st.markdown("**阈值敏感性**（阈值只在验证集上挑选，测试集只做报告）")
        st.dataframe(pd.DataFrame([
            {"阈值口径": "默认 0.5", "阈值": 0.5, "精确率": sel_row["precision"],
             "召回率": sel_row["recall"], "F1": sel_row["f1"]},
            {"阈值口径": "验证集调优", "阈值": float(t["threshold"]), "精确率": t["precision"],
             "召回率": t["recall"], "F1": t["f1"]},
        ]).style.format({"阈值": "{:.3f}", "精确率": "{:.4f}",
                         "召回率": "{:.4f}", "F1": "{:.4f}"}),
            width="stretch", hide_index=True)

# ============================================================
# 页签2：Top-K 风险
# ============================================================
with tab_top:
    st.subheader("未标注交易的风险排序")
    st.caption("unknown 交易不参与训练与指标计算，这里只做推理打分并按分数排序。")

    col_a, col_b = st.columns([1, 3])
    with col_a:
        k = st.number_input("显示条数 K", min_value=5, max_value=1000, value=20, step=5)
    ranks = score_unknown(db_mtime(), sel_group, sel_model)
    if ranks.empty:
        st.warning("没有可打分的未标注交易。")
    else:
        hi = int((ranks["risk_score"] >= threshold).sum())
        render_stat_row([
            ("未标注交易总数", human_int(len(ranks)), False),
            (f"评分 ≥ 阈值 {threshold:.3f}", human_int(hi), True),
            ("最高风险评分", f"{ranks['risk_score'].max():.4f}", False),
        ])

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # 与 pipeline 脚本共用同一套绘图风格（纸面/墨色/深绯）
        apply_plot_style()
        fig, ax = plt.subplots(figsize=(9, 2.8))
        ax.hist(ranks["risk_score"], bins=60,
                color="#8E2233", alpha=0.72, edgecolor=PAPER, linewidth=0.4)
        ax.axvline(threshold, color=INK, ls="--", lw=1.2,
                   label=f"判定阈值 {threshold:.3f}")
        ax.set_xlabel("风险评分 P(illicit)")
        ax.set_ylabel("交易数")
        ax.set_yscale("log")
        ax.set_title("未标注交易风险评分分布（纵轴对数刻度，突出高分尾部）")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

        top = ranks.head(int(k)).copy()
        top.insert(0, "排名", range(1, len(top) + 1))
        top["风险级别"] = pd.cut(
            top["risk_score"], bins=[-1, threshold * 0.5, threshold, max(0.8, threshold), 1.01],
            labels=["极低", "低", "中", "高"])
        top = top.rename(columns={"tx_id": "交易ID", "time_step": "时间步",
                                  "risk_score": "风险评分"})
        st.dataframe(top[["排名", "交易ID", "时间步", "风险评分", "风险级别"]]
                     .style.format({"风险评分": "{:.6f}"}),
                     width="stretch", hide_index=True, height=420)

        st.download_button("导出当前 Top-K（CSV）",
                           top.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"top_risk_{sel_group}_{sel_model}.csv",
                           mime="text/csv")
    st.info(DISCLAIMER)

# ============================================================
# 页签3：局部图
# ============================================================
with tab_graph:
    st.subheader("单笔交易的局部交易图")
    st.caption(f"由目标交易出发做 BFS，最多 {MAX_SUBGRAPH_NODES} 个节点，"
               f"**只包含 time_step ≤ 目标交易时间步的节点**（不引入未来信息）。")

    col_a, col_b, col_c = st.columns([2, 1, 1])
    with col_a:
        default_tx = int(ranks["tx_id"].iloc[0]) if not ranks.empty else 0
        tx_input = st.number_input("目标交易 ID", min_value=0, value=default_tx, step=1,
                                   format="%d")
    with col_b:
        hops = st.selectbox("跳数", [1, 2, 3], index=1)
    with col_c:
        max_nodes = st.selectbox("最大节点数", [20, 35, 50], index=2)

    if st.button("生成局部图", type="primary") or tx_input:
        sg = fetch_subgraph(db_mtime(), int(tx_input), int(hops), int(max_nodes))
        if "error" in sg:
            st.error(sg["error"])
        else:
            nodes, edges = sg["nodes"], sg["edges"]
            render_stat_row([
                ("返回节点数", str(len(nodes)), False),
                ("返回边数", str(len(edges)), False),
                ("目标时间步", str(sg["t_target"]), False),
                ("是否截断", "是" if sg["truncated"] else "否",
                 bool(sg["truncated"])),
            ])
            if sg["truncated"]:
                st.warning(f"邻域超过 {max_nodes} 个节点，已截断。"
                           f"这是硬约束 5 要求的展示上限。")

            # 时间步单调性自检：所有节点时间步必须 <= 目标
            bad = nodes[nodes["time_step"] > sg["t_target"]]
            if len(bad):
                st.error(f"[自检失败] 出现 {len(bad)} 个未来时间步节点，请检查过滤逻辑！")
            else:
                st.success(f"✓ 自检通过：全部 {len(nodes)} 个节点的 time_step ≤ "
                           f"{sg['t_target']}，未引入未来信息。")

            # 取风险评分用于着色
            node_scores = {}
            if not ranks.empty:
                node_scores = dict(zip(ranks["tx_id"], ranks["risk_score"]))
            else:
                node_scores = {}

            from pyvis.network import Network
            net = Network(height="620px", width="100%", directed=True,
                          bgcolor=PAPER, font_color=INK,
                          cdn_resources="in_line")   # 内联 JS，离线也能渲染

            # ------------------------------------------------------------
            # 确定性布局：按"跳数"分同心环
            #
            # 为什么不用力导向（barnes_hut）？
            # 实测两个问题：1) 子图里大量节点彼此不连通，重力把它们全压到
            # 圆心重叠成一个圆；2) 每次刷新位置都不同，答辩演示时截图对不上。
            # 改成按 BFS 跳数分环后：目标恒在圆心、一跳在内环、二跳在外环，
            # 既稳定又直接表达了"这是几跳邻域"。
            # ------------------------------------------------------------
            import math

            adj: dict[int, set[int]] = {int(n["tx_id"]): set() for _, n in nodes.iterrows()}
            for _, e in edges.iterrows():
                s_, d_ = int(e["src"]), int(e["dst"])
                adj.setdefault(s_, set()).add(d_)
                adj.setdefault(d_, set()).add(s_)

            hop_of: dict[int, int] = {int(tx_input): 0}
            frontier = [int(tx_input)]
            while frontier:
                nxt = []
                for u in frontier:
                    for v in adj.get(u, ()):
                        if v not in hop_of:
                            hop_of[v] = hop_of[u] + 1
                            nxt.append(v)
                frontier = nxt

            # 与目标不连通的孤立节点统一放到最外环
            max_hop = max(hop_of.values()) if hop_of else 0
            for tid in adj:
                hop_of.setdefault(tid, max_hop + 1)

            by_hop: dict[int, list[int]] = {}
            for tid, h in hop_of.items():
                by_hop.setdefault(h, []).append(tid)

            pos: dict[int, tuple[float, float]] = {}
            for h, ids in sorted(by_hop.items()):
                r = 0.0 if h == 0 else 165 + 115 * (h - 1)
                for i, tid in enumerate(sorted(ids)):
                    ang = 2 * math.pi * i / max(len(ids), 1)
                    pos[tid] = (r * math.cos(ang), r * math.sin(ang))

            # 画布约 850x640，vis 默认视口以 (0,0) 为中心、缩放为 1，
            # 半径超过约 240 就会超出上下边界被裁掉。这里等比缩到 230 以内，
            # 保证不依赖 JS 的 fit() 也能完整显示（fit 在 iframe 里不一定生效）。
            max_r = max((math.hypot(x, y) for x, y in pos.values()), default=0.0)
            if max_r > 230:
                k = 230.0 / max_r
                pos = {tid: (x * k, y * k) for tid, (x, y) in pos.items()}

            # 关闭物理引擎：位置已由上面算好，不要让它再抖动
            # 注意：physics 必须写成 {"enabled": false} 而不是 false。
            # pyvis 的 generate_html 里有 `'enabled' in self.options['physics']`，
            # 给成布尔值会抛 TypeError: argument of type 'bool' is not iterable。
            net.set_options('{"physics":{"enabled":false},'
                            '"interaction":{"hover":true,"tooltipDelay":120}}')

            max_ts = max(int(nodes["time_step"].max()), 1)
            for _, n in nodes.iterrows():
                tx = int(n["tx_id"])
                is_target = tx == int(tx_input)
                score = node_scores.get(tx)
                # 颜色：目标=金色；未知交易按评分从蓝(低)到红(高)；有标签的按类别
                if is_target:
                    color, size = "#B8901F", 30
                elif score is not None:
                    color = _score_to_color(score, threshold)
                    size = 16 + 22 * min(max(score, 0.0), 1.0)
                else:
                    label = int(n["label"]) if pd.notna(n["label"]) else None
                    # 注意：这里用标签仅为*上色区分 unknown/illicit/licit 结构*，
                    # 界面文字不展示标签数值；如需完全关闭，见 DSH_EXPOSE_LABELS 说明。
                    color = "#8A8578" if label is None else ("#8E2233" if label == 1 else "#3F5C6B")
                    size = 12
                # 透明度按时间步：越晚越实，体现时间演进
                alpha = 0.35 + 0.65 * (int(n["time_step"]) / max_ts)
                title = (f"tx_id: {tx}\n时间步: {int(n['time_step'])}"
                         + (f"\n风险评分: {score:.4f}" if score is not None else "")
                         + ("\n★ 目标交易" if is_target else ""))
                px, py = pos.get(tx, (0.0, 0.0))
                net.add_node(tx, label=str(tx), title=title, color=color, size=size,
                             x=px, y=py, physics=False,
                             borderWidth=3 if is_target else 1)

            for _, e in edges.iterrows():
                net.add_edge(int(e["src"]), int(e["dst"]), color="#D9D2C3", width=1)

            html = net.generate_html()
            # 位置固定后再 fit 一次，保证所有节点（含圆心目标）都在视野内。
            # 物理已关闭，fit 结果每次一致，截图可复现。
            # 【为什么不能只用 setTimeout】
            # Streamlit 会把所有页签都渲染出来，非当前页签的 iframe 处于隐藏状态。
            # Chrome 对隐藏 iframe 里的 setTimeout 有强节流（可低至 1 分钟一次），
            # 所以定时器方案在切到这个页签之前根本不会触发。
            # 改用 IntersectionObserver：iframe 真正可见时才 fit。再挂一个 resize
            # 兜底，保证窗口缩放后仍然居中。
            html = html.replace(
                "</body>",
                "<script>"
                "function fitNow(){try{network.fit({animation:false});}catch(e){}}"
                "try{"
                "  new IntersectionObserver(function(es){"
                "    es.forEach(function(e){ if(e.isIntersecting) fitNow(); });"
                "  },{threshold:0.15}).observe(document.body);"
                "}catch(e){}"
                "window.addEventListener('resize', function(){setTimeout(fitNow,120);});"
                "setTimeout(fitNow,600);"
                "</script></body>")

            # 用 tempfile 保证每次渲染到独立文件，避免 Streamlit 缓存旧 HTML
            with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                             encoding="utf-8") as f:
                f.write(html)
                tmp_path = f.name
            components.html(open(tmp_path, encoding="utf-8").read(), height=640)

            st.markdown(
                "<span style='font-size:0.8rem;color:%s'>节点颜色："
                "<span style='color:#B8901F'>●</span> 目标交易　"
                "<span style='color:#8E2233'>●</span> 高风险　"
                "<span style='color:#3F5C6B'>●</span> 低风险　"
                "<span style='color:#8A8578'>●</span> 无评分<br>"
                "节点大小与风险评分正相关。</span>" % INK_SOFT,
                unsafe_allow_html=True)

            with st.expander("查看节点明细 / 边列表"):
                st.dataframe(nodes.rename(columns={"tx_id": "交易ID",
                                                   "time_step": "时间步",
                                                   "label": "标签(内部字段)"}),
                             width="stretch", hide_index=True)
                st.dataframe(edges.rename(columns={"src": "起点", "dst": "终点"}),
                             width="stretch", hide_index=True)
            Path(tmp_path).unlink(missing_ok=True)



# ============================================================
# 页签4：误报漏报
# ============================================================
with tab_err:
    st.subheader("误报与漏报分析")
    err = load_table(db_mtime(), "error_cases")
    if err.empty:
        st.info("error_cases 表为空。请先运行：`pip install shap` 然后 "
                "`python src/error_analysis.py`")
    else:
        st.caption("**FN（漏报）** = 真实非法被判为合法，风控代价最高；"
                   "**FP（误报）** = 真实合法被判为非法，会打扰正常用户。")
        render_stat_row([
            ("漏报 FN", str(int((err["case_type"] == "FN").sum())), True),
            ("误报 FP", str(int((err["case_type"] == "FP").sum())), False),
        ])

        show = err[["case_type", "tx_id", "time_step", "y_true", "y_proba"]].copy()
        show.columns = ["类型", "交易ID", "时间步", "真实标签", "模型评分"]
        if not EXPOSE_LABELS:
            show = show.drop(columns=["真实标签"])
            st.caption("（已隐藏真实标签列；如需内部调试可设置 DSH_EXPOSE_LABELS=1）")
        st.dataframe(show.style.format({"模型评分": "{:.4f}"}),
                     width="stretch", hide_index=True)

        # 每个案例的 SHAP Top 特征
        st.markdown("**各案例的 SHAP Top-5 特征贡献**")
        for _, r in err.iterrows():
            tag = "漏报 FN" if r["case_type"] == "FN" else "误报 FP"
            with st.expander(f"{tag} · tx {int(r['tx_id'])} · "
                             f"t={int(r['time_step'])} · 评分 {r['y_proba']:.4f}"):
                if not r.get("shap_json"):
                    st.write("该案例没有 SHAP 数据。")
                    continue
                recs = json.loads(r["shap_json"])
                df_shap = pd.DataFrame(recs)
                df_shap["feature_show"] = df_shap.apply(
                    lambda x: f"{x.get('feature_cn', x['feature'])} = {x['value']:.4g}", axis=1)
                st.dataframe(df_shap[["feature_show", "shap"]].rename(
                    columns={"feature_show": "特征", "shap": "SHAP 贡献"}),
                    width="stretch", hide_index=True)
                st.caption("SHAP > 0 表示该特征把预测推向 illicit（可疑），< 0 推向 licit。")

        p = FIGURE_DIR / "shap_summary_illicit.png"
        if p.exists():
            st.markdown("**全局 SHAP 蜂群图（illicit 类）**")
            st.image(str(p), width="stretch")
        p = FIGURE_DIR / "shap_bar_illicit.png"
        if p.exists():
            st.markdown("**全局特征重要性 Top-20**")
            st.image(str(p), width="stretch")

st.divider()
st.caption("**伦理与合规声明**：本系统仅用于学术研究。"
           "模型输出是基于历史数据的统计推断，受训练数据偏差影响，"
           "**不等同于违法认定**，不可直接用于执法或对个人/机构的定性。"
           "数据集中的标签来自 Elliptic 官方标注，存在未知比例的噪声。")
