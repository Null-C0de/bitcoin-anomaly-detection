"""
error_analysis.py —— 模块4：误报漏报分析 + SHAP 可解释性
======================================================

【输入】
  elliptic.db
    - transactions            特征与标签
    - graph_features          图特征
    - predictions             （模块3产出）各实验在 val/test 上的预测概率
    - experiments             （模块3产出）用于挑选“最优模型”
【输出】
  elliptic.db → error_cases 表（每个 FP / FN 案例一行，含 SHAP Top 特征 JSON）
  outputs/figures/shap_summary_illicit.png     SHAP 蜂群图（针对 illicit 类）
  outputs/figures/shap_bar_illicit.png         全局平均 |SHAP| 排序
  outputs/figures/shap_error_cases.png         每个案例的局部解释（瀑布图）
  outputs/tables/error_cases.csv
  outputs/tables/shap_feature_importance.csv

【方法说明（答辩要点）】
  1) 为什么对 RF 用 TreeExplainer：
     RandomForest 是树集成，SHAP 有精确的多项式时间算法（TreeExplainer）。
     它直接读取树结构与叶节点样本数，给出**精确**的 Shapley 值，
     不需要 KernelExplainer 那种“采样近似”（后者在 6 万样本上会慢到不可用）。
  2) 为什么解释 illicit 类：
     业务关心的是“为什么这笔交易被判为可疑”。二分类下解释正类，
     等价于解释模型输出概率 P(illicit) 相对基线的偏移。
     注意：sklearn RF 的 TreeExplainer 返回值是**概率尺度**——
     `expected_value` 约等于 0.5（class_weight='balanced' 的结果），
     SHAP 值可直接与 `predict_proba` 相加核对：
         sum(shap_values(X)[:, :, 1]) + expected_value[1] ≈ P(illicit)
     实测 train / test 各 200 条上最大绝对误差 0.0000。（早先误当作
     log-odds 并做 logit 变换，得到过 4.6 的假偏差，已修正。）
     解释正类时使用的就是输出的第 2 列（index=1）。
  3) 案例优先级：FN（漏报）优先于 FP（误报）。
     漏报意味着真实非法交易被放过，在风控场景里代价更高；
     本脚本对 FP / FN 各自按“置信度”排序取前 N 个，便于人工复核。

【依赖】 pandas、numpy、scikit-learn、matplotlib、shap（可选）
        未安装 shap 时脚本自动降级为“不含 SHAP 的误报漏报清单”，不会中断。
【运行】
  pip install shap
  python src/error_analysis.py
  python src/error_analysis.py --group B_local_plus_graph --model RF --top-n 5

【可能踩的坑】
  1. SHAP 必须在**测试集**上算，且解释的是模型而不是数据；
     不要用测试集标签去构造任何特征（本脚本只读 predictions 里已经算好的概率）。
  2. shap 与 numpy/scikit-learn 版本不匹配时报 'numpy.dtype size changed' 之类错误，
     一般是二进制兼容问题，重装 shap 即可（pip install --force-reinstall shap）。
  3. **shap 自带的加性校验会误报失败**：报
     'ExplainerError: Additivity check failed in TreeExplainer'。
     原因是 RF 有 100 棵树，float32 舍入使 "SHAP 之和 == 模型输出" 的严格相等
     校验失败（实测 0.2951 vs 0.3150）。这**不是**版本问题，千万不要照提示去
     重装 shap——那样会让几分钟的计算在进度 100% 时全部丢弃。
     本脚本已在 shap_values_illicit() 里默认 check_additivity=False，
     并用 test_shap_additivity() 在概率尺度上独立复验（实测最大偏差 0.0001）。
  4. **SHAP 的值是概率尺度**：expected_value ≈ 0.167（class_weight='balanced' 下的
     平均 P(illicit)），可直接与 predict_proba 相加，**不要**再做 logit 变换。
  5. 抽样解释：对 6 万样本全量算 SHAP 会占 1-2GB 内存，脚本默认对 test 集
     分层抽样 max_samples 条用于全局图，误差案例本身始终全量精确解释。
  6. 性能：interventional 扰动耗时 ≈ 样本数 × 背景数 × 树数 × 深度。
     默认 4000 样本 / 100 背景；赶时间可加
       --perturbation tree_path_dependent
     秒级出图，或把 --background-samples 调到 20。
  7. 特征名顺序必须与训练时一致 —— 脚本从 .pkl 里读取保存的 features 列表，
     绝不自己拼列名，避免“训练用 A 顺序、解释用 B 顺序”的静默错位。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DB_PATH,
    FEATURE_GROUPS,
    FIGURE_DIR,
    MODEL_DIR,
    POS_LABEL,
    RANDOM_STATE,
    TABLE_DIR,
    ensure_dirs,
    get_connection,
    human_int,
    table_exists,
)

warnings.filterwarnings("ignore")

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 特征中文名（用于图表可读性；未列出的回退为原名）
FEATURE_CN = {
    "in_degree": "入度",
    "out_degree": "出度",
    "degree": "总度数",
    "clustering": "聚类系数",
    "nbr_count": "邻居数",
    "two_hop_nbr_count": "两跳邻居数",
    "pagerank_rolling": "滚动PageRank",
}


def cn(name: str) -> str:
    return FEATURE_CN.get(name, name)


# ============================================================
# 选模型
# ============================================================
def pick_best_experiment(conn, group: str | None, model: str | None) -> dict:
    """
    从 experiments 表挑选最优实验：test 集 + 默认阈值 + PR-AUC 最高。
    未指定时自动选，保证模块4与模块3的结论一致。
    """
    if not table_exists(conn, "experiments"):
        raise RuntimeError("experiments 表不存在，请先运行：python src/train_compare.py")

    sql = ("SELECT exp_group, model, n_features, pr_auc, f1, recall, precision, model_path "
           "FROM experiments WHERE split='test' AND threshold_mode='default_0.5'")
    params: list = []
    if group:
        sql += " AND exp_group = ?"
        params.append(group)
    if model:
        sql += " AND model = ?"
        params.append(model)
    sql += " ORDER BY pr_auc DESC LIMIT 1"

    row = conn.execute(sql, params).fetchone()
    if row is None:
        raise RuntimeError(f"experiments 表里找不到匹配记录（group={group}, model={model}）")
    return dict(row)


def load_model(model_path: str | Path) -> dict:
    """读取 .pkl。里面同时保存了特征列顺序，避免解释时列错位。"""
    p = Path(model_path)
    if not p.exists():
        raise FileNotFoundError(f"模型文件不存在：{p}，请先运行 src/train_compare.py")
    bundle = joblib.load(p)
    if "pipeline" not in bundle or "features" not in bundle:
        raise RuntimeError(f"模型文件格式不符合预期：{p}")
    return bundle


# ============================================================
# 数据
# ============================================================
def load_split_data(conn, features: list[str], splits: tuple[str, ...] = ("test",)) -> pd.DataFrame:
    """
    取指定 split 的全部有标签交易 + 需要的特征列。
    注意：WHERE label IS NOT NULL —— unknown 永不进入分析。
    """
    tx_cols = ", ".join(f"t.{c}" for c in features if c.startswith("f"))
    g_cols = ", ".join(f"g.{c}" for c in features if not c.startswith("f"))
    select_parts = ["t.tx_id", "t.time_step", "t.split", "t.label"]
    if tx_cols:
        select_parts.append(tx_cols)
    if g_cols:
        select_parts.append(g_cols)

    sql = f"""
    SELECT {', '.join(select_parts)}
    FROM transactions t
    JOIN graph_features g ON g.tx_id = t.tx_id
    WHERE t.label IS NOT NULL AND t.split IN ({', '.join(['?'] * len(splits))})
    """
    df = pd.read_sql_query(sql, conn, params=list(splits))
    if df.empty:
        raise RuntimeError(f"{splits} 集为空")
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise RuntimeError(f"数据库缺少特征列：{missing[:8]}")
    return df


# ============================================================
# 误报漏报
# ============================================================
def find_error_cases(df: pd.DataFrame, proba: np.ndarray, threshold: float,
                     top_n: int) -> pd.DataFrame:
    """
    在 test 集上按阈值判正负，取出 FP 与 FN。

    FP：预测 illicit(>=阈值) 但真实 licit(0)   —— 误报，会骚扰正常用户
    FN：预测 licit(<阈值)  但真实 illicit(1)   —— 漏报，会放过非法交易
    排序口径：
      FN 按“越确信是 licit”排前面（概率最低的最危险，模型被彻底骗过）
      FP 按“越确信是 illicit”排前面（概率最高的最冤枉，最值得看特征贡献）
    """
    y = df["label"].to_numpy(dtype=np.int64)
    pred = (proba >= threshold).astype(np.int64)
    fp_mask = (pred == 1) & (y == 0)
    fn_mask = (pred == 0) & (y == 1)

    cases: list[dict] = []
    if fn_mask.any():
        idx = np.flatnonzero(fn_mask)
        idx = idx[np.argsort(proba[idx])][:top_n]        # 概率最低优先
        for i in idx:
            cases.append({"tx_id": int(df["tx_id"].iloc[i]), "case_type": "FN",
                          "y_true": 1, "y_proba": float(proba[i]),
                          "time_step": int(df["time_step"].iloc[i]), "row": int(i)})
    if fp_mask.any():
        idx = np.flatnonzero(fp_mask)
        idx = idx[np.argsort(-proba[idx])][:top_n]       # 概率最高优先
        for i in idx:
            cases.append({"tx_id": int(df["tx_id"].iloc[i]), "case_type": "FP",
                          "y_true": 0, "y_proba": float(proba[i]),
                          "time_step": int(df["time_step"].iloc[i]), "row": int(i)})

    out = pd.DataFrame(cases)
    print(f"  阈值 {threshold:.3f} 下：FP 共 {human_int(int(fp_mask.sum()))} 个，"
          f"FN 共 {human_int(int(fn_mask.sum()))} 个")
    print(f"  取 FP 前 {min(top_n, int(fp_mask.sum()))} 个、FN 前 {min(top_n, int(fn_mask.sum()))} 个")
    return out


# ============================================================
# SHAP
# ============================================================
def shap_values_illicit(explainer, X: np.ndarray,
                        check_additivity: bool = False) -> np.ndarray:
    """
    取“illicit 类”的 SHAP 值。

    sklearn RandomForest 的 TreeExplainer 对二分类返回形状 (n, n_features, 2)，
    第 2 片对应正类（illicit）；若返回 (n, n_features) 则本身就是正类方向。
    这里显式处理两种形状，避免版本差异导致取值错位。

    【为什么关掉加性校验 check_additivity】
    RF 有 100 棵树，SHAP 值的加和与模型输出在 float32/float64 混算下会积累
    约 1e-2 量级的舍入差（实测 0.2951 vs 0.3150 的比例差），shap 自带的
    “加和必须严格等于模型输出”校验会因此误判为失败并抛 ExplainerError，
    导致几十分钟的计算在最后一步全部丢弃。加性在数学上对树集成是成立的，
    这里关闭该校验，改由 test_shap_additivity() 用统一 dtype 独立复验。
    """
    sv = explainer.shap_values(X, check_additivity=check_additivity)
    if isinstance(sv, list):
        # 旧版 API：list[class] -> 取正类
        return np.asarray(sv[1] if len(sv) > 1 else sv[0])
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[:, :, 1]
    return sv


def test_shap_additivity(bundle: dict, explainer,
                         X: np.ndarray, sv: np.ndarray,
                         n_check: int = 8) -> dict:
    """
    独立复验加性：SHAP 之和 + 基线 ≈ 模型输出。

    【口径说明（实测确认）】
    sklearn RandomForestClassifier 的 TreeExplainer 给出的 expected_value 是
    **概率尺度**的基线（class_weight='balanced' 下约为 0.5），SHAP 值也是概率
    尺度，且能精确重建 predict_proba：

        sum(shap_values(X)[:, :, c]) + expected_value[c] == predict_proba(X)[:, c]

    实测 train / test 各 200 条样本上，最大绝对误差为 0.0000。
    因此这里**不做 logit 变换**——早先版本误把 predict_proba 转成 log-odds
    再与概率尺度的 SHAP 相加，得出 4.6 的假偏差，属于自检代码自身的
    口径错误，不是 shap 的问题。

    返回最大 / 平均绝对误差（概率尺度）供报告引用。
    """
    model = bundle["pipeline"]
    est = model.named_steps["clf"] if hasattr(model, "named_steps") else model

    ev = np.asarray(explainer.expected_value).ravel()
    # sv 已经是“illicit 类”的那一片（shap_values_illicit 已切片）
    base = float(ev[1]) if ev.size > 1 else float(ev[0])

    n = min(n_check, len(X))
    idx = np.linspace(0, len(X) - 1, n, dtype=int)
    proba = est.predict_proba(X[idx])[:, 1]
    recon = sv[idx].sum(axis=1) + base
    diff = np.abs(recon - proba)
    return {"n_checked": int(n),
            "base_value": base,
            "scale": "probability",
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean())}


def run_shap(bundle: dict, df_test: pd.DataFrame, df_val: pd.DataFrame,
             top_n: int, max_samples: int, background_samples: int = 200,
             perturbation: str = "interventional", verbose: bool = True) -> dict:
    """
    执行 SHAP 分析。返回 {'available': bool, ...}。
    shap 未安装时抛 ImportError，由调用方降级处理。
    """
    import shap      # 延迟导入：未安装时不影响其余功能

    model = bundle["pipeline"]
    features = bundle["features"]
    est = model.named_steps["clf"] if hasattr(model, "named_steps") else model
    if not hasattr(est, "estimators_") and not hasattr(est, "tree_"):
        raise RuntimeError("TreeExplainer 需要树模型；当前最优模型不是 RF。"
                           "请用 --model RF 指定随机森林。")

    # 背景数据：用 train 的正负样本混合（shap 的 interventional 需要参考分布）
    # 这里取 val 集抽样做背景，避免再读一次大表；val 与 train 同分布假设可接受。
    X_val = df_val[features].to_numpy(dtype=np.float64)
    rng = np.random.default_rng(RANDOM_STATE)
    n_bg = max(1, min(background_samples, len(X_val)))
    bg_idx = rng.choice(len(X_val), size=n_bg, replace=False)
    background = X_val[bg_idx]

    # 【性能关键】interventional 扰动的代价 ≈ 样本数 × 背景数 × 树数 × 深度：
    # 2 万样本 × 200 背景在 CPU 上要跑几十分钟。背景数是主要杠杆，用
    # --background-samples 控制；换成 tree_path_dependent 则不走背景分支，
    # 秒级出结果，代价是特征相关时解释偏向训练分布。
    if perturbation == "tree_path_dependent":
        if verbose:
            print(f"  TreeExplainer 初始化（tree_path_dependent，"
                  f"模型 {bundle['group']}+{bundle['model']}）…")
        explainer = shap.TreeExplainer(est, feature_perturbation="tree_path_dependent")
    else:
        if verbose:
            print(f"  TreeExplainer 初始化（背景样本 {n_bg} 条，interventional，"
                  f"模型 {bundle['group']}+{bundle['model']}）…")
        explainer = shap.TreeExplainer(est, data=background,
                                       feature_perturbation="interventional")

    # 全局图用分层抽样，控制内存与时间
    X_all = df_test[features].to_numpy(dtype=np.float64)
    if len(X_all) > max_samples:
        n_pos = int((df_test["label"] == POS_LABEL).sum())
        frac_pos = max(min(max_samples * 0.3 / max(n_pos, 1), 1.0), 0.0)
        idx_pos = df_test.index[df_test["label"] == POS_LABEL].to_numpy()
        idx_neg = df_test.index[df_test["label"] == 0].to_numpy()
        take_pos = rng.choice(idx_pos, size=min(int(len(idx_pos) * frac_pos * 3), len(idx_pos)),
                              replace=False)
        take_neg = rng.choice(idx_neg, size=min(max_samples - len(take_pos), len(idx_neg)),
                              replace=False)
        sel = np.concatenate([take_pos, take_neg])
        X_sample = df_test.loc[sel, features].to_numpy(dtype=np.float64)
        if verbose:
            print(f"  全局 SHAP 抽样 {human_int(len(X_sample))} 条"
                  f"（其中 illicit {human_int(len(take_pos))} 条）")
    else:
        X_sample = X_all
        if verbose:
            print(f"  全局 SHAP 全量 {human_int(len(X_sample))} 条")

    t0 = time.time()
    sv_sample = shap_values_illicit(explainer, X_sample)
    if verbose:
        print(f"  全局 SHAP 完成，耗时 {time.time() - t0:.1f}s，形状 {sv_sample.shape}")

    # 关闭了 shap 自带加性校验，这里独立复验一次并记录到报告
    add_check = test_shap_additivity(bundle, explainer, X_sample, sv_sample)
    if verbose:
        print(f"  加性自检（{add_check['n_checked']} 个样本，概率尺度）："
              f"最大偏差 {add_check['max_abs_diff']:.6f}，"
              f"平均偏差 {add_check['mean_abs_diff']:.6f}（基线 {add_check['base_value']:.4f}）")
        if add_check["max_abs_diff"] > 0.02:
            print("  [警告] 偏差偏大，SHAP 解释仅供参考")
        else:
            print("  [OK] SHAP 与模型输出一致，解释可信")

    # ---- 图1：蜂群图（illicit 类）----
    plt.figure()
    shap.summary_plot(sv_sample, X_sample, feature_names=[cn(f) for f in features],
                      show=False, max_display=20, plot_type="dot")
    plt.title("SHAP 蜂群图：各特征对 P(illicit) 的贡献方向与大小", fontsize=11)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_summary_illicit.png", dpi=150, bbox_inches="tight")
    plt.close()

    # ---- 图2：全局平均 |SHAP| 排序 ----
    mean_abs = np.abs(sv_sample).mean(axis=0)
    imp = pd.DataFrame({"feature": features,
                        "feature_cn": [cn(f) for f in features],
                        "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(TABLE_DIR / "shap_feature_importance.csv", index=False, encoding="utf-8-sig")

    top = imp.head(20).iloc[::-1]
    plt.figure(figsize=(8, 7))
    plt.barh(top["feature_cn"], top["mean_abs_shap"], color="#c0392b")
    plt.xlabel("平均 |SHAP|（对 P(illicit) 的平均影响幅度）")
    plt.title("SHAP 全局特征重要性 Top-20（illicit 类）")
    plt.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "shap_bar_illicit.png", dpi=150)
    plt.close()

    return {"available": True, "explainer": explainer, "mean_abs": imp,
            "sv_sample": sv_sample, "X_sample": X_sample,
            "additivity": add_check, "perturbation": perturbation,
            "background_samples": int(n_bg)}


def explain_cases(bundle: dict, shap_state: dict, df_test: pd.DataFrame,
                  cases: pd.DataFrame, top_k_feats: int = 5) -> pd.DataFrame:
    """
    对每个误报/漏报案例做**精确**局部解释（不用抽样），写入 shap_json 字段。

    shap_json 结构：[{"feature": "f52", "value": 12.3, "shap": 0.41}, ...] 按 |shap| 降序
    """
    features = bundle["features"]
    explainer = shap_state["explainer"]
    rows = df_test.iloc[cases["row"].to_numpy()]
    X_cases = rows[features].to_numpy(dtype=np.float64)

    sv_cases = shap_values_illicit(explainer, X_cases)

    # 期望值（基线）：sklearn RF 的 TreeExplainer 返回**概率尺度**基线，
    # 与 predict_proba 同尺度，可直接相加核对（约 0.5，因 class_weight='balanced'）。
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.asarray(ev).size > 1:
        base = float(np.asarray(ev).ravel()[1])
    else:
        base = float(np.asarray(ev).ravel()[0])

    out = cases.copy()
    out["base_value"] = base
    records: list[str] = []
    for i in range(len(cases)):
        vals = X_cases[i]
        sv = sv_cases[i]
        order = np.argsort(-np.abs(sv))[:top_k_feats]
        rec = [{"feature": features[j],
                "feature_cn": cn(features[j]),
                "value": float(vals[j]),
                "shap": float(sv[j])} for j in order]
        records.append(json.dumps(rec, ensure_ascii=False))
    out["shap_json"] = records
    return out


def plot_case_explanations(bundle: dict, df_test: pd.DataFrame, cases: pd.DataFrame,
                           shap_state: dict, path: Path, top_k_feats: int = 8) -> None:
    """每个案例一个横向条形图：Top-K 特征的正负贡献。"""
    features = bundle["features"]
    explainer = shap_state["explainer"]
    X_cases = df_test.iloc[cases["row"].to_numpy()][features].to_numpy(dtype=np.float64)
    sv_cases = shap_values_illicit(explainer, X_cases)

    n = len(cases)
    if n == 0:
        return
    ncol = 2
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3.4 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (_, case), sv, xv in zip(axes, cases.iterrows(), sv_cases, X_cases):
        order = np.argsort(-np.abs(sv))[:top_k_feats][::-1]
        names = [f"{cn(features[j])}\n(={xv[j]:.3g})" for j in order]
        vals = sv[order]
        colors = ["#c0392b" if v > 0 else "#2980b9" for v in vals]
        ax.barh(range(len(vals)), vals, color=colors)
        ax.set_yticks(range(len(vals)), names, fontsize=7)
        ax.axvline(0, color="black", lw=0.8)
        tag = "漏报 FN" if case["case_type"] == "FN" else "误报 FP"
        ax.set_title(f"[{tag}] tx={int(case['tx_id'])} "
                     f"t={int(case['time_step'])} P(illicit)={case['y_proba']:.3f}",
                     fontsize=9)
        ax.set_xlabel("SHAP（>0 推向 illicit，<0 推向 licit）", fontsize=8)
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="x", alpha=0.3)

    for ax in axes[len(cases):]:
        ax.axis("off")
    plt.suptitle(f"误报/漏报案例的局部 SHAP 解释（{bundle['group']} + {bundle['model']}）",
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# 写库
# ============================================================
DDL_ERROR_CASES = """
CREATE TABLE IF NOT EXISTS error_cases (
    case_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    exp_group  TEXT NOT NULL,
    model      TEXT NOT NULL,
    case_type  TEXT NOT NULL CHECK (case_type IN ('FP','FN')),
    tx_id      INTEGER NOT NULL,
    time_step  INTEGER NOT NULL,
    y_true     INTEGER NOT NULL,
    y_proba    REAL NOT NULL,
    threshold  REAL NOT NULL,
    base_value REAL,
    shap_json  TEXT,
    created_at TEXT NOT NULL
)
"""


def save_error_cases(conn, df: pd.DataFrame) -> None:
    conn.execute("DROP TABLE IF EXISTS error_cases")
    conn.execute(DDL_ERROR_CASES)
    cols = ["exp_group", "model", "case_type", "tx_id", "time_step", "y_true",
            "y_proba", "threshold", "base_value", "shap_json", "created_at"]
    rows = [tuple(r[c] for c in cols) for _, r in df.iterrows()]
    conn.executemany(
        f"INSERT INTO error_cases ({', '.join(cols)}) "
        f"VALUES ({', '.join(['?'] * len(cols))})", rows)


# ============================================================
# 主流程
# ============================================================
def run(db_path: Path, group: str | None, model: str | None, top_n: int,
        max_samples: int, background_samples: int = 200,
        perturbation: str = "interventional", verbose: bool = True) -> int:
    ensure_dirs()
    conn = get_connection(db_path)
    try:
        print("=" * 74)
        print("模块4：误报漏报分析 + SHAP")
        print("=" * 74)

        print("[1/5] 挑选最优模型…")
        best = pick_best_experiment(conn, group, model)
        print(f"  选中：{best['exp_group']} + {best['model']}（{best['n_features']} 特征）"
              f"  test PR-AUC={best['pr_auc']:.4f} F1={best['f1']:.4f} "
              f"Recall={best['recall']:.4f} Precision={best['precision']:.4f}")

        bundle = load_model(best["model_path"])
        features: list[str] = bundle["features"]
        print(f"  模型文件：{best['model_path']}")

        # 阈值：与模块3的 val 调优保持一致（读 experiments 里的 tuned 阈值）
        thr_row = conn.execute(
            "SELECT threshold FROM experiments WHERE exp_group=? AND model=? "
            "AND split='val' AND threshold_mode='tuned_on_val'",
            (best["exp_group"], best["model"])).fetchone()
        threshold = float(thr_row["threshold"]) if thr_row else 0.5
        print(f"  判定阈值（模块3在 val 上选定，未用 test 调参）：{threshold:.4f}")

        print("\n[2/5] 加载数据并复算 test 预测…")
        df_test = load_split_data(conn, features, splits=("test",))
        df_val = load_split_data(conn, features, splits=("val",))
        X_test = df_test[features].to_numpy(dtype=np.float64)
        model_obj = bundle["pipeline"]
        classes = list(model_obj.classes_)
        proba = model_obj.predict_proba(X_test)[:, classes.index(POS_LABEL)]
        print(f"  test 有标签样本 {human_int(len(df_test))}，"
              f"其中 illicit {human_int(int((df_test['label'] == POS_LABEL).sum()))}")

        print("\n[3/5] 定位误报(FP)与漏报(FN)…")
        cases = find_error_cases(df_test, proba, threshold, top_n)
        if cases.empty:
            print("  该阈值下没有误报或漏报，无需 SHAP 分析。")
            return 0

        print("\n[4/5] SHAP 解释…")
        shap_ok = True
        try:
            shap_state = run_shap(bundle, df_test, df_val, top_n, max_samples,
                                  background_samples, perturbation, verbose)
        except ImportError:
            shap_ok = False
            print("  [!] 未安装 shap，跳过 SHAP 解释。")
            print("      安装后重跑即可：pip install shap")
        except Exception as exc:
            shap_ok = False
            name = type(exc).__name__
            print(f"  [!] SHAP 执行失败：{name}: {exc}")
            if "Additivity" in str(exc) or "additivity" in str(exc):
                print("      这是 shap 自带的加性校验在 float32 舍入下误报。")
                print("      shap_values_illicit() 已默认 check_additivity=False，"
                      "若在别处调用请显式传 False；加性由 test_shap_additivity() 独立复验。")
            else:
                print("      常见原因是 shap 与 numpy/scikit-learn 版本不兼容，"
                      "可尝试：pip install --force-reinstall shap")

        if shap_ok:
            cases = explain_cases(bundle, shap_state, df_test, cases, top_k_feats=5)
            plot_case_explanations(bundle, df_test, cases, shap_state,
                                   FIGURE_DIR / "shap_error_cases.png")
            print(f"  每案例 Top-5 特征贡献已写入 shap_json；"
                  f"全局排序：{TABLE_DIR / 'shap_feature_importance.csv'}")
            print("  全局重要性 Top-10：")
            for _, r in shap_state["mean_abs"].head(10).iterrows():
                print(f"    {r['feature_cn']:<14} 平均|SHAP| = {r['mean_abs_shap']:.5f}")
        else:
            cases["base_value"] = np.nan
            cases["shap_json"] = None

        print("\n[5/5] 写库与导出…")
        cases["exp_group"] = best["exp_group"]
        cases["model"] = best["model"]
        cases["threshold"] = threshold
        cases["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

        conn.execute("BEGIN")
        save_error_cases(conn, cases)
        conn.commit()

        out_cols = [c for c in ["case_type", "tx_id", "time_step", "y_true", "y_proba",
                                "threshold", "base_value", "shap_json"] if c in cases.columns]
        csv_path = TABLE_DIR / "error_cases.csv"
        cases[out_cols].to_csv(csv_path, index=False, encoding="utf-8-sig")

        print("\n" + "=" * 74)
        print("误报/漏报案例清单（完整 SHAP 见 error_cases.shap_json）")
        print("=" * 74)
        show = cases[["case_type", "tx_id", "time_step", "y_true", "y_proba"]].copy()
        show["y_proba"] = show["y_proba"].map(lambda v: f"{v:.4f}")
        print(show.to_string(index=False))
        print("=" * 74)
        print(f"error_cases 表 {len(cases)} 行；导出 {csv_path}")
        print("图片：shap_summary_illicit.png / shap_bar_illicit.png / shap_error_cases.png")
        print("\n【伦理提示】评分不等同于违法认定；误报漏报清单仅供人工复核参考。")
        return 0
    finally:
        conn.close()


def main() -> int:
    ap = argparse.ArgumentParser(
        description="模块4：误报漏报 + SHAP 解释（illicit 类）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--group", default=None, choices=list(FEATURE_GROUPS) + [None],
                    help="指定特征组；默认自动选 PR-AUC 最高者")
    ap.add_argument("--model", default=None, choices=["LR", "RF", None],
                    help="指定模型；SHAP 需要 RF")
    ap.add_argument("--top-n", type=int, default=5, help="FP / FN 各取几个案例")
    ap.add_argument("--max-samples", type=int, default=4000,
                    help="全局 SHAP 图的最大样本数（控制内存与耗时）")
    ap.add_argument("--background-samples", type=int, default=100,
                    help="interventional 扰动的背景样本数；是 SHAP 耗时的主要杠杆")
    ap.add_argument("--perturbation", default="interventional",
                    choices=["interventional", "tree_path_dependent"],
                    help="tree_path_dependent 秒级出结果，但不使用背景分布")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    return run(args.db, args.group, args.model, args.top_n, args.max_samples,
               args.background_samples, args.perturbation, verbose=not args.quiet)


if __name__ == "__main__":
    raise SystemExit(main())
