"""
train_compare.py —— 模块3：模型训练 + 对比实验
=============================================

【输入】
  elliptic.db
    - transactions(tx_id, time_step, f1..f165, label, split)   label 为 NULL 的是 unknown
    - graph_features(结构类图特征)                              由 graph_features.py 生成

【三组特征（本次对齐后的方案）】
  A_local            = f1..f94                 纯局部特征（Elliptic 官方“local”部分）
  B_local_plus_graph = A + 结构图特征           在 A 之上加图结构信息
  C_official_full    = f1..f165                官方全量特征（f95..f165 已含 1-hop 聚合）
  说明：A → B 回答“结构图特征有没有增量价值”；A → C 回答“官方聚合特征值多少”。
        三者一起才能支撑论文里“图特征到底有没有用”的论证。

【模型】
  LogisticRegression(class_weight='balanced')  —— 训练前必须 StandardScaler
  RandomForestClassifier(class_weight='balanced') —— 树模型不需要标准化

【硬约束落实】
  1) unknown 不参与训练与指标：SQL 里 WHERE label IS NOT NULL，标签缺失的行根本不进 X/y。
  2) 划分 1-29 / 30-34 / 35-49，全部来自数据库里的 split 列，脚本内不做任何随机划分。
  3) 只用 A 组的局部特征时不引入任何图信息；B 组的图特征本身不含标签。
  4) class_weight='balanced'，指标重点 PR-AUC 与 F1。
  5) 阈值只在 val 上挑，绝不拿 test 调参。

【输出】
  elliptic.db → experiments 表（逐行记录每次实验）
  elliptic.db → predictions 表（test/val 上每条交易的概率，供模块4复用）
  outputs/models/{group}_{model}.pkl
  outputs/tables/experiments.csv
  outputs/figures/pr_curves.png、confusion_matrices.png、model_comparison.png

【依赖】 pandas、numpy、scikit-learn、matplotlib、joblib
【运行】 python src/train_compare.py
         python src/train_compare.py --groups A_local B_local_plus_graph
【耗时】本机实测：LR 秒级；RF 数分钟（200 棵树上限，可调）。

【可能踩的坑】
  1. 用 train_test_split 随机划分 → 直接违反硬约束 2，本脚本不导入该函数。
  2. unknown 当 0 用 → 训练标签被污染；本脚本用 IS NOT NULL 过滤，并在打印里显示被排除数量。
  3. LR 不标准化 → 收敛慢且系数不可比；用 Pipeline 把 scaler 与模型绑在一起保存。
  4. 用 test 选阈值/调参 → 典型泄漏；阈值只在 val 上选。
  5. accuracy 在不平衡数据上没有意义（全预测 licit 也有 ~89%），必须看 PR-AUC / F1。
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

matplotlib.use("Agg")     # 无显示环境（脚本/服务器）下必须用 Agg，否则 plt.show 会阻塞
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CLASS_WEIGHT,
    DB_PATH,
    FEATURE_GROUPS,
    FIGURE_DIR,
    GRAPH_FEATS,
    LOCAL_FEATS,
    MODEL_DIR,
    N_FEATURES,
    OFFICIAL_FEATS,
    POS_LABEL,
    RANDOM_STATE,
    TABLE_DIR,
    ensure_dirs,
    get_connection,
    human_int,
    table_exists,
)

warnings.filterwarnings("ignore", category=FutureWarning)

# 中文图表标签（Windows 自带 SimHei / Microsoft YaHei）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# RF 训练上限：毕设阶段够用，且笔记本上可控时间
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=1,
    class_weight=CLASS_WEIGHT,
    random_state=RANDOM_STATE,
    n_jobs=1,          # 避免在内存受限时开多进程复制数据；可自行调大
    oob_score=True,    # 袋外分数：免费的额外诊断
)
LR_PARAMS = dict(
    C=1.0,
    class_weight=CLASS_WEIGHT,
    max_iter=2000,
    solver="lbfgs",
    random_state=RANDOM_STATE,
)


# ============================================================
# 数据加载
# ============================================================
def load_dataset(conn, verbose: bool = True) -> tuple[pd.DataFrame, list[str]]:
    """
    读取“有标签”交易 + 全部候选特征。unknown 行被 SQL 直接排除。

    返回 (df, feature_columns)：
      df 含列 tx_id, time_step, split, label, f1..f165, 以及图特征
    """
    if not table_exists(conn, "graph_features"):
        raise RuntimeError("graph_features 表不存在，请先运行：python src/graph_features.py")

    tx_cols = ", ".join(f"t.f{i}" for i in range(1, N_FEATURES + 1))
    g_cols = ", ".join(f"g.{c}" for c in GRAPH_FEATS)

    # 关键：label IS NOT NULL —— unknown(157,205 条) 完全不进入 X/y
    sql = f"""
    SELECT t.tx_id, t.time_step, t.split, t.label, {tx_cols}, {g_cols}
    FROM transactions t
    JOIN graph_features g ON g.tx_id = t.tx_id
    WHERE t.label IS NOT NULL
    """
    df = pd.read_sql_query(sql, conn)
    feature_cols = list(LOCAL_FEATS) + list(GRAPH_FEATS)
    # 官方全量特征 f1..f165 也在 df 里（上面 tx_cols 已取全），供 C 组使用
    for c in OFFICIAL_FEATS:
        if c not in df.columns:
            raise RuntimeError(f"缺少特征列 {c}")

    if verbose:
        n_unknown = int(conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE label IS NULL").fetchone()[0])
        print(f"  已加载有标签交易 {human_int(len(df))} 条")
        print(f"  按硬约束 1 排除 unknown(NULL) {human_int(n_unknown)} 条，不参与训练与指标")
        print("\n  各 split 的有标签样本数（模块3要求：报告每组样本数）")
        for sp in ("train", "val", "test"):
            sub = df[df["split"] == sp]
            n_pos = int((sub["label"] == POS_LABEL).sum())
            n_neg = int((sub["label"] == 0).sum())
            ratio = n_pos / max(n_neg, 1)
            print(f"    {sp:<6} 合计 {human_int(len(sub)):>9}  "
                  f"illicit(1) {human_int(n_pos):>7}  licit(0) {human_int(n_neg):>7}  "
                  f"正负比 1:{n_neg / max(n_pos, 1):.2f}  (正样本占比 {ratio:.4f})")
    return df, feature_cols


def split_xy(df: pd.DataFrame, group: str, verbose: bool = True
             ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    按 split 列切 X/y（不随机！）。

    关键点：
      - 特征列由 FEATURE_GROUPS[group] 决定，A 组绝不掺入图特征。
      - 缺失值统一填 0（图特征在模块2已填，这里再兜底一次）。
      - 加载时为了省内存一次性取了全部候选列，这里再按需裁剪，
        并顺手报告缺口，防止“以为在用图特征其实列没取到”这类静默错误。
    """
    cols = list(FEATURE_GROUPS[group])
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"特征组 {group} 缺少列：{missing[:10]}"
                           f"{' 等' if len(missing) > 10 else ''}")

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for sp in ("train", "val", "test"):
        sub = df[df["split"] == sp]
        if sub.empty:
            raise RuntimeError(f"{sp} 为空，划分有误")
        X = sub[cols].to_numpy(dtype=np.float64, copy=True)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = sub["label"].to_numpy(dtype=np.int64, copy=True)
        out[sp] = (X, y)
    return out


# ============================================================
# 模型
# ============================================================
def build_model(model_name: str):
    """
    LR 必须标准化，所以包成 Pipeline（scaler 随模型一起保存，避免线上忘记标准化）。
    RF 不做标准化（树模型对单调变换不敏感），直接放模型。
    """
    if model_name == "LR":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(**LR_PARAMS)),
        ])
    if model_name == "RF":
        return RandomForestClassifier(**RF_PARAMS)
    raise ValueError(f"未知模型：{model_name}")


def pos_proba(model, X: np.ndarray) -> np.ndarray:
    """取“illicit(=1)”这一列的概率。"""
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    if POS_LABEL not in classes:
        raise RuntimeError(f"模型未学到正类 {POS_LABEL}，classes_={classes}")
    return proba[:, classes.index(POS_LABEL)]


def evaluate(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """
    一次性算出所有指标。指标口径固定，避免不同实验之间不可比。

    pr_auc 用 average_precision_score（AP），对不平衡数据比 ROC-AUC 敏感得多。
    f1/precision/recall 都是**针对 illicit 正类**计算的（pos_label=1）。
    """
    y_pred = (proba >= threshold).astype(np.int64)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, pos_label=POS_LABEL, zero_division=0)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_samples": int(len(y_true)),
        "n_pos": int((y_true == POS_LABEL).sum()),
    }


def best_f1_threshold(y_true: np.ndarray, proba: np.ndarray) -> float:
    """
    在 val 上挑 F1 最大的阈值。

    只在 val 上挑 —— test 只被“报告”，不被“挑选”，这是不泄漏的关键。
    """
    prec, rec, thr = precision_recall_curve(y_true, proba, pos_label=POS_LABEL)
    # precision_recall_curve 返回的 thr 长度比 prec/rec 少 1
    f1 = np.divide(2 * prec * rec, prec + rec,
                   out=np.zeros_like(prec), where=(prec + rec) > 0)
    if thr.size == 0:
        return 0.5
    best = int(np.argmax(f1[:-1])) if f1.size == thr.size + 1 else int(np.argmax(f1))
    return float(thr[min(best, thr.size - 1)])


# ============================================================
# experiments 表
# ============================================================
DDL_EXPERIMENTS = """
CREATE TABLE IF NOT EXISTS experiments (
    exp_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at  TEXT    NOT NULL,
    exp_group   TEXT    NOT NULL,     -- A_local / B_local_plus_graph / C_official_full
    model       TEXT    NOT NULL,     -- LR / RF
    n_features  INTEGER NOT NULL,
    threshold_mode TEXT NOT NULL,     -- default_0.5 / tuned_on_val
    split       TEXT    NOT NULL,     -- val / test
    threshold   REAL,
    "precision" REAL, recall REAL, f1 REAL, pr_auc REAL, roc_auc REAL,
    tn INTEGER, fp INTEGER, fn INTEGER, tp INTEGER,
    n_samples   INTEGER, n_pos INTEGER,
    n_train     INTEGER, n_val INTEGER, n_test INTEGER,
    train_seconds REAL, oob_score REAL,
    model_path  TEXT
)
"""

DDL_PREDICTIONS = """
CREATE TABLE IF NOT EXISTS predictions (
    exp_group TEXT NOT NULL,
    model     TEXT NOT NULL,
    tx_id     INTEGER NOT NULL,
    split     TEXT NOT NULL,
    y_true    INTEGER NOT NULL,
    y_proba   REAL NOT NULL,
    PRIMARY KEY (exp_group, model, tx_id)
)
"""


def save_experiments(conn, rows: list[dict]) -> None:
    conn.execute("DROP TABLE IF EXISTS experiments")
    conn.execute(DDL_EXPERIMENTS)
    if not rows:
        return
    # exp_id 由数据库自增，不写
    cols = [c for c in rows[0].keys() if c != "exp_id"]
    sql = (f"INSERT INTO experiments ({', '.join(cols)}) "
           f"VALUES ({', '.join(['?'] * len(cols))})")
    conn.executemany(sql, [tuple(r[c] for c in cols) for r in rows])
    conn.commit()


def save_predictions(conn, pred_rows: list[tuple]) -> None:
    conn.execute("DROP TABLE IF EXISTS predictions")
    conn.execute(DDL_PREDICTIONS)
    if pred_rows:
        conn.executemany(
            "INSERT INTO predictions (exp_group, model, tx_id, split, y_true, y_proba) "
            "VALUES (?, ?, ?, ?, ?, ?)", pred_rows)
    conn.commit()


# ============================================================
# 绘图
# ============================================================
def plot_pr_curves(curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]],
                   path: Path) -> None:
    plt.figure(figsize=(8, 6.5))
    for (group, model), (y, proba, ap) in curves.items():
        prec, rec, _ = precision_recall_curve(y, proba, pos_label=POS_LABEL)
        plt.plot(rec, prec, lw=1.6, label=f"{group} + {model}  (AP={ap:.4f})")
    base = next(iter(curves.values()))[0]
    pos_rate = float((base == POS_LABEL).mean())
    plt.axhline(pos_rate, ls="--", c="gray", lw=1,
                label=f"随机基线 (正类占比={pos_rate:.4f})")
    plt.xlabel("Recall（illicit 召回率）")
    plt.ylabel("Precision（illicit 精确率）")
    plt.title("PR 曲线对比（test 集 35-49）")
    plt.legend(fontsize=8, loc="upper right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_confusion_matrices(results: list[dict], path: Path) -> None:
    """只画 test 集、默认阈值的结果，3 组 × 2 模型。"""
    sel = [r for r in results if r["split"] == "test" and r["threshold_mode"] == "default_0.5"]
    if not sel:
        return
    n = len(sel)
    ncol = 3
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax, r in zip(axes, sel):
        cm = np.array([[r["tn"], r["fp"]], [r["fn"], r["tp"]]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=10)
        ax.set_xticks([0, 1], ["预测 licit", "预测 illicit"], fontsize=8)
        ax.set_yticks([0, 1], ["实际 licit", "实际 illicit"], fontsize=8)
        ax.set_title(f"{r['exp_group']} + {r['model']}\n"
                     f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}",
                     fontsize=9)
    for ax in axes[len(sel):]:
        ax.axis("off")
    plt.suptitle("混淆矩阵（test 集，阈值 0.5，正类=illicit）", fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_model_comparison(results: list[dict], path: Path) -> None:
    """test 集默认阈值下，按组汇总 PR-AUC / F1 / Recall 三项主指标。"""
    sel = [r for r in results if r["split"] == "test" and r["threshold_mode"] == "default_0.5"]
    if not sel:
        return
    labels = [f"{r['exp_group']}\n{r['model']}" for r in sel]
    metrics = ["pr_auc", "f1", "recall"]
    x = np.arange(len(sel))
    width = 0.26
    plt.figure(figsize=(max(8, 1.8 * len(sel)), 5.5))
    for k, m in enumerate(metrics):
        plt.bar(x + (k - 1) * width, [r[m] for r in sel], width, label=m.upper())
    plt.xticks(x, labels, fontsize=8)
    plt.ylabel("指标值（越高越好）")
    plt.title("模型对比：PR-AUC / F1 / Recall（test 集，正类=illicit）")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


# ============================================================
# 主流程
# ============================================================
def run(db_path: Path, groups: list[str], models: list[str], verbose: bool = True) -> pd.DataFrame:
    ensure_dirs()
    conn = get_connection(db_path)
    results: list[dict] = []
    pred_rows: list[tuple] = []
    curves: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, float]] = {}

    try:
        for g in groups:
            if g not in FEATURE_GROUPS:
                raise ValueError(f"未知特征组 {g}，可选 {list(FEATURE_GROUPS)}")
        for m in models:
            if m not in ("LR", "RF"):
                raise ValueError(f"未知模型 {m}，可选 LR / RF")

        print("=" * 74)
        print("模块3：模型训练 + 对比实验")
        print("=" * 74)
        print(f"特征组：{groups}")
        print(f"模型　：{models}")
        print(f"数据库：{db_path}\n")

        print("[1/4] 加载数据…")
        df, _ = load_dataset(conn, verbose=verbose)

        n_train = int((df["split"] == "train").sum())
        n_val = int((df["split"] == "val").sum())
        n_test = int((df["split"] == "test").sum())

        print("\n[2/4] 打印各组特征数…")
        for g in groups:
            print(f"  {g:<20} {len(FEATURE_GROUPS[g]):>3} 个特征")

        print("\n[3/4] 训练与评估…")
        for g in groups:
            data = split_xy(df, g)
            Xtr, ytr = data["train"]
            Xva, yva = data["val"]
            Xte, yte = data["test"]
            print(f"\n  ── 特征组 {g}（{len(FEATURE_GROUPS[g])} 特征）"
                  f"  train {human_int(len(ytr))} / val {human_int(len(yva))} / "
                  f"test {human_int(len(yte))} ──")

            for m in models:
                t0 = time.time()
                model = build_model(m)
                model.fit(Xtr, ytr)
                train_sec = time.time() - t0

                oob = float(getattr(model, "oob_score_", np.nan)) if m == "RF" else None
                model_path = MODEL_DIR / f"{g}__{m}.pkl"
                joblib.dump(
                    {"group": g, "model": m, "features": FEATURE_GROUPS[g],
                     "pipeline": model, "pos_label": POS_LABEL},
                    model_path,
                )

                proba_va = pos_proba(model, Xva)
                proba_te = pos_proba(model, Xte)

                # 阈值只在 val 上挑
                thr = best_f1_threshold(yva, proba_va)

                for mode, th in (("default_0.5", 0.5), ("tuned_on_val", thr)):
                    for sp, y, pr in (("val", yva, proba_va), ("test", yte, proba_te)):
                        r = evaluate(y, pr, threshold=th)
                        r.update({
                            "exp_group": g, "model": m,
                            "n_features": len(FEATURE_GROUPS[g]),
                            "threshold_mode": mode, "split": sp,
                            "n_train": n_train, "n_val": n_val, "n_test": n_test,
                            "train_seconds": round(train_sec, 3),
                            "oob_score": oob, "model_path": str(model_path),
                        })
                        results.append(r)

                # 预测明细（模块4直接复用，避免重复推理）
                tx_va = df.loc[df["split"] == "val", "tx_id"].to_numpy()
                tx_te = df.loc[df["split"] == "test", "tx_id"].to_numpy()
                pred_rows += [(g, m, int(t), "val", int(y), float(p))
                              for t, y, p in zip(tx_va, yva, proba_va)]
                pred_rows += [(g, m, int(t), "test", int(y), float(p))
                              for t, y, p in zip(tx_te, yte, proba_te)]

                curves[(g, m)] = (yte, proba_te, float(average_precision_score(yte, proba_te)))

                base = next(
                    r for r in results
                    if r["exp_group"] == g and r["model"] == m
                    and r["split"] == "test" and r["threshold_mode"] == "default_0.5"
                )
                print(f"    {m:<3} 训练 {train_sec:6.1f}s  "
                      f"val_thr={thr:.3f}  "
                      f"test P={base['precision']:.4f} R={base['recall']:.4f} "
                      f"F1={base['f1']:.4f} PR-AUC={base['pr_auc']:.4f} "
                      f"ROC-AUC={base['roc_auc']:.4f}"
                      + (f"  oob={oob:.4f}" if oob is not None else ""))
                print(f"         混淆矩阵(test) TN={base['tn']:,} FP={base['fp']:,} "
                      f"FN={base['fn']:,} TP={base['tp']:,}")

        print("\n[4/4] 写库、导表、画图…")
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            r["created_at"] = now
            r["exp_id"] = None      # 占位，DB 自增

        conn.execute("BEGIN")
        try:
            save_experiments(conn, results)
            save_predictions(conn, pred_rows)
        except Exception:
            conn.rollback()
            raise

        res_df = pd.DataFrame(results)
        csv_path = TABLE_DIR / "experiments.csv"
        res_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        plot_pr_curves(curves, FIGURE_DIR / "pr_curves.png")
        plot_confusion_matrices(results, FIGURE_DIR / "confusion_matrices.png")
        plot_model_comparison(results, FIGURE_DIR / "model_comparison.png")

        # 汇总表：test + 默认阈值
        summary = res_df[(res_df["split"] == "test") &
                         (res_df["threshold_mode"] == "default_0.5")].copy()
        summary = summary.sort_values("pr_auc", ascending=False)[
            ["exp_group", "model", "n_features", "precision", "recall", "f1",
             "pr_auc", "roc_auc", "tn", "fp", "fn", "tp"]]
        print("\n" + "=" * 96)
        print("test 集对比汇总（阈值 0.5，按 PR-AUC 降序；正类=illicit）")
        print("=" * 96)
        print(summary.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
        print("=" * 96)
        print(f"experiments 表 {len(results)} 行；predictions 表 {len(pred_rows):,} 行")
        print(f"表格：{csv_path}")
        print(f"图片：{FIGURE_DIR}")
        print(f"模型：{MODEL_DIR}")

        # --- 结论提示：B 是否优于 A（对应手动检查清单 E）---
        print("\n【A/B/C 对比提示】")
        for m in models:
            def ap(g):
                s = summary[(summary["exp_group"] == g) & (summary["model"] == m)]
                return float(s["pr_auc"].iloc[0]) if len(s) else float("nan")
            a, b, c = ap("A_local"), ap("B_local_plus_graph"), ap("C_official_full")
            print(f"  {m}: PR-AUC  A={a:.4f}  B={b:.4f}  C={c:.4f}")
            if not np.isnan(a) and not np.isnan(b):
                if b > a:
                    print(f"      结构图特征带来提升 Δ(B-A) = +{b - a:.4f}")
                else:
                    print(f"      [!] B 未优于 A，Δ(B-A) = {b - a:+.4f}，"
                          f"需在论文中分析原因（如结构特征与官方聚合特征信息重叠）")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return pd.DataFrame(results)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="模块3：A/B/C 三组特征 × LR/RF 对比实验",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--db", type=Path, default=DB_PATH)
    ap.add_argument("--groups", nargs="+",
                    default=["A_local", "B_local_plus_graph", "C_official_full"],
                    choices=list(FEATURE_GROUPS))
    ap.add_argument("--models", nargs="+", default=["LR", "RF"], choices=["LR", "RF"])
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    run(args.db, args.groups, args.models, verbose=not args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
