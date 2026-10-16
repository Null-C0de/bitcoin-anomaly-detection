"""
load_data.py —— 模块1：数据加载 + SQLite 建库
=============================================

【输入】
  Data/elliptic_txs_features.csv   无表头，167 列 = txId, time_step, f1..f165
  Data/elliptic_txs_edgelist.csv   有表头 txId1,txId2
  Data/elliptic_txs_classes.csv    有表头 txId,class   （1=illicit, 2=licit, unknown）

【输出】
  elliptic.db（默认在项目根）
    ├─ transactions(tx_id PK, time_step, f1..f165, label, split)
    ├─ edges(src PK, dst PK)  WITHOUT ROWID
    ├─ view transactions_v（带 label_name 的中文友好视图）
    └─ 索引：transactions.time_step / label / split，edges.src / edges.dst

【标签编码（硬约束 1）】
  '1'       -> label = 1   (illicit，正类)
  '2'       -> label = 0   (licit)
  'unknown' -> label = NULL（绝不存 0！用 SQL NULL 表示“未知”，
                            这样 WHERE label IS NULL / IS NOT NULL 天然区分，
                            也避免被模型当成负样本）

【时间划分（硬约束 2）】
  1-29 train / 30-34 val / 35-49 test，只按 time_step 切，无随机划分。

【依赖】 pandas、numpy、sqlite3(标准库)
【运行】 python src/load_data.py
         python src/load_data.py --verify-only     # 只校验已建好的库
         python src/load_data.py --force           # 覆盖重建
【耗时】 本机实测约 1-3 分钟（features.csv 658MB，一次性事务 + executemany）

【可能踩的坑】（详见文件末尾自检）
  1. features.csv 无表头且列数是 167 不是 166，必须先手动赋列名。
  2. 使用默认的 read_csv(header=0) 会把第一行真实数据当表头吃掉，总数少 1 行。
  3. 逐行 INSERT 会慢到不可接受（20 万行 × 10 分钟级），必须 executemany。
  4. 建索引放在导入之后，否则每插一行都要维护 B 树。
  5. label 列若建成 NOT NULL 就无法表示 unknown。
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 允许 `python src/load_data.py` 直接运行时能 import 同目录的 common
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    CLASS_TO_LABEL,
    CLASSES_CSV,
    DB_PATH,
    EDGELIST_CSV,
    FEATURES_CSV,
    LABEL_TO_NAME,
    N_EDGES,
    N_FEATURES,
    N_TXS,
    SPLITS,
    count_rows,
    ensure_dirs,
    get_connection,
    human_int,
    table_exists,
    time_step_to_split,
    validate_split_edges,
)

# ============================================================
# 列名与 DDL
# ============================================================
FEATURE_COLS: list[str] = [f"f{i}" for i in range(1, N_FEATURES + 1)]
# features.csv 的物理列顺序：第0列 txId，第1列 time_step，其后 f1..f165
FEATURES_USECOLS_NAMES: list[str] = ["txId", "time_step"] + FEATURE_COLS

DDL_TRANSACTIONS = f"""
CREATE TABLE transactions (
    tx_id      INTEGER PRIMARY KEY,          -- 交易 ID
    time_step  INTEGER NOT NULL,             -- 时间步 1..49
    {", ".join(f"{c} REAL" for c in FEATURE_COLS)},   -- f1..f165 局部/聚合特征
    label      INTEGER,                      -- 1=illicit, 0=licit, NULL=unknown
    split      TEXT NOT NULL CHECK (split IN ('train','val','test')),
    CHECK (label IS NULL OR label IN (0, 1)),
    CHECK (time_step BETWEEN 1 AND 49)
)
"""

DDL_EDGES = """
CREATE TABLE edges (
    src INTEGER NOT NULL,
    dst INTEGER NOT NULL,
    PRIMARY KEY (src, dst)
) WITHOUT ROWID
"""

DDL_VIEW = """
CREATE VIEW transactions_v AS
SELECT tx_id, time_step, label, split,
       CASE WHEN label IS NULL THEN 'unknown'
            WHEN label = 1      THEN 'illicit'
            ELSE 'licit' END AS label_name
FROM transactions
"""

DDL_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_tx_time_step ON transactions(time_step)",
    "CREATE INDEX IF NOT EXISTS idx_tx_label     ON transactions(label)",
    "CREATE INDEX IF NOT EXISTS idx_tx_split     ON transactions(split)",
    "CREATE INDEX IF NOT EXISTS idx_tx_split_lbl ON transactions(split, label)",
    "CREATE INDEX IF NOT EXISTS idx_edges_src    ON edges(src)",
    "CREATE INDEX IF NOT EXISTS idx_edges_dst    ON edges(dst)",
]


# ============================================================
# 读取与导入
# ============================================================
def read_classes() -> pd.DataFrame:
    """读取标签表，校验取值域，返回 txId -> class 字符串 的 DataFrame。"""
    if not CLASSES_CSV.exists():
        raise FileNotFoundError(f"缺少标签文件：{CLASSES_CSV}")
    df = pd.read_csv(CLASSES_CSV, dtype={"txId": "int64", "class": "string"})
    df.columns = ["tx_id", "class"]
    if df["tx_id"].duplicated().any():
        n = int(df["tx_id"].duplicated().sum())
        raise ValueError(f"classes.csv 中 txId 重复 {n} 个，数据异常，请检查源文件")

    valid = set(CLASS_TO_LABEL)
    bad = set(df["class"].dropna().unique()) - valid
    if bad:
        raise ValueError(f"classes.csv 出现未预期的标签取值：{sorted(bad)}，应为 {sorted(valid)}")

    # 硬约束 1：unknown -> None（最终写 SQL NULL）
    df["label"] = df["class"].map(CLASS_TO_LABEL)
    # pandas 会把 None 变成 NaN，显式转回 object 并置 None
    df["label"] = df["label"].astype("object").where(df["label"].notna(), None)
    return df


def load_features_table(conn: sqlite3.Connection, batch_size: int = 50_000,
                        verbose: bool = True) -> int:
    """
    分块流式导入 features.csv -> transactions。

    为什么分块：658MB 的 CSV 一次性读进内存约需 2.7GB（167 列 float64），
    笔记本上容易爆内存；chunksize 让峰值内存只与 batch_size 有关。

    返回导入行数。
    """
    if not FEATURES_CSV.exists():
        raise FileNotFoundError(f"缺少特征文件：{FEATURES_CSV}")

    # 先准备标签映射（内存占用小：20 万条）
    cls = read_classes()
    label_map: dict[int, int | None] = dict(zip(cls["tx_id"].tolist(), cls["label"].tolist()))

    insert_sql = (
        f"INSERT INTO transactions (tx_id, time_step, "
        f"{', '.join(FEATURE_COLS)}, label, split) "
        f"VALUES ({', '.join(['?'] * (2 + N_FEATURES + 2))})"
    )

    total = 0
    n_unknown = 0
    t0 = time.time()
    # 关键：header=None —— 该文件无表头，用 header=0 会吃掉整行数据
    reader = pd.read_csv(
        FEATURES_CSV,
        header=None,
        names=FEATURES_USECOLS_NAMES,
        chunksize=batch_size,
        dtype={"txId": "int64", "time_step": "int16"},
    )

    for chunk_idx, chunk in enumerate(reader, start=1):
        # --- 空数据防护 ---
        if chunk.empty:
            continue
        # --- 缺失值防护：特征列理论上不该有空值，若有则填 0 并计数 ---
        feat_block = chunk[FEATURE_COLS]
        n_na = int(feat_block.isna().sum().sum())
        if n_na:
            if verbose:
                print(f"  [警告] 第 {chunk_idx} 块特征含 {human_int(n_na)} 个空值，已填 0")
            feat_block = feat_block.fillna(0.0)

        tx_ids = chunk["txId"].to_numpy(dtype=np.int64)
        steps = chunk["time_step"].to_numpy(dtype=np.int64)

        # --- 标签：查字典；未命中或 unknown 一律 NULL ---
        labels = [label_map.get(int(t)) for t in tx_ids]
        n_unknown += sum(1 for v in labels if v is None)

        # --- split：由 time_step 决定，不使用任何随机函数 ---
        splits = [time_step_to_split(int(s)) for s in steps]

        # --- 组装参数：必须是纯 Python 标量，float(nan) 会被写成 NULL 需先清洗 ---
        feats = feat_block.to_numpy(dtype=np.float64, copy=False)
        feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
        rows = [
            (int(tx_ids[i]), int(steps[i]), *feats[i].tolist(), labels[i], splits[i])
            for i in range(len(tx_ids))
        ]

        conn.executemany(insert_sql, rows)
        total += len(rows)

        if verbose and (chunk_idx % 1 == 0):
            elapsed = time.time() - t0
            rate = total / elapsed if elapsed > 0 else 0
            print(f"  导入中… 第 {chunk_idx:>3} 块  累计 {human_int(total)} 行  "
                  f"{elapsed:6.1f}s  速率 {human_int(int(rate))}/s")

    if verbose:
        print(f"  transactions 导入完成：{human_int(total)} 行，其中 unknown(NULL) "
              f"{human_int(n_unknown)} 行，耗时 {time.time() - t0:.1f}s")
    return total


def load_edges(conn: sqlite3.Connection, verbose: bool = True) -> int:
    """导入 edgelist.csv -> edges（executemany 批量，非逐行）。"""
    if not EDGELIST_CSV.exists():
        raise FileNotFoundError(f"缺少边表文件：{EDGELIST_CSV}")

    df = pd.read_csv(EDGELIST_CSV, dtype={"txId1": "int64", "txId2": "int64"})
    df.columns = ["src", "dst"]

    n_self = int((df["src"] == df["dst"]).sum())
    n_dup = int(df.duplicated().sum())
    if verbose:
        print(f"  edges 原始 {human_int(len(df))} 条；自环 {n_self} 条；重复 {n_dup} 条")
    if n_self:
        df = df[df["src"] != df["dst"]]
    if n_dup:
        df = df.drop_duplicates()

    rows = list(zip(df["src"].tolist(), df["dst"].tolist()))
    conn.executemany("INSERT OR IGNORE INTO edges (src, dst) VALUES (?, ?)", rows)
    if verbose:
        print(f"  edges 导入完成：{human_int(len(rows))} 条")
    return len(rows)


# ============================================================
# 打印与校验
# ============================================================
def print_split_stats(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    打印各 split 的 illicit / licit / unknown 样本数（模块1要求第6条）。
    unknown 用 label IS NULL 判定 —— 这就是“NULL 而非 0”的价值所在。
    """
    sql = """
    SELECT split,
           SUM(CASE WHEN label = 1 THEN 1 ELSE 0 END) AS illicit,
           SUM(CASE WHEN label = 0 THEN 1 ELSE 0 END) AS licit,
           SUM(CASE WHEN label IS NULL THEN 1 ELSE 0 END) AS unknown,
           COUNT(*) AS total
    FROM transactions
    GROUP BY split
    """
    df = pd.read_sql_query(sql, conn)
    order = {"train": 0, "val": 1, "test": 2}
    df = df.sort_values("split", key=lambda s: s.map(order)).reset_index(drop=True)

    print("\n" + "=" * 74)
    print("各 split 样本数（unknown 为 label IS NULL，不参与训练与指标计算）")
    print("=" * 74)
    print(f"{'split':<8}{'时间步':<14}{'illicit':>10}{'licit':>10}{'unknown':>10}{'合计':>10}")
    print("-" * 74)
    for _, r in df.iterrows():
        lo, hi = SPLITS[r["split"]]
        print(f"{r['split']:<8}{f'{lo}-{hi}':<14}{int(r['illicit']):>10,}"
              f"{int(r['licit']):>10,}{int(r['unknown']):>10,}{int(r['total']):>10,}")
    print("-" * 74)
    tot = df[["illicit", "licit", "unknown", "total"]].sum()
    print(f"{'合计':<8}{'1-49':<14}{int(tot['illicit']):>10,}{int(tot['licit']):>10,}"
          f"{int(tot['unknown']):>10,}{int(tot['total']):>10,}")
    print("=" * 74)
    return df


def verify(conn: sqlite3.Connection) -> bool:
    """
    建库后自动校验（对应手动检查清单 A、B 的机器可判部分）。
    返回 True 表示全部通过。
    """
    print("\n" + "=" * 74)
    print("自动校验")
    print("=" * 74)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    # --- A. 数据层 ---
    n_tx = count_rows(conn, "transactions")
    n_edges = count_rows(conn, "edges")
    check(f"transactions 行数 = {N_TXS:,}", n_tx == N_TXS, f"实际 {n_tx:,}")
    check(f"edges 行数 = {N_EDGES:,}", n_edges == N_EDGES, f"实际 {n_edges:,}")

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(transactions)")]
    check("列名含 tx_id/time_step/f1/f165/label/split",
          all(c in cols for c in ("tx_id", "time_step", "f1", "f165", "label", "split")))
    check(f"特征列恰为 {N_FEATURES} 个 f*",
          sum(1 for c in cols if c.startswith("f")) == N_FEATURES)

    idx = {r["name"] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'")}
    for need in ("idx_tx_time_step", "idx_tx_label", "idx_tx_split", "idx_edges_src", "idx_edges_dst"):
        check(f"索引已建：{need}", need in idx)

    # --- 标签合法性：unknown 必须是 NULL，绝不能是 0 ---
    n_null = int(conn.execute("SELECT COUNT(*) FROM transactions WHERE label IS NULL").fetchone()[0])
    n_zero = int(conn.execute("SELECT COUNT(*) FROM transactions WHERE label = 0").fetchone()[0])
    n_one = int(conn.execute("SELECT COUNT(*) FROM transactions WHERE label = 1").fetchone()[0])
    check("unknown 用 NULL 存储（157,205）", n_null == 157_205, f"实际 {n_null:,}")
    check("licit=0 数量（42,019）", n_zero == 42_019, f"实际 {n_zero:,}")
    check("illicit=1 数量（4,545）", n_one == 4_545, f"实际 {n_one:,}")
    check("label 取值只有 {NULL,0,1}",
          int(conn.execute(
              "SELECT COUNT(*) FROM transactions WHERE label NOT IN (0,1) AND label IS NOT NULL"
          ).fetchone()[0]) == 0)

    # --- B. 时间划分 ---
    for sp, (lo, hi) in SPLITS.items():
        bad = int(conn.execute(
            "SELECT COUNT(*) FROM transactions WHERE split=? AND (time_step<? OR time_step>?)",
            (sp, lo, hi)).fetchone()[0])
        check(f"{sp} 的 time_step 全部落在 {lo}-{hi}", bad == 0, f"越界 {bad} 行")

    overlap = int(conn.execute("""
        SELECT COUNT(*) FROM transactions
        WHERE time_step <= ? AND time_step > ?
    """, (29, 34)).fetchone()[0])
    check("train 全部 <=29 且 val 全部 30-34（区间无交叠）", overlap == 0, f"异常 {overlap} 行")

    n_test_illicit = int(conn.execute(
        "SELECT COUNT(*) FROM transactions WHERE split='test' AND label=1").fetchone()[0])
    check("test 集 illicit > 0", n_test_illicit > 0, f"test illicit = {n_test_illicit:,}")

    # --- 检查有没有误用随机划分：split 与 time_step 必须严格单调对应 ---
    bad_split = int(conn.execute(f"""
        SELECT COUNT(*) FROM transactions
        WHERE (time_step <= {29} AND split <> 'train')
           OR (time_step >  {29} AND time_step <= {34} AND split <> 'val')
           OR (time_step >  {34} AND split <> 'test')
    """).fetchone()[0])
    check("split 完全由 time_step 决定（无随机划分痕迹）", bad_split == 0, f"不一致 {bad_split} 行")

    # --- 边表引用完整性 ---
    orphan = int(conn.execute(
        "SELECT COUNT(*) FROM edges e LEFT JOIN transactions t ON t.tx_id = e.src "
        "WHERE t.tx_id IS NULL").fetchone()[0])
    orphan += int(conn.execute(
        "SELECT COUNT(*) FROM edges e LEFT JOIN transactions t ON t.tx_id = e.dst "
        "WHERE t.tx_id IS NULL").fetchone()[0])
    check("edges 两端都能在 transactions 中找到", orphan == 0, f"悬空 {orphan} 条")

    print("=" * 74)
    print("校验结果：" + ("全部通过 ✓" if ok else "存在 FAIL 项，请检查上方输出 ✗"))
    return ok


# ============================================================
# 主流程
# ============================================================
def build_database(db_path: Path, batch_size: int, force: bool, verbose: bool = True) -> bool:
    ensure_dirs()
    validate_split_edges()

    if db_path.exists() and not force:
        print(f"[跳过] 数据库已存在：{db_path}")
        print("       若需重建，请加 --force")
        conn = get_connection(db_path)
        try:
            print_split_stats(conn)
            return verify(conn)
        finally:
            conn.close()

    if db_path.exists() and force:
        print(f"[覆盖] 删除旧库：{db_path.name}")
        for suffix in ("", "-wal", "-shm", "-journal"):
            p = Path(str(db_path) + suffix)
            if p.exists():
                p.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"目标数据库：{db_path}")
    print(f"数据目录　：{FEATURES_CSV.parent}")
    print(f"分块大小　：{batch_size:,} 行/块\n")

    t_start = time.time()
    conn = get_connection(db_path)
    try:
        print("[1/5] 建表…")
        conn.execute("BEGIN")
        conn.execute("DROP VIEW  IF EXISTS transactions_v")
        conn.execute("DROP TABLE IF EXISTS transactions")
        conn.execute("DROP TABLE IF EXISTS edges")
        conn.execute(DDL_TRANSACTIONS)
        conn.execute(DDL_EDGES)
        conn.commit()

        print("[2/5] 导入 transactions（关键步骤，勿中断）…")
        conn.execute("BEGIN")
        load_features_table(conn, batch_size=batch_size, verbose=verbose)
        conn.commit()

        print("[3/5] 导入 edges…")
        conn.execute("BEGIN")
        load_edges(conn, verbose=verbose)
        conn.commit()

        print("[4/5] 建索引与视图（放在导入之后，否则插入会极慢）…")
        for ddl in DDL_INDEXES:
            conn.execute(ddl)
        conn.execute(DDL_VIEW)
        conn.commit()

        print("[5/5] 统计与校验…")
        conn.execute("ANALYZE")     # 让查询计划器有统计信息
        conn.commit()
        print_split_stats(conn)
        ok = verify(conn)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"\n总耗时 {time.time() - t_start:.1f}s，数据库：{db_path}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="模块1：Elliptic 数据加载 + SQLite 建库",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--db", type=Path, default=DB_PATH, help="SQLite 输出路径")
    ap.add_argument("--batch-size", type=int, default=50_000, help="分块导入行数")
    ap.add_argument("--force", action="store_true", help="已存在时覆盖重建")
    ap.add_argument("--verify-only", action="store_true", help="只校验已有数据库，不导入")
    ap.add_argument("--quiet", action="store_true", help="减少进度输出")
    args = ap.parse_args()

    if args.verify_only:
        if not args.db.exists():
            print(f"数据库不存在：{args.db}")
            return 2
        conn = get_connection(args.db)
        try:
            print_split_stats(conn)
            return 0 if verify(conn) else 1
        finally:
            conn.close()

    ok = build_database(args.db, args.batch_size, args.force, verbose=not args.quiet)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
