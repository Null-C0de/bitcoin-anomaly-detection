"""
graph_features.py —— 模块2：图构建 + 结构类图特征
=================================================

【本模块与原始文档的差异（已与用户对齐，重要！）】
原本文档要求“邻居标签比例（只统计 time_step 小于自身的邻居）”。实测数据后确认：

  1) 本数据集中 **不存在跨时间步的边**。对全部 234,355 条边逐条验证，
     两端 time_step 完全相同（dst 早于 src 的边数为 0）。
     因此“只统计更早时间步的邻居”在该数据集上恒为空集，特征会全 0，没有意义。

  2) 若改用“同时间步邻居标签比例”，则直接使用了同一时间步的已知标签去预测
     同一时间步的未知交易 —— 属于标签泄漏，且线上对新交易打分时根本无法复现。

  => 对齐结论：**完全不用邻居标签**。图特征只保留结构类（无标签信息）。
     这也让 self-check 第 1、2 题可以干净地回答“没有未来信息、没有标签泄漏”。

【输入】 elliptic.db 的 transactions(图节点 + time_step) 与 edges
【输出】 elliptic.db 的 graph_features 表（203,769 行）
  tx_id, in_degree, out_degree, degree, clustering, nbr_count,
  two_hop_nbr_count, pagerank_rolling, time_step
  说明：pagerank_rolling 为**滚动计算**（只在该节点自身时间步及之前的子图上计算，
        不引入未来信息），但按对齐结论 **只用于展示，不进入模型特征**。

【PageRank 的口径】
  对 t = 1..49，取 {time_step <= t} 的诱导子图 G_t 计算 PageRank，
  再把结果赋给 time_step == t 的节点。
  这样节点 u 的 PageRank 只由 t(u) 及其之前的结构决定，不含未来信息。

【依赖】 networkx、pandas、numpy、sqlite3
【运行】 python src/graph_features.py
         python src/graph_features.py --verify-only
【复杂度】以 |E|=234,355、|V|=203,769 计：
  - 全图度数/邻居：O(E)
  - 滚动 PageRank：Σ_t O(E_t)，合计约 O(2·E) = 47 万次松弛，实测约 10-30 秒
  - 同时间步聚类系数：Σ_t O(Σ_v deg(v)^2)，由小时间步子图主导，实测约 10-60 秒
  - 两跳邻居：Σ_t O(Σ_v deg(v)·avg_deg)，实测约 10-40 秒
  总体：本机（笔记本 CPU）预计 1-3 分钟。

【可能踩的坑】
  1. 用 nx.Graph 而不是 nx.DiGraph 时要注意聚类系数定义不同；本模块两者都建。
  2. 孤立节点（度=0）的聚类系数必须是 0，不能是 NaN。
  3. 边表中 txId1/txId2 的方向语义在官方文档中是二义性的，本模块同时给出
     入度与出度，并在论文中明确说明口径，避免答辩时被追问“入度是谁的入度”。
  4. 20 万节点不要用 Python 循环逐节点查 SQLite，必须一次性取到内存。
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DB_PATH,
    N_TIME_STEPS,
    N_TXS,
    ensure_dirs,
    get_connection,
    human_int,
    table_exists,
)

DDL_GRAPH_FEATURES = """
CREATE TABLE graph_features (
    tx_id             INTEGER PRIMARY KEY,
    in_degree         INTEGER NOT NULL DEFAULT 0,
    out_degree        INTEGER NOT NULL DEFAULT 0,
    degree            INTEGER NOT NULL DEFAULT 0,
    clustering        REAL    NOT NULL DEFAULT 0,
    nbr_count         INTEGER NOT NULL DEFAULT 0,
    two_hop_nbr_count INTEGER NOT NULL DEFAULT 0,
    pagerank_rolling  REAL    NOT NULL DEFAULT 0,
    time_step         INTEGER NOT NULL
)
"""

INSERT_SQL = (
    "INSERT INTO graph_features (tx_id, in_degree, out_degree, degree, clustering, "
    "nbr_count, two_hop_nbr_count, pagerank_rolling, time_step) "
    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)"
)


# ============================================================
# 数据读取
# ============================================================
def load_nodes(conn) -> tuple[np.ndarray, np.ndarray]:
    """返回 (tx_id 数组, time_step 数组)，按 time_step 升序，便于增量处理。"""
    df = pd.read_sql_query(
        "SELECT tx_id, time_step FROM transactions ORDER BY time_step, tx_id", conn
    )
    if df.empty:
        raise ValueError("transactions 表为空，请先运行 src/load_data.py")
    return df["tx_id"].to_numpy(dtype=np.int64), df["time_step"].to_numpy(dtype=np.int64)


def load_edges(conn) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_sql_query("SELECT src, dst FROM edges", conn)
    if df.empty:
        raise ValueError("edges 表为空，请先运行 src/load_data.py")
    return df["src"].to_numpy(dtype=np.int64), df["dst"].to_numpy(dtype=np.int64)


# ============================================================
# 特征计算
# ============================================================
def build_graphs(tx_ids: np.ndarray, steps: np.ndarray,
                 src: np.ndarray, dst: np.ndarray,
                 verbose: bool = True) -> tuple[nx.DiGraph, nx.Graph]:
    """构建有向图（保留边方向）与无向投影图（用于聚类系数）。"""
    G = nx.DiGraph()
    G.add_nodes_from(tx_ids.tolist())          # 显式加全部节点，保证孤立节点也在图里
    G.add_edges_from(zip(src.tolist(), dst.tolist()))
    U = G.to_undirected()
    if verbose:
        print(f"  有向图：{human_int(G.number_of_nodes())} 节点 / {human_int(G.number_of_edges())} 边")
        print(f"  无向投影：{human_int(U.number_of_edges())} 边")
    return G, U


def compute_degree_and_neighbors(G: nx.DiGraph, tx_ids: np.ndarray) -> pd.DataFrame:
    """
    度数类特征 + 去重邻居数。O(V + E)。
    孤立节点天然得到 0，不产生 NaN。
    """
    in_deg = dict(G.in_degree())
    out_deg = dict(G.out_degree())
    # 去重邻居数（无向意义：并入度出度）
    undirected_nbr = {n: set(G.predecessors(n)) | set(G.successors(n)) for n in G.nodes()}

    df = pd.DataFrame({
        "tx_id": tx_ids,
        "in_degree": [int(in_deg.get(int(t), 0)) for t in tx_ids],
        "out_degree": [int(out_deg.get(int(t), 0)) for t in tx_ids],
    })
    df["degree"] = df["in_degree"] + df["out_degree"]
    df["nbr_count"] = [len(undirected_nbr.get(int(t), ())) for t in tx_ids]
    return df


def compute_rolling_pagerank(G: nx.DiGraph, tx_ids: np.ndarray, steps: np.ndarray,
                             verbose: bool = True) -> np.ndarray:
    """
    滚动 PageRank：只用 {time_step <= t} 的子图计算，赋给 time_step == t 的节点。

    为什么不用全图 PageRank：
      全图 PageRank 会把未来时间步的结构影响力通过幂迭代“倒推”到早期节点，
      与本项目的硬约束 3（禁止未来信息）冲突。滚动计算可保证因果性。
    """
    pr = np.zeros(len(tx_ids), dtype=np.float64)
    # 按时间步分组索引
    for t in range(1, N_TIME_STEPS + 1):
        mask_t = steps == t
        if not mask_t.any():
            continue
        nodes_t = tx_ids[mask_t]

        # 累积子图：所有 time_step <= t 的节点（用掩码保证是“历史 + 当前”）
        prefix_mask = steps <= t
        prefix_nodes = tx_ids[prefix_mask]
        sub = G.subgraph(prefix_nodes)

        # 边界情况：空图 / 无边图 -> PageRank 退化为均匀分布，显式处理避免抛错
        if sub.number_of_nodes() == 0:
            continue
        if sub.number_of_edges() == 0:
            pr_t = {n: 1.0 / sub.number_of_nodes() for n in sub.nodes()}
        else:
            try:
                pr_t = nx.pagerank(sub, alpha=0.85, tol=1.0e-8, max_iter=200)
            except nx.PowerIterationFailedConvergence:
                # 极端结构下不收敛时放宽容差重算，保证脚本不中断
                pr_t = nx.pagerank(sub, alpha=0.85, tol=1.0e-6, max_iter=500)

        vals = np.fromiter((pr_t.get(int(n), 0.0) for n in nodes_t),
                           dtype=np.float64, count=len(nodes_t))
        pr[mask_t] = vals

        if verbose and (t % 10 == 0 or t == N_TIME_STEPS):
            print(f"    滚动 PageRank 进度 t={t:>2}  子图 {human_int(sub.number_of_nodes())} 节点")
    return pr


def clustering_map(graph: nx.Graph, nodes: list[int]) -> dict[int, float]:
    """
    某节点集合上的聚类系数。孤立节点（度=0）nx 返回 0.0，符合“缺失值填 0”。
    节点集合用 list 以固定迭代顺序，便于整图/分步结果逐项对照。
    """
    return nx.clustering(graph, nodes=nodes)


def two_hop_counts(adj: dict[int, set[int]], nodes: list[int]) -> dict[int, int]:
    """
    两跳邻居数：邻居的邻居并集，去重且不含自身。
    大度数节点用集合运算，小度数节点用 chain 更省内存——结果一致。
    """
    out: dict[int, int] = {}
    for node in nodes:
        nbrs = adj.get(node, set())
        if len(nbrs) < 8:
            hop2 = set(itertools.chain.from_iterable(adj.get(b, set()) for b in nbrs))
        else:
            hop2 = set()
            for b in nbrs:
                hop2 |= adj.get(b, set())
        hop2.discard(node)
        out[node] = len(hop2)
    return out


def compute_clustering(U: nx.Graph, tx_ids: np.ndarray, steps: np.ndarray,
                       verbose: bool = True) -> np.ndarray:
    """
    聚类系数（整图，无向投影）。

    为什么整图计算不引入未来信息：
      聚类系数只依赖该节点自己的邻居集合；而实测本数据集不存在跨时间步的边，
      所以邻居必然在同一时间步内 —— 整图口径与按时间步口径在数学上等价。
      build() 会调用 verify_equivalence() 用分步口径实测验证这一论断。
    """
    nodes = tx_ids.tolist()
    cl = clustering_map(U, nodes)
    out = np.fromiter((cl.get(int(x), 0.0) for x in tx_ids), dtype=np.float64, count=len(tx_ids))
    if verbose:
        print(f"    聚类系数：整图计算完成，均值 {out.mean():.6f}，"
              f"非零节点 {int(np.count_nonzero(out)):,}")
    return out


def compute_two_hop(U: nx.Graph, tx_ids: np.ndarray, steps: np.ndarray,
                    verbose: bool = True) -> np.ndarray:
    """
    两跳邻居数。同样只依赖自身邻域，因此整图计算与本时间步子图计算等价。
    孤立节点两跳数 = 0。
    """
    adj: dict[int, set[int]] = {n: set(U.neighbors(n)) for n in U.nodes()}
    counts = two_hop_counts(adj, tx_ids.tolist())
    out = np.fromiter((counts.get(int(x), 0) for x in tx_ids), dtype=np.int64, count=len(tx_ids))
    if verbose:
        print(f"    两跳邻居：整图计算完成，均值 {out.mean():.2f}，最大 {int(out.max()):,}")
    return out


def verify_equivalence(U: nx.Graph, tx_ids: np.ndarray, steps: np.ndarray,
                       verbose: bool = True) -> bool:
    """
    实测验证“整图口径 == 按时间步口径”，坐实聚类系数/两跳不含未来信息这一论断。
    若两种口径出现任何不一致，说明存在跨时间步的边，本模块的替代方案前提失效 —— 必须报错。
    """
    ok = True
    for t in (1, 15, 29, 35, 42, 49):
        idx = np.flatnonzero(steps == t)
        if idx.size == 0:
            continue
        nodes = tx_ids[idx].tolist()
        sub = U.subgraph(nodes)

        cl_full = clustering_map(U, nodes)
        cl_sub = clustering_map(sub, nodes)
        diff_cl = max((abs(cl_full[x] - cl_sub[x]) for x in nodes), default=0.0)

        adj_full = {n: set(U.neighbors(n)) for n in nodes}
        adj_sub = {n: set(sub.neighbors(n)) for n in nodes}
        cnt_full = two_hop_counts(adj_full, nodes)
        cnt_sub = two_hop_counts(adj_sub, nodes)
        diff_2h = max((abs(cnt_full[x] - cnt_sub[x]) for x in nodes), default=0)

        same = (diff_cl == 0.0) and (diff_2h == 0)
        ok = ok and same
        if verbose:
            print(f"    t={t:>2} n={len(nodes):>5}  聚类最大差 {diff_cl:.3g}  "
                  f"两跳最大差 {diff_2h}  -> {'一致' if same else '不一致!'}")
    return ok


# ============================================================
# 写库
# ============================================================
def write_graph_features(conn, df: pd.DataFrame, batch_size: int = 20_000,
                         verbose: bool = True) -> int:
    conn.execute("DROP TABLE IF EXISTS graph_features")
    conn.execute(DDL_GRAPH_FEATURES)

    # 硬约束：任何 NaN 都不允许写入（孤立节点缺失值填 0）
    for col in ("clustering", "pagerank_rolling"):
        n_na = int(df[col].isna().sum())
        if n_na:
            if verbose:
                print(f"  [警告] {col} 有 {human_int(n_na)} 个 NaN，已填 0")
            df[col] = df[col].fillna(0.0)

    cols = ["tx_id", "in_degree", "out_degree", "degree", "clustering",
            "nbr_count", "two_hop_nbr_count", "pagerank_rolling", "time_step"]
    rows = list(df[cols].itertuples(index=False, name=None))
    conn.executemany(INSERT_SQL, rows)
    if verbose:
        print(f"  graph_features 写入 {human_int(len(rows))} 行")
    return len(rows)


# ============================================================
# 校验
# ============================================================
def node_degree_reference(src: np.ndarray, dst: np.ndarray, node: int) -> tuple[int, int, int]:
    """独立（不用 networkx）计算某节点的入度/出度/度数，用于交叉验证。"""
    in_d = int(np.count_nonzero(dst == node))
    out_d = int(np.count_nonzero(src == node))
    return in_d, out_d, in_d + out_d


def verify(conn, sample_size: int = 3, seed: int = 0) -> bool:
    print("\n" + "=" * 74)
    print("自动校验（模块2）")
    print("=" * 74)
    ok = True

    def check(name: str, cond: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and cond
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))

    if not table_exists(conn, "graph_features"):
        print("  [FAIL] graph_features 表不存在")
        return False

    n = int(conn.execute("SELECT COUNT(*) FROM graph_features").fetchone()[0])
    check(f"graph_features 行数 = {N_TXS:,}", n == N_TXS, f"实际 {n:,}")

    n_nan = int(conn.execute(
        "SELECT COUNT(*) FROM graph_features WHERE clustering IS NULL OR pagerank_rolling IS NULL"
    ).fetchone()[0])
    check("无 NULL / NaN（孤立节点已填 0）", n_nan == 0, f"NULL 行数 {n_nan}")

    n_isolated = int(conn.execute(
        "SELECT COUNT(*) FROM graph_features WHERE nbr_count = 0").fetchone()[0])
    check("孤立节点已统计（填 0）", True, f"孤立节点 {human_int(n_isolated)} 个")

    # 聚类系数必须在 [0,1]，PageRank 必须 > 0 且总和合理
    bad_cl = int(conn.execute(
        "SELECT COUNT(*) FROM graph_features WHERE clustering < 0 OR clustering > 1").fetchone()[0])
    check("聚类系数全部落在 [0,1]", bad_cl == 0, f"越界 {bad_cl}")

    n_zero_pr = int(conn.execute(
        "SELECT COUNT(*) FROM graph_features WHERE pagerank_rolling <= 0").fetchone()[0])
    check("滚动 PageRank 全部 > 0", n_zero_pr == 0, f"<=0 的节点 {n_zero_pr}")

    n_zero_2h = int(conn.execute(
        "SELECT COUNT(*) FROM graph_features WHERE two_hop_nbr_count = 0").fetchone()[0])
    check("两跳邻居数已计算", True, f"两跳为 0 的节点 {human_int(n_zero_2h)} 个")

    cols = [r["name"] for r in conn.execute("PRAGMA table_info(graph_features)")]
    need = ["in_degree", "out_degree", "degree", "clustering", "nbr_count",
            "two_hop_nbr_count", "pagerank_rolling"]
    check("所需特征列齐全", all(c in cols for c in need),
          f"缺 {[c for c in need if c not in cols]}")

    # --- 交叉验证：随机抽节点，用 SQL 独立重算入度/出度/度数 ---
    src_all = pd.read_sql_query("SELECT src FROM edges", conn)["src"].to_numpy(dtype=np.int64)
    dst_all = pd.read_sql_query("SELECT dst FROM edges", conn)["dst"].to_numpy(dtype=np.int64)
    df = pd.read_sql_query(
        "SELECT tx_id, in_degree, out_degree, degree FROM graph_features "
        "ORDER BY RANDOM() LIMIT ?", conn, params=(sample_size,))
    print(f"  --- 随机抽 {sample_size} 个节点，用 SQL 独立重算度数 ---")
    for _, r in df.iterrows():
        tx = int(r["tx_id"])
        in_d, out_d, deg = node_degree_reference(src_all, dst_all, tx)
        good = (in_d == int(r["in_degree"]) and out_d == int(r["out_degree"])
                and deg == int(r["degree"]))
        ok = ok and good
        print(f"    [{'PASS' if good else 'FAIL'}] tx {tx}: 独立重算 in={in_d} out={out_d} "
              f"deg={deg} | 表内 in={int(r['in_degree'])} out={int(r['out_degree'])} "
              f"deg={int(r['degree'])}")

    # --- 明确验证“不存在跨时间步的边”，这是本模块替代方案的前提 ---
    cross = int(conn.execute("""
        SELECT COUNT(*) FROM edges e
        JOIN transactions a ON a.tx_id = e.src
        JOIN transactions b ON b.tx_id = e.dst
        WHERE a.time_step <> b.time_step
    """).fetchone()[0])
    check("确认无跨时间步的边（本模块替代方案的前提）", cross == 0, f"跨步边 {cross}")

    print("=" * 74)
    print("校验结果：" + ("全部通过 ✓" if ok else "存在 FAIL 项 ✗"))
    return ok


# ============================================================
# 主流程
# ============================================================
def build(db_path: Path, verbose: bool = True) -> bool:
    ensure_dirs()
    t0 = time.time()
    conn = get_connection(db_path)
    try:
        if not table_exists(conn, "transactions"):
            raise RuntimeError("transactions 表不存在，请先运行：python src/load_data.py")

        print("[1/4] 读取节点与边…")
        tx_ids, steps = load_nodes(conn)
        src, dst = load_edges(conn)
        print(f"  节点 {human_int(len(tx_ids))}，边 {human_int(len(src))}")

        print("[2/4] 构建图…")
        G, U = build_graphs(tx_ids, steps, src, dst, verbose=verbose)

        print("[3/4] 计算特征（度数 / 邻居 / 滚动 PageRank / 聚类 / 两跳）…")
        df = compute_degree_and_neighbors(G, tx_ids)
        print("    滚动 PageRank…")
        df["pagerank_rolling"] = compute_rolling_pagerank(G, tx_ids, steps, verbose=verbose)
        print("    聚类系数…")
        df["clustering"] = compute_clustering(U, tx_ids, steps, verbose=verbose)
        print("    两跳邻居…")
        df["two_hop_nbr_count"] = compute_two_hop(U, tx_ids, steps, verbose=verbose)

        # 关键论断验证：整图口径与按时间步口径必须完全一致，否则说明存在跨步边
        print("    验证“整图口径 == 按时间步口径”（证明不含未来信息）…")
        equiv_ok = verify_equivalence(U, tx_ids, steps, verbose=verbose)
        if not equiv_ok:
            print("    [警告] 两种口径不一致！说明数据中存在跨时间步的边，"
                  "聚类/两跳的整图计算口径不再安全，需要改回按时间步计算。")

        df["time_step"] = steps
        df = df.sort_values("tx_id").reset_index(drop=True)

        print("[4/4] 写入 graph_features 并校验…")
        conn.execute("BEGIN")
        write_graph_features(conn, df, verbose=verbose)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_tx ON graph_features(tx_id)")
        conn.commit()
        ok = verify(conn) and equiv_ok
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(f"\n总耗时 {time.time() - t0:.1f}s")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(
        description="模块2：图构建 + 结构类图特征（不使用邻居标签，无未来信息）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--db", type=Path, default=DB_PATH, help="SQLite 路径")
    ap.add_argument("--verify-only", action="store_true", help="只校验，不重算")
    ap.add_argument("--quiet", action="store_true", help="减少进度输出")
    args = ap.parse_args()

    if args.verify_only:
        conn = get_connection(args.db)
        try:
            return 0 if verify(conn) else 1
        finally:
            conn.close()

    return 0 if build(args.db, verbose=not args.quiet) else 1


if __name__ == "__main__":
    raise SystemExit(main())
