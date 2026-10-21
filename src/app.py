"""
app.py —— 模块5：FastAPI 后端
=============================

【启动】
  uvicorn app:app --host 127.0.0.1 --port 8000 --reload
  # 在本项目根目录执行；或 python -m uvicorn src.app:app --port 8000
  # 交互式文档： http://127.0.0.1:8000/docs
【输入 / 输出 / 依赖】
  输入：elliptic.db（transactions / edges / graph_features / experiments）
        以及 outputs/models/{group}__{model}.pkl（模块3产出）
  输出：JSON
  依赖：fastapi、uvicorn、pydantic v2、pandas、numpy、joblib、scikit-learn
        psutil（可选，用于更精确的内存统计）

【接口】
  GET  /health                     服务与模型状态
  POST /score                      输入 tx_id 列表（<=1000 笔），返回风险评分
  GET  /top_risk?k=20&scope=unknown 返回 Top-K 高风险交易
  GET  /subgraph/{tx_id}?hops=2    返回 <=50 节点的局部图，**只含可观测历史**
  GET  /stats                      累计请求数、平均耗时、峰值内存

【设计要点（答辩重点）】
  1) 启动只加载一次模型与 SQLite 连接（FastAPI startup 事件），
     避免每个请求都反序列化 .pkl。
  2) /score 单次请求上限 1000 笔，超出返回 413，防止一次打爆内存。
  3) /subgraph 严格只返回 time_step <= 目标交易 time_step 的节点：
     本项目实测不存在跨时间步的边，所以邻域天然都在同一时间步内，
     但仍然显式加这一步过滤 —— 万一将来换数据集/补边，不至于泄漏未来。
  4) **不返回标签**。测试集标签绝不通过接口暴露，
     只有显式设置环境变量 DSH_EXPOSE_LABELS=1 时才在响应里带上 label 字段，
    且默认关闭，界面上也不展示。
  5) 输出含免责声明字段：评分不等同于违法认定。

【可能踩的坑】
  1. SQLite 连接不是线程安全的：FastAPI 的同步路由跑在线程池里，
     连接按线程独立（thread-local），而不是全进程共享一个连接：
     FastAPI 的同步路由跑在线程池里，同一个连接跨线程使用会直接抛
     "SQLite objects created in a thread can only be used in that same thread"。
     共享连接即使关掉 check_same_thread 也容易踩指针/事务错乱的坑，
     所以这里每个线程自己开一条只读连接，用完随线程回收。
     高并发场景应换成连接池或改异步驱动，毕设规模（单机演示）足够。
  2. 直接对 50 万行做全表打分很慢：/top_risk 用离线缓存表 risk_scores，
     首次请求时批量算一次，之后走索引。
  3. 模型的特征顺序必须与训练一致 —— 从 .pkl 的 features 字段取，不要手写列名。
  4. Pydantic v2 与 v1 的写法不兼容（这里用 v2：list[int]、model_dump）。
"""
from __future__ import annotations

import os
import secrets
import sqlite3
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import (  # noqa: E402
    DB_PATH,
    MAX_SCORE_BATCH,
    MAX_SUBGRAPH_NODES,
    POS_LABEL,
    SUBGRAPH_HOPS,
    get_connection,
    human_int,
    select_feature_sql,
    table_exists,
)

DISCLAIMER = "风险评分仅为模型统计推断，不等同于违法认定，需人工复核。"
EXPOSE_LABELS = os.environ.get("DSH_EXPOSE_LABELS", "0") == "1"

# ============================================================
# 接口鉴权（可选，默认关闭）
# ============================================================
# 设置环境变量 DSH_API_KEY 后，除根路径外的所有接口都必须带
#   X-API-Key: <你的密钥>
# 请求头，否则返回 401。未设置时保持本地开发体验不变（不校验）。
#
# 设计取舍：
#  1) 用 secrets.compare_digest 做**定长比较**，避免通过响应时间差
#     逐字节猜出密钥（timing attack）。普通的 == 会短路退出，
#     攻击者可以用大量请求的耗时分布反推密钥前缀。
#  2) 密钥只从环境变量读，**绝不写进代码或仓库**，也不会出现在日志里。
#  3) 校验失败只回 "无效的 API Key"，不区分"没带"和"带错"，
#     避免给攻击者提供密钥是否存在的信号。
API_KEY = os.environ.get("DSH_API_KEY", "").strip()


def require_api_key(x_api_key: Optional[str] = Header(default=None,
                                                     alias="X-API-Key")) -> None:
    """依赖项：配置了 DSH_API_KEY 时校验请求头。"""
    if not API_KEY:                      # 未配置 = 本地开发模式，放行
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="无效的 API Key",
            headers={"WWW-Authenticate": "X-API-Key"},
        )


# 所有业务接口挂上鉴权依赖（未配置密钥时自动放行）
AUTH = [Depends(require_api_key)]


# 进程级资源统计（psutil 可选）
try:
    import psutil
    _PROC = psutil.Process(os.getpid())
except Exception:                                    # pragma: no cover
    psutil = None
    _PROC = None

# 全局串行锁：SQLite 连接被多个工作线程共用，必须串行化访问。
# 毕设规模（单机演示、请求量小）下，用锁换取实现的简单与正确。
DB_LOCK = threading.Lock()

STATE: dict[str, Any] = {
    "conn": None,
    "bundle": None,
    "features": [],
    "group": None,
    "model_name": None,
    "threshold": 0.5,
    "metrics": {},
    "n_requests": 0,
    "total_seconds": 0.0,
    "peak_memory_mb": 0.0,
}


# ============================================================
# 请求 / 响应模型
# ============================================================
class ScoreRequest(BaseModel):
    tx_ids: list[int] = Field(..., min_length=1, description="交易 ID 列表，最多 1000 笔")
    with_explanation: bool = Field(False, description="是否附带图特征明细")


class ScoreItem(BaseModel):
    tx_id: int
    risk_score: Optional[float] = None
    risk_level: Optional[str] = None
    time_step: Optional[int] = None
    features: Optional[dict[str, float]] = None
    label: Optional[int] = None          # 默认不返回；仅 DSH_EXPOSE_LABELS=1 时给出
    error: Optional[str] = None


class ScoreResponse(BaseModel):
    model: str
    threshold: float
    requested: int
    scored: int
    elapsed_ms: float
    peak_memory_mb: float
    disclaimer: str
    results: list[ScoreItem]


class TopRiskItem(BaseModel):
    rank: int
    tx_id: int
    risk_score: float
    risk_level: str
    time_step: int


class TopRiskResponse(BaseModel):
    k: int
    scope: str
    total_candidates: int
    computed_at: str
    elapsed_ms: float
    disclaimer: str
    results: list[TopRiskItem]


class SubgraphNode(BaseModel):
    tx_id: int
    time_step: int
    risk_score: Optional[float] = None
    is_target: bool
    label: Optional[int] = None


class SubgraphEdge(BaseModel):
    src: int
    dst: int


class SubgraphResponse(BaseModel):
    target: int
    target_time_step: int
    hops: int
    observable_only: bool
    truncated: bool
    node_count: int
    edge_count: int
    elapsed_ms: float
    disclaimer: str
    nodes: list[SubgraphNode]
    edges: list[SubgraphEdge]


# ============================================================
# 工具
# ============================================================
def risk_level(score: float, threshold: float) -> str:
    """把概率映射到业务可读的风险级别。"""
    if score >= max(0.8, threshold):
        return "high"
    if score >= threshold:
        return "medium"
    if score >= threshold * 0.5:
        return "low"
    return "minimal"


def current_memory_mb() -> float:
    if _PROC is not None:
        try:
            return round(_PROC.memory_info().rss / 1024 / 1024, 2)
        except Exception:
            pass
    return 0.0


def track(elapsed_s: float) -> None:
    STATE["n_requests"] += 1
    STATE["total_seconds"] += elapsed_s
    STATE["peak_memory_mb"] = max(STATE["peak_memory_mb"], current_memory_mb())


def load_best_bundle(conn) -> tuple[dict, dict]:
    """
    从 experiments 表挑 test PR-AUC 最高的实验，加载对应 .pkl。
    这么做的好处：模块3一改实验设置，后端自动跟着走，不会用错模型。
    """
    if not table_exists(conn, "experiments"):
        raise RuntimeError("experiments 表不存在，请先运行 src/train_compare.py")

    row = conn.execute("""
        SELECT exp_group, model, model_path, pr_auc, f1, recall, precision, threshold
        FROM experiments
        WHERE split='test' AND threshold_mode='default_0.5'
        ORDER BY pr_auc DESC LIMIT 1
    """).fetchone()
    if row is None:
        raise RuntimeError("experiments 表为空，请先运行 src/train_compare.py")

    info = dict(row)
    thr_row = conn.execute("""
        SELECT threshold FROM experiments
        WHERE exp_group=? AND model=? AND split='val' AND threshold_mode='tuned_on_val'
    """, (info["exp_group"], info["model"])).fetchone()
    threshold = float(thr_row["threshold"]) if thr_row else 0.5

    p = Path(info["model_path"])
    if not p.exists():
        raise FileNotFoundError(f"模型文件缺失：{p}，请先运行 src/train_compare.py")
    bundle = joblib.load(p)
    return bundle, {"info": info, "threshold": threshold}


def row_features(conn, tx_id: int, features: list[str]) -> Optional[np.ndarray]:
    """取单个交易的特征向量（顺序严格按模型保存的 features）。"""
    sql = f"""
        SELECT t.time_step, {select_feature_sql(features)}
        FROM transactions t JOIN graph_features g ON g.tx_id = t.tx_id
        WHERE t.tx_id = ?
    """
    row = conn.execute(sql, (tx_id,)).fetchone()
    if row is None:
        return None
    return np.asarray([row[c] for c in features], dtype=np.float64)


def fetch_meta(conn, tx_ids: list[int]) -> dict[int, dict]:
    if not tx_ids:
        return {}
    marks = ",".join("?" * len(tx_ids))
    sql = (f"SELECT tx_id, time_step, label FROM transactions WHERE tx_id IN ({marks})")
    return {int(r["tx_id"]): {"time_step": int(r["time_step"]),
                             "label": None if r["label"] is None else int(r["label"])}
            for r in conn.execute(sql, tx_ids)}


# ============================================================
# 启动 / 关闭
# ============================================================
@asynccontextmanager
async def lifespan(_: FastAPI):
    # 建一条用于启动期加载模型的只读连接，用完即关；
    # 运行期的连接由 get_conn() 按线程按需创建。
    conn = get_connection(DB_PATH, readonly=True, check_same_thread=False)
    conn.execute("PRAGMA query_only=ON")
    try:
        bundle, meta = load_best_bundle(conn)
    finally:
        conn.close()
    STATE["bundle"] = bundle
    STATE["features"] = list(bundle["features"])
    STATE["group"] = bundle["group"]
    STATE["model_name"] = bundle["model"]
    STATE["threshold"] = meta["threshold"]
    STATE["metrics"] = meta["info"]
    print(f"[启动] 模型 {STATE['group']} + {STATE['model_name']}，"
          f"{len(STATE['features'])} 个特征，阈值 {STATE['threshold']:.4f}")
    print(f"[启动] test PR-AUC={STATE['metrics']['pr_auc']:.4f} "
          f"F1={STATE['metrics']['f1']:.4f} Recall={STATE['metrics']['recall']:.4f}")
    close_thread_connections()
    yield


app = FastAPI(
    title="比特币异常交易识别 API",
    description="基于图特征与机器学习的 Elliptic 数据集异常交易检测。"
                "评分不等同于违法认定。",
    version="1.0.0",
    lifespan=lifespan,
)


_TLS = threading.local()
_ALL_CONNS: list[sqlite3.Connection] = []
_ALL_CONNS_LOCK = threading.Lock()


def get_conn() -> sqlite3.Connection:
    """
    取当前线程专属的只读连接（懒创建）。

    为什么不用全局共享连接：FastAPI 同步路由在线程池中执行，
    跨线程复用 sqlite3.Connection 会抛
    "SQLite objects created in a thread can only be used in that same thread"。
    按线程各持一条连接最省心，也不需要给连接加锁。
    """
    conn = getattr(_TLS, "conn", None)
    if conn is None:
        conn = get_connection(DB_PATH, readonly=True, check_same_thread=False)
        conn.execute("PRAGMA query_only=ON")
        _TLS.conn = conn
        with _ALL_CONNS_LOCK:
            _ALL_CONNS.append(conn)
    return conn


def close_thread_connections() -> None:
    """进程退出时统一关闭各线程的连接（尽力而为）。"""
    with _ALL_CONNS_LOCK:
        for c in _ALL_CONNS:
            try:
                c.close()
            except Exception:
                pass
        _ALL_CONNS.clear()


# ============================================================
# 接口
# ============================================================
@app.get("/health", summary="健康检查", dependencies=AUTH)
def health() -> dict:
    return {
        "status": "ok" if STATE["bundle"] is not None else "loading",
        "model": STATE["model_name"],
        "feature_group": STATE["group"],
        "n_features": len(STATE["features"]),
        "threshold": STATE["threshold"],
        # 只回文件名，不回绝对路径：绝对路径会泄漏服务器的
        # 目录结构与用户名，属于无害但没必要的情报披露。
        "db": DB_PATH.name,
        "auth_required": bool(API_KEY),
        "labels_exposed": EXPOSE_LABELS,
        "disclaimer": DISCLAIMER,
    }


@app.post("/score", response_model=ScoreResponse, summary="批量风险评分", dependencies=AUTH)
def score(req: ScoreRequest) -> ScoreResponse:
    """
    输入 tx_id 列表，返回每笔交易的风险评分（P(illicit)）。

    - 单次请求最多 1000 笔（超出返回 413）
    - 不在库中的 tx_id 不报错，对应结果 error 字段说明原因
    - 默认不返回真实标签（防止测试集标签从接口泄漏）
    """
    t0 = time.perf_counter()
    if len(req.tx_ids) > MAX_SCORE_BATCH:
        raise HTTPException(
            status_code=413,
            detail=f"单次请求最多 {MAX_SCORE_BATCH} 笔，当前 {len(req.tx_ids)} 笔")

    model = STATE["bundle"]["pipeline"]
    features: list[str] = STATE["features"]
    threshold: float = STATE["threshold"]
    pos_idx = list(model.classes_).index(POS_LABEL)

    # 去重但保持顺序，避免同一笔被算多次
    uniq_ids: list[int] = list(dict.fromkeys(int(x) for x in req.tx_ids))

    results: list[ScoreItem] = []
    X_buf: list[np.ndarray] = []
    buf_ids: list[int] = []
    # 连接与游标不跨线程共享：整个读写段用锁包住
    with DB_LOCK:
        conn = get_conn()
        meta = fetch_meta(conn, uniq_ids)
        for tx in uniq_ids:
            info = meta.get(tx)
            if info is None:
                results.append(ScoreItem(tx_id=tx, error="交易不在数据库中"))
                continue
            vec = row_features(conn, tx, features)
            if vec is None:
                results.append(ScoreItem(tx_id=tx, error="缺少图特征，请先运行 graph_features.py"))
                continue
            X_buf.append(np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0))
            buf_ids.append(tx)
        explain_vectors = ({tx: row_features(conn, tx, features) for tx in buf_ids}
                           if req.with_explanation else {})

    if X_buf:
        X = np.vstack(X_buf)
        proba = model.predict_proba(X)[:, pos_idx]
        for tx, p in zip(buf_ids, proba):
            info = meta[tx]
            item = ScoreItem(
                tx_id=tx,
                risk_score=round(float(p), 6),
                risk_level=risk_level(float(p), threshold),
                time_step=info["time_step"],
                label=info["label"] if EXPOSE_LABELS else None,
            )
            if req.with_explanation:
                vec = explain_vectors.get(tx)
                if vec is not None:
                    item.features = {f: float(v) for f, v in
                                     zip(features, np.nan_to_num(vec, nan=0.0))}
            results.append(item)

    elapsed = time.perf_counter() - t0
    track(elapsed)
    mem = current_memory_mb()
    return ScoreResponse(
        model=f"{STATE['group']} + {STATE['model_name']}",
        threshold=threshold,
        requested=len(req.tx_ids),
        scored=len(X_buf),
        elapsed_ms=round(elapsed * 1000, 2),
        peak_memory_mb=mem,
        disclaimer=DISCLAIMER,
        results=results,
    )


RISK_TABLE = "risk_scores"


def ensure_risk_scores(conn, force: bool = False) -> str:
    """
    离线给“未标注交易(unknown)”批量打分并缓存到 risk_scores 表。

    为什么缓存：/top_risk 每次全表打分要几十秒；
             缓存后 Top-K 查询走索引，毫秒级返回。
    注意：这里对 unknown 打分只是**推理**，unknown 标签始终不参与训练与指标。
    """
    conn2 = get_connection(DB_PATH, check_same_thread=False)   # 缓存表需要可写连接
    try:
        if force or not table_exists(conn2, RISK_TABLE):
            features: list[str] = STATE["features"]
            bundle = STATE["bundle"]
            model = bundle["pipeline"]
            classes = list(model.classes_)
            pos_idx = classes.index(POS_LABEL)

            sql = f"""
                SELECT t.tx_id, t.time_step, {select_feature_sql(features)}
                FROM transactions t JOIN graph_features g ON g.tx_id = t.tx_id
                WHERE t.label IS NULL
            """
            df = pd.read_sql_query(sql, conn2)
            if df.empty:
                raise HTTPException(status_code=404, detail="没有未标注交易可供排序")

            X = np.nan_to_num(df[features].to_numpy(dtype=np.float64),
                              nan=0.0, posinf=0.0, neginf=0.0)
            proba = model.predict_proba(X)[:, pos_idx]

            conn2.execute(f"DROP TABLE IF EXISTS {RISK_TABLE}")
            conn2.execute(f"""
                CREATE TABLE {RISK_TABLE} (
                    tx_id INTEGER PRIMARY KEY,
                    risk_score REAL NOT NULL,
                    time_step INTEGER NOT NULL,
                    computed_at TEXT NOT NULL
                )
            """)
            now = time.strftime("%Y-%m-%d %H:%M:%S")
            rows = [(int(t), float(p), int(s), now)
                    for t, p, s in zip(df["tx_id"], proba, df["time_step"])]
            conn2.executemany(
                f"INSERT INTO {RISK_TABLE} (tx_id, risk_score, time_step, computed_at) "
                f"VALUES (?, ?, ?, ?)", rows)
            conn2.execute(f"CREATE INDEX idx_risk_score ON {RISK_TABLE}(risk_score DESC)")
            conn2.execute(f"CREATE INDEX idx_risk_ts ON {RISK_TABLE}(time_step)")
            conn2.commit()
            print(f"[缓存] risk_scores 写入 {human_int(len(rows))} 行")

        row = conn2.execute(f"SELECT computed_at FROM {RISK_TABLE} LIMIT 1").fetchone()
        return row["computed_at"] if row else ""
    finally:
        conn2.close()


@app.get("/top_risk", response_model=TopRiskResponse,
         summary="Top-K 高风险交易", dependencies=AUTH)
def top_risk(
    k: int = Query(20, ge=1, le=1000, description="返回条数"),
    scope: str = Query("unknown", pattern="^(unknown|all)$",
                       description="unknown=仅未标注交易（推荐）；all=全部交易"),
    refresh: bool = Query(False, description="是否强制重算缓存"),
) -> TopRiskResponse:
    t0 = time.perf_counter()

    if scope == "unknown":
        with DB_LOCK:
            conn = get_conn()
            computed_at = ensure_risk_scores(conn, force=refresh)
            total = int(conn.execute(f"SELECT COUNT(*) FROM {RISK_TABLE}").fetchone()[0])
            rows = conn.execute(f"""
                SELECT tx_id, risk_score, time_step FROM {RISK_TABLE}
                ORDER BY risk_score DESC LIMIT ?
            """, (k,)).fetchall()
    else:
        # scope=all：对全库实时打分（含已标注交易，仅用于对比展示）
        features: list[str] = STATE["features"]
        model = STATE["bundle"]["pipeline"]
        pos_idx = list(model.classes_).index(POS_LABEL)
        with DB_LOCK:
            df = pd.read_sql_query(f"""
                SELECT t.tx_id, t.time_step, {select_feature_sql(features)}
                FROM transactions t JOIN graph_features g ON g.tx_id = t.tx_id
            """, get_conn())
        X = np.nan_to_num(df[features].to_numpy(dtype=np.float64), nan=0.0)
        df["risk_score"] = model.predict_proba(X)[:, pos_idx]
        df = df.nlargest(k, "risk_score")
        computed_at = time.strftime("%Y-%m-%d %H:%M:%S")
        total = len(df)
        rows = df[["tx_id", "risk_score", "time_step"]].to_dict("records")

    threshold: float = STATE["threshold"]
    results = [
        TopRiskItem(rank=i + 1, tx_id=int(r["tx_id"]),
                    risk_score=round(float(r["risk_score"]), 6),
                    risk_level=risk_level(float(r["risk_score"]), threshold),
                    time_step=int(r["time_step"]))
        for i, r in enumerate(rows)
    ]
    elapsed = time.perf_counter() - t0
    track(elapsed)
    return TopRiskResponse(
        k=k, scope=scope, total_candidates=total, computed_at=computed_at,
        elapsed_ms=round(elapsed * 1000, 2), disclaimer=DISCLAIMER, results=results)


@app.get("/subgraph/{tx_id}", response_model=SubgraphResponse,
         summary="局部交易图", dependencies=AUTH)
def subgraph(
    tx_id: int,
    hops: int = Query(SUBGRAPH_HOPS, ge=1, le=3, description="跳数，默认 2"),
    max_nodes: int = Query(MAX_SUBGRAPH_NODES, ge=2, le=MAX_SUBGRAPH_NODES,
                           description=f"最大节点数，上限 {MAX_SUBGRAPH_NODES}"),
) -> SubgraphResponse:
    """
    取目标交易周围的局部图。

    **只含可观测历史**：所有返回节点的 time_step <= 目标交易的 time_step。
    本项目实测边只在同一时间步内，所以天然满足；这里仍然显式过滤，
    以保证即使将来数据集补上跨步边也不会泄漏未来信息。
    """
    t0 = time.perf_counter()
    with DB_LOCK:
        conn = get_conn()
        target = conn.execute(
            "SELECT tx_id, time_step FROM transactions WHERE tx_id=?", (tx_id,)).fetchone()
    if target is None:
        raise HTTPException(status_code=404, detail=f"交易 {tx_id} 不存在")

    t_target = int(target["time_step"])

    # 逐跳 BFS，每跳都只允许 time_step <= t_target 的节点
    frontier = {tx_id}
    seen = {tx_id}
    edges: set[tuple[int, int]] = set()
    truncated = False

    with DB_LOCK:
      conn = get_conn()
      for _ in range(hops):
        if not frontier or len(seen) >= max_nodes:
            break
        marks = ",".join("?" * len(frontier))
        sql = f"""
            SELECT e.src, e.dst
            FROM edges e
            JOIN transactions a ON a.tx_id = e.src
            JOIN transactions b ON b.tx_id = e.dst
            WHERE (e.src IN ({marks}) OR e.dst IN ({marks}))
              AND a.time_step <= ? AND b.time_step <= ?
        """
        params = list(frontier) + list(frontier) + [t_target, t_target]
        new_frontier: set[int] = set()
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

    # 节点信息（锁内读取）
    with DB_LOCK:
        conn = get_conn()
        marks = ",".join("?" * len(seen))
        node_rows = conn.execute(
            f"SELECT tx_id, time_step, label FROM transactions WHERE tx_id IN ({marks})",
            list(seen)).fetchall()
        scores: dict[int, float] = {}
        if table_exists(conn, RISK_TABLE):
            srows = conn.execute(
                f"SELECT tx_id, risk_score FROM {RISK_TABLE} WHERE tx_id IN ({marks})",
                list(seen)).fetchall()
            scores = {int(r["tx_id"]): float(r["risk_score"]) for r in srows}

    nodes = [
        SubgraphNode(
            tx_id=int(r["tx_id"]), time_step=int(r["time_step"]),
            risk_score=round(scores[int(r["tx_id"])], 6) if int(r["tx_id"]) in scores else None,
            is_target=int(r["tx_id"]) == tx_id,
            label=(None if r["label"] is None else int(r["label"])) if EXPOSE_LABELS else None,
        )
        for r in node_rows
    ]
    # 只保留两端都在节点集合里的边
    kept_edges = [SubgraphEdge(src=s, dst=d) for s, d in edges if s in seen and d in seen]

    elapsed = time.perf_counter() - t0
    track(elapsed)
    return SubgraphResponse(
        target=tx_id, target_time_step=t_target, hops=hops, observable_only=True,
        truncated=truncated, node_count=len(nodes), edge_count=len(kept_edges),
        elapsed_ms=round(elapsed * 1000, 2), disclaimer=DISCLAIMER,
        nodes=sorted(nodes, key=lambda n: (not n.is_target, n.tx_id)),
        edges=kept_edges,
    )


@app.get("/stats", summary="服务统计", dependencies=AUTH)
def stats() -> dict:
    n = STATE["n_requests"]
    return {
        "requests": n,
        "avg_elapsed_ms": round(STATE["total_seconds"] / n * 1000, 2) if n else 0.0,
        "current_memory_mb": current_memory_mb(),
        "peak_memory_mb": round(STATE["peak_memory_mb"], 2),
        "psutil_available": psutil is not None,
        "model": f"{STATE['group']} + {STATE['model_name']}",
        "threshold": STATE["threshold"],
    }


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):     # pragma: no cover
    return JSONResponse(status_code=500,
                        content={"detail": f"服务内部错误：{type(exc).__name__}: {exc}"})
