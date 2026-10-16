"""
common.py —— 全局配置与公共工具
================================
本文件是全项目唯一的“参数中心”与“路径中心”。所有脚本都从这里导入，
不要在其它脚本里再硬编码路径或划分比例。

【改划分比例 / 改特征组 / 改路径，全部改这里】
  - 改时间划分      -> SPLITS / TRAIN_END / VAL_END
  - 改特征组 A/B/C  -> LOCAL_FEATS / OFFICIAL_FEATS / GRAPH_FEATS
  - 改数据库位置    -> DB_PATH
  - 改数据 CSV 位置 -> DATA_DIR

【设计约束（与毕设硬约束一一对应）】
  1. unknown 标签一律存 NULL，绝不存 0 —— label 列可空，None 表示未知
  2. 时间划分固定 1-29 / 30-34 / 35-49，禁止随机划分
  3. 图特征只用可观测历史结构，禁止未来信息
  4. 类别不平衡用 class_weight='balanced'，看 PR-AUC / F1
  5. 局部图展示邻域 <= 50 节点
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path
from typing import Iterator, Optional

# ============================================================
# 零、控制台编码（Windows 必需）
# ============================================================
# Windows 默认控制台代码页是 GBK(936)，本项目的输出含 '✓' '✗' '…'
# 等字符，直接 print 会抛 UnicodeEncodeError: 'gbk' codec can't encode character。
# 这里把标准输出/错误切成 UTF-8，既保留中文又不会崩。
# 注意：必须在任何 print 之前执行；已在 load_data 等入口统一调用 ensure_utf8_console()。
def ensure_utf8_console() -> None:
    """把 stdout/stderr 切到 UTF-8，避免 Windows GBK 控制台编码报错。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # 某些重定向场景下不可重配置，忽略即可（errors 已在别处兜底）
            pass


ensure_utf8_console()

# ============================================================
# 一、路径中心
# ============================================================
# common.py 位于 <项目根>/src/common.py，向上两级即项目根
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

DATA_DIR: Path = PROJECT_ROOT / "Data"          # 三个原始 CSV 放这里
DOC_DIR: Path = PROJECT_ROOT / "Doc"            # 文档
OUTPUT_DIR: Path = PROJECT_ROOT / "outputs"     # 所有派生结果
MODEL_DIR: Path = OUTPUT_DIR / "models"         # 模型 .pkl
FIGURE_DIR: Path = OUTPUT_DIR / "figures"       # 图片
TABLE_DIR: Path = OUTPUT_DIR / "tables"         # 导出的表格
DB_PATH: Path = PROJECT_ROOT / "elliptic.db"    # SQLite 数据库

# 原始 CSV 文件名（保持官方命名，不重命名文件）
FEATURES_CSV: Path = DATA_DIR / "elliptic_txs_features.csv"
EDGELIST_CSV: Path = DATA_DIR / "elliptic_txs_edgelist.csv"
CLASSES_CSV: Path = DATA_DIR / "elliptic_txs_classes.csv"

# ============================================================
# 二、数据集常量（由实际数据核对得到，2024 校验）
# ============================================================
N_TXS: int = 203_769          # transactions 期望行数
N_EDGES: int = 234_355        # edges 期望行数
N_TIME_STEPS: int = 49        # 时间步 1..49
N_FEATURES: int = 165         # 实际特征数 f1..f165（注意：不是 166！CSV 共 167 列）

# ============================================================
# 三、时间划分（硬约束 2）
# ============================================================
# 语义：time_step <= TRAIN_END -> train
#       TRAIN_END < time_step <= VAL_END -> val
#       time_step > VAL_END -> test
TRAIN_END: int = 29
VAL_END: int = 34

# 供打印/文档使用的可读划分表
SPLITS: dict[str, tuple[int, int]] = {
    "train": (1, TRAIN_END),
    "val": (TRAIN_END + 1, VAL_END),
    "test": (VAL_END + 1, N_TIME_STEPS),
}

# ============================================================
# 四、标签编码（硬约束 1）
# ============================================================
# 官方 classes.csv: '1'=illicit(非法), '2'=licit(合法), 'unknown'=未知
# 本项目的编码： illicita=1, licit=0, unknown=None(存 SQL NULL)
LABEL_ILLICIT: int = 1
LABEL_LICIT: int = 0
CLASS_TO_LABEL: dict[str, Optional[int]] = {
    "1": LABEL_ILLICIT,
    "2": LABEL_LICIT,
    "unknown": None,
}
LABEL_TO_NAME: dict[Optional[int], str] = {
    LABEL_ILLICIT: "illicit",
    LABEL_LICIT: "licit",
    None: "unknown",
}
POS_LABEL: int = LABEL_ILLICIT          # 正类 = illicit，PR-AUC / Recall 都针对它

# ============================================================
# 五、特征组定义（本次对齐后的三组方案）
# ============================================================
# Elliptic 官方说明：f1~f94 为局部（local）特征，f95~f165 为已含 1-hop 聚合的特征。
# A 组 = 纯局部特征
LOCAL_FEATS: list[str] = [f"f{i}" for i in range(1, 95)]        # f1..f94  (94 个)
# C 组 = 官方全量特征
OFFICIAL_FEATS: list[str] = [f"f{i}" for i in range(1, N_FEATURES + 1)]  # f1..f165 (165 个)
# 结构类图特征（B 组在 A 组基础上追加这些；不含任何标签信息）
GRAPH_FEATS: list[str] = [
    "in_degree",            # 入度：作为 txId2 出现的次数
    "out_degree",           # 出度：作为 txId1 出现的次数
    "degree",               # 总度数（无向）
    "clustering",           # 聚类系数（同时间步邻域）
    "nbr_count",            # 去重邻居数
    "two_hop_nbr_count",    # 2 跳邻居数（仅展示/分析，列已算好）
]
# 注意：pagerank_rolling 被刻意排除在 GRAPH_FEATS 之外 ——
#       按对齐结论，PageRank 只做展示，不进入模型特征。

FEATURE_GROUPS: dict[str, list[str]] = {
    "A_local": LOCAL_FEATS,
    "B_local_plus_graph": LOCAL_FEATS + GRAPH_FEATS,
    "C_official_full": OFFICIAL_FEATS,
}

# ============================================================
# 六、评测与不平衡处理（硬约束 4）
# ============================================================
CLASS_WEIGHT = "balanced"     # sklearn 的 class_weight 取值
RANDOM_STATE: int = 42        # 仅用于模型自身的随机性，绝不用于划分数据
PRIMARY_METRIC: str = "pr_auc"   # 主指标：不平衡场景下比 accuracy 有意义得多

# 系统侧限制（硬约束 5、模块5 要求）
MAX_SUBGRAPH_NODES: int = 50    # 单次局部图展示邻域上限
MAX_SCORE_BATCH: int = 1000     # /score 单次请求上限
SUBGRAPH_HOPS: int = 2          # /subgraph 默认跳数

# ============================================================
# 七、公共工具函数
# ============================================================


def ensure_dirs() -> None:
    """确保所有输出目录存在（脚本被 clone 到新机器后可一键跑）。"""
    for d in (OUTPUT_DIR, MODEL_DIR, FIGURE_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)


def time_step_to_split(time_step: int) -> str:
    """时间步 -> split 名。改划分比例只需改 TRAIN_END / VAL_END。"""
    if time_step <= TRAIN_END:
        return "train"
    if time_step <= VAL_END:
        return "val"
    return "test"


def validate_split_edges() -> None:
    """自检：确认划分区间首尾相接、无空洞、无重叠。"""
    assert SPLITS["train"][1] + 1 == SPLITS["val"][0], "train/val 之间有空洞"
    assert SPLITS["val"][1] + 1 == SPLITS["test"][0], "val/test 之间有空洞"
    assert SPLITS["train"][0] == 1, "train 不从第 1 步开始"
    assert SPLITS["test"][1] == N_TIME_STEPS, "test 未覆盖到最后一步"


def get_connection(db_path: Path | str = DB_PATH, readonly: bool = False,
                   check_same_thread: bool = True) -> sqlite3.Connection:
    """
    获取 SQLite 连接。

    readonly=True 时以只读模式打开（供 API / 前端使用，避免误写）。
    注意：只读模式要求文件已存在。

    check_same_thread=False 是给 FastAPI 用的：FastAPI 的同步路由跑在线程池里，
    同一连接会被不同线程使用，SQLite 默认会抛
    "SQLite objects created in a thread can only be used in that same thread"。
    服务端必须显式关掉这个检查，并由上层用锁串行化访问。
    """
    db_path = Path(db_path)
    if readonly:
        if not db_path.exists():
            raise FileNotFoundError(f"数据库不存在：{db_path}，请先运行 src/load_data.py")
        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, check_same_thread=check_same_thread)
    else:
        conn = sqlite3.connect(db_path, check_same_thread=check_same_thread)

    conn.row_factory = sqlite3.Row
    # 性能相关 pragma（对读多写少的分析场景友好）
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")
    # 让 SQLite 能用更多内存做排序/索引，避免大数据量下退化
    conn.execute("PRAGMA cache_size=-200000")   # 约 200MB
    return conn


def select_feature_sql(features: list[str]) -> str:
    """
    拼接 SELECT 的特征列（transactions 别名 t，graph_features 别名 g）。

    【坑】特征组 C（官方全量 f1-f165）里**没有任何图特征**，
    这时图特征部分为空串，若写成 f"{tx_cols}, {g_cols}" 就会留下一个
    多余的逗号，SQLite 直接报 `near "FROM": syntax error`。
    app.py 和 streamlit_app.py 都踩过这个坑，所以统一放这里。
    """
    parts = [f"t.{c}" for c in features if c.startswith("f")]
    parts += [f"g.{c}" for c in features if not c.startswith("f")]
    return ", ".join(parts)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table','view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def batched(iterable: Iterator, size: int) -> Iterator[list]:
    """把迭代器按 size 切块，用于 executemany 批量写入。"""
    buf: list = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf


def human_int(n: int) -> str:
    return f"{n:,}"
