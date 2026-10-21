#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run.py —— 一键启动脚本（环境自检 → 按需构建 → 启动 API + 前端）

【这个脚本解决什么问题】
本项目有 6 个模块，正确的使用顺序是
    load_data → graph_features → train_compare → error_analysis → app / streamlit_app
手工依次执行容易漏步骤、报错信息也难定位。本脚本把它收敛成一条命令，
并**先检查再动手**：缺什么补什么，不重复跑已经完成的重活。

【为什么两个服务是独立的】
`src/app.py`（FastAPI）和 `src/streamlit_app.py`（Streamlit）都**直接读同一个
SQLite 文件**，前端不通过 HTTP 调后端（streamlit_app.py 里没有任何 requests 调用）。
所以：
  - 两个服务启动顺序无所谓；
  - 只起前端也能看完整的模型对比 / Top-K / 局部图 / 误报漏报；
  - 只起 API 适合用 curl / Postman / 其他程序调用。
两者同时开的好处是：一个演示界面，一个对外接口，答辩时都能现场演示。

【用法】
    python run.py                     # 检查 + 按需补齐必需产物 + 同时启动两个服务
    python run.py --check             # 只做环境和产物检查，不启动任何服务
    python run.py --build             # 从头重建全部产物（含 SHAP，约 7 分钟）
    python run.py --build-core        # 只重建必需产物（数据+图特征+模型，约 2 分钟）
    python run.py --api-only          # 只启动 API
    python run.py --ui-only           # 只启动前端
    python run.py --port-api 8000 --port-ui 8501
    python run.py --no-browser        # 不自动打开浏览器
    python run.py --api-key mykey     # 给 API 加鉴权（等价于设 DSH_API_KEY）

【环境变量】
    DSH_API_KEY          设置后 API 的 5 个接口都要求请求头 X-API-Key，否则 401
    DSH_EXPOSE_LABELS=1  让接口/前端在响应里带上真实标签（**仅限本地调试**）

【为什么脚本顶部要改 stdout 编码】
Windows 控制台默认 GBK。本项目大量使用 ✓ / ⚠ / ─ 等字符，
直接 print 会抛
    UnicodeEncodeError: 'gbk' codec can't encode character '\\u2713'
而且常常发生在"活已经干完了、只是最后一行打印失败"的时候，极易误判为任务失败。
所以第一件事就是把标准输出切成 UTF-8。
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path


# ============================================================
# 0. 控制台编码：必须在任何 print 之前执行
# ============================================================
def _force_utf8_console() -> None:
    """把 stdout/stderr 切成 UTF-8，避免 GBK 控制台打印符号时崩溃。"""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]
        except (AttributeError, ValueError):
            # 被重定向到不支持 reconfigure 的对象时忽略，不影响主流程
            pass


_force_utf8_console()


# ============================================================
# 1. 常量
# ============================================================
PROJECT_ROOT: Path = Path(__file__).resolve().parent
SRC_DIR: Path = PROJECT_ROOT / "src"
DATA_DIR: Path = PROJECT_ROOT / "Data"
DB_PATH: Path = PROJECT_ROOT / "elliptic.db"
MODEL_DIR: Path = PROJECT_ROOT / "outputs" / "models"
FIGURE_DIR: Path = PROJECT_ROOT / "outputs" / "figures"

FEATURES_CSV: Path = DATA_DIR / "elliptic_txs_features.csv"
EDGELIST_CSV: Path = DATA_DIR / "elliptic_txs_edgelist.csv"
CLASSES_CSV: Path = DATA_DIR / "elliptic_txs_classes.csv"

# 期望的模型数量：A/B/C 三组 × LR/RF 两个模型 = 6 个 .pkl
EXPECTED_MODELS: int = 6

# 依赖分两类：缺了核心依赖根本跑不起来；可选依赖只影响某个功能
CORE_DEPS: dict[str, str] = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "networkx": "networkx",
    "matplotlib": "matplotlib",
}
OPTIONAL_DEPS: dict[str, str] = {
    "fastapi": "整个 API 服务（模块5）",
    "uvicorn": "启动 API 服务（模块5）",
    "streamlit": "网页前端（模块6）",
    "pyvis": "前端里的交互式局部图（模块6）",
    "shap": "SHAP 可解释性（模块4）",
    "jinja2": "前端带样式的表格（pandas .style 依赖它）",
    "psutil": "接口返回峰值内存（缺了自动降级）",
}

# 构建阶段定义：顺序不能乱，每一步都依赖前一步的产物
BUILD_STEPS: list[tuple[str, str, list[str]]] = [
    # (阶段名, 脚本, 额外参数)
    ("数据导入 load_data", "load_data.py", []),
    ("图特征 graph_features", "graph_features.py", []),
    ("模型训练 train_compare", "train_compare.py", []),
    ("误差分析 error_analysis", "error_analysis.py", []),
]
CORE_STEP_COUNT: int = 3      # 前 3 步是"必需"，第 4 步（SHAP）是增强


# ============================================================
# 2. 输出小工具
# ============================================================
def hr(title: str = "", width: int = 74) -> None:
    """打印一条分隔线，带可选标题。"""
    if title:
        print("\n" + "=" * width)
        print(title)
        print("=" * width)
    else:
        print("-" * width)


def step(msg: str) -> None:
    print(f"\n>>> {msg}")


def ok(msg: str) -> None:
    print(f"  [OK]   {msg}")


def warn(msg: str) -> None:
    print(f"  [WARN] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


# ============================================================
# 3. 环境检查
# ============================================================
def check_python() -> bool:
    """Python 版本检查。项目用了 `list[str]` 这类 3.9+ 的写法，但 type 语法需要 3.10+。"""
    step("检查 Python 版本")
    major, minor = sys.version_info[:2]
    print(f"  当前解释器：{sys.executable}")
    print(f"  版本：{sys.version.split()[0]}")
    if (major, minor) < (3, 10):
        bad(f"需要 Python 3.10 及以上，当前 {major}.{minor}")
        return False
    ok("Python 版本满足要求（>=3.10）")
    return True


def _can_import(module: str) -> bool:
    """用当前解释器子进程试探能否 import，避免把依赖装到别的环境里。"""
    try:
        r = subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60,
        )
        return r.returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def check_deps(need_ui: bool, need_api: bool, need_shap: bool) -> tuple[bool, list[str]]:
    """
    检查依赖。

    【为什么要按需检查】
    `--api-only` 时没必要因为缺 streamlit 就拒绝启动。按本次要跑的组件
    来判断，能少装很多包。返回 (是否可用, 缺失包的建议安装命令)。
    """
    step("检查依赖包")

    required: dict[str, str] = dict(CORE_DEPS)
    if need_api:
        required["fastapi"] = "fastapi"
        required["uvicorn"] = "uvicorn"
    if need_ui:
        required["streamlit"] = "streamlit"
        required["pyvis"] = "pyvis"
        required["jinja2"] = "jinja2"      # pandas 的 .style 强依赖
    if need_shap:
        required["shap"] = "shap"

    missing: list[str] = []
    for module, pip_name in required.items():
        if _can_import(module):
            print(f"    [OK]   {module}")
        else:
            print(f"    [MISS] {module}")
            missing.append(pip_name)

    # 可选依赖单独提示，不阻断启动
    optional_missing = [pip for mod, pip in OPTIONAL_DEPS.items()
                        if pip not in required and not _can_import(mod)]
    if optional_missing:
        warn(f"可选依赖缺失（相关功能会降级）：{', '.join(optional_missing)}")

    if missing:
        bad(f"缺少必需依赖：{', '.join(missing)}")
        print("\n  安装命令：")
        print(f"    {sys.executable} -m pip install {' '.join(missing)}")
        print("  或一次性装全：")
        print(f"    {sys.executable} -m pip install -r {PROJECT_ROOT / 'requirements.txt'}")
        return False, missing
    ok("必需依赖齐全")
    return True, []


def check_data_files() -> bool:
    """三个原始 CSV 是 203,769 笔交易的全部来源，缺了什么都做不了。"""
    step("检查原始数据文件")
    all_ok = True
    for path in (FEATURES_CSV, EDGELIST_CSV, CLASSES_CSV):
        if path.exists():
            ok(f"{path.name}  {human_bytes(path.stat().st_size)}")
        else:
            bad(f"缺少 {path} —— 请先把 Elliptic 数据集的 3 个 CSV 放进 Data/ 目录")
            all_ok = False
    return all_ok


def probe_artifacts() -> dict[str, bool]:
    """
    探测各阶段产物是否已经就绪。

    【为什么用 sqlite 直接查而不看文件是否存在】
    `elliptic.db` 可能在导入到一半时中断，文件存在但表是空的。
    所以每个阶段都检查"表里有数据"，而不是"文件在不在"。
    """
    state = {
        "db": False,
        "transactions": False,
        "edges": False,
        "graph_features": False,
        "experiments": False,
        "models": False,
        "error_cases": False,
        "figures": False,
    }

    if DB_PATH.exists():
        state["db"] = True
        import sqlite3
        try:
            conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=5)
            try:
                def count(table: str) -> int:
                    try:
                        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
                    except sqlite3.Error:
                        return 0

                state["transactions"] = count("transactions") > 0
                state["edges"] = count("edges") > 0
                state["graph_features"] = count("graph_features") > 0
                state["experiments"] = count("experiments") > 0
                state["error_cases"] = count("error_cases") > 0
            finally:
                conn.close()
        except sqlite3.Error as exc:
            warn(f"读取数据库失败：{exc}")

    if MODEL_DIR.exists():
        state["models"] = len(list(MODEL_DIR.glob("*.pkl"))) >= EXPECTED_MODELS
    if FIGURE_DIR.exists():
        state["figures"] = any(FIGURE_DIR.glob("*.png"))

    return state


def describe_artifacts(state: dict[str, bool]) -> None:
    step("检查已构建的产物")
    labels = [
        ("db", "elliptic.db"),
        ("transactions", "transactions 表"),
        ("edges", "edges 表"),
        ("graph_features", "graph_features 表"),
        ("experiments", "experiments 表"),
        ("models", f"模型文件（{EXPECTED_MODELS} 个 .pkl）"),
        ("error_cases", "error_cases 表（SHAP 增强）"),
        ("figures", "结果图片"),
    ]
    for key, text in labels:
        if state.get(key):
            ok(text)
        else:
            warn(f"{text}  —— 未就绪")


def decide_build_plan(state: dict[str, bool], mode: str) -> list[tuple[str, str, list[str]]]:
    """
    根据探测结果决定要跑哪几步。

    mode:
        "full"  强制重建全部（含 SHAP）
        "core"  强制重建必需部分
        "auto"  缺什么补什么；第 4 步（SHAP）不自动跑，太慢
        "none"  不构建
    """
    if mode == "none":
        return []
    if mode == "full":
        return list(BUILD_STEPS)
    if mode == "core":
        return list(BUILD_STEPS[:CORE_STEP_COUNT])

    # auto：按依赖顺序，后面步骤依赖前面产物
    plan: list[tuple[str, str, list[str]]] = []
    need_load = not (state["transactions"] and state["edges"])
    need_graph = not state["graph_features"]
    need_train = not (state["experiments"] and state["models"])

    if need_load:
        plan.append(BUILD_STEPS[0])
        # 数据表被重建 = 库是从头导入的，下游产物一定要重算：
        # 否则会出现"新数据 + 旧模型"这种静默不一致，指标看着正常但根本对不上。
        need_graph = True
        need_train = True
    if need_graph:
        plan.append(BUILD_STEPS[1])
    if need_train:
        plan.append(BUILD_STEPS[2])
    return plan


# ============================================================
# 4. 构建
# ============================================================
def run_step(name: str, script: str, extra: list[str]) -> bool:
    """运行一个构建阶段，实时透传输出。返回是否成功。"""
    script_path = SRC_DIR / script
    if not script_path.exists():
        bad(f"找不到脚本 {script_path}")
        return False

    cmd = [sys.executable, "-u", str(script_path), *extra]
    print(f"\n  命令：{' '.join(cmd)}")
    t0 = time.time()
    try:
        # 不捕获输出：让子进程直接写终端，长任务才能看到实时进度
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    except KeyboardInterrupt:
        print("\n  [中断] 用户按下了 Ctrl+C")
        return False
    except OSError as exc:
        bad(f"启动子进程失败：{exc}")
        return False

    elapsed = time.time() - t0
    if result.returncode == 0:
        ok(f"{name} 完成，耗时 {elapsed:.1f}s")
        return True
    bad(f"{name} 失败，退出码 {result.returncode}，耗时 {elapsed:.1f}s")
    return False


def build(plan: list[tuple[str, str, list[str]]]) -> bool:
    if not plan:
        return True
    hr("开始构建产物（缺失的步骤会自动补齐）")
    for i, (name, script, extra) in enumerate(plan, 1):
        step(f"[{i}/{len(plan)}] {name}")
        if not run_step(name, script, extra):
            print("\n构建中断。修复上面的报错后重新执行 `python run.py` 即可，")
            print("已完成的步骤不会重跑。")
            return False
    return True


# ============================================================
# 5. 端口与进程管理
# ============================================================
def port_is_free(port: int, host: str = "127.0.0.1") -> bool:
    """尝试绑定端口判断是否被占用。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def pick_port(preferred: int) -> int:
    """首选端口被占用时，往后找 20 个端口。"""
    if port_is_free(preferred):
        return preferred
    for offset in range(1, 20):
        candidate = preferred + offset
        if port_is_free(candidate):
            warn(f"端口 {preferred} 已被占用，改用 {candidate}")
            return candidate
    raise RuntimeError(f"从 {preferred} 起连续 20 个端口都被占用，请手动用 --port-api 指定")


def start_process(cmd: list[str], title: str) -> subprocess.Popen:
    """启动一个长期运行的服务进程，输出直接透传到当前终端。"""
    print(f"  启动 {title}")
    print(f"    命令：{' '.join(cmd)}")
    return subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), env=os.environ.copy())


def stop_processes(procs: list[tuple[str, subprocess.Popen]]) -> None:
    """
    优雅关闭所有子进程。

    【为什么先 terminate 再 kill】
    terminate() 在 Windows 上是 TerminateProcess，会直接终止；
    但如果进程已经自己退出，再 terminate 会抛异常，所以要判断 poll()。
    先 terminate 再等 3 秒，还没退就 kill，保证 Ctrl+C 之后终端能干净返回。
    """
    for title, proc in procs:
        if proc.poll() is not None:
            continue
        print(f"  正在关闭 {title}（pid={proc.pid}）…")
        try:
            proc.terminate()
        except OSError:
            pass
    deadline = time.time() + 3.0
    for title, proc in procs:
        while proc.poll() is None and time.time() < deadline:
            time.sleep(0.1)
        if proc.poll() is None:
            print(f"  {title} 未响应 terminate，强制结束")
            try:
                proc.kill()
            except OSError:
                pass


def wait_for_api(port: int, api_key: str | None, timeout: float = 60.0) -> bool:
    """
    轮询 /health 直到服务就绪。

    注意：如果启用了鉴权，/health 也必须带 X-API-Key，否则会拿到 401。
    401 说明"服务已起来、只是鉴权拒绝"，这本身也是就绪信号。
    """
    url = f"http://127.0.0.1:{port}/health"
    headers = {"X-API-Key": api_key} if api_key else {}
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except urllib.error.HTTPError as exc:
            if exc.code == 401:      # 服务活着，只是没通过鉴权
                return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(0.5)
    return False


# ============================================================
# 6. 主流程
# ============================================================
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Elliptic 比特币异常交易识别系统 —— 一键启动",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--check", action="store_true", help="只检查环境和产物，不启动服务")
    ap.add_argument("--build", action="store_true", help="强制重建全部产物（含 SHAP）")
    ap.add_argument("--build-core", action="store_true", help="强制重建必需产物（不含 SHAP）")
    ap.add_argument("--api-only", action="store_true", help="只启动 FastAPI 服务")
    ap.add_argument("--ui-only", action="store_true", help="只启动 Streamlit 前端")
    ap.add_argument("--port-api", type=int, default=8000, help="API 端口（默认 8000）")
    ap.add_argument("--port-ui", type=int, default=8501, help="前端端口（默认 8501）")
    ap.add_argument("--api-key", default=None,
                    help="给 API 加鉴权，等价于设置 DSH_API_KEY")
    ap.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = ap.parse_args(argv)

    if args.api_only and args.ui_only:
        bad("--api-only 和 --ui-only 不能同时使用")
        return 2

    need_api = not args.ui_only
    need_ui = not args.api_only

    # 把 --api-key 注入环境变量：app.py 在 import 时读取 DSH_API_KEY
    if args.api_key:
        os.environ["DSH_API_KEY"] = args.api_key
    api_key = os.environ.get("DSH_API_KEY") or None

    hr("Elliptic 比特币异常交易识别系统 · 启动自检")
    print(f"  项目根目录：{PROJECT_ROOT}")
    print(f"  组件：{'API ' if need_api else ''}{'+ 前端' if need_ui and need_api else ('前端' if need_ui else '')}")

    # ---- 6.1 环境 ----
    if not check_python():
        return 2

    # --check 时不因为缺 shap 就报错，只如实反映
    if not check_data_files():
        print("\n原始数据缺失，无法继续。请参照 README 第 2 节准备数据。")
        return 2

    deps_ok, _ = check_deps(need_ui=need_ui, need_api=need_api, need_shap=False)
    if not deps_ok:
        return 2

    # ---- 6.2 产物 ----
    state = probe_artifacts()
    describe_artifacts(state)

    if args.check:
        missing_shap = not state["error_cases"]
        print()
        if not state["graph_features"] or not state["models"]:
            print("结论：必需产物不全，执行 `python run.py` 会自动补齐。")
        else:
            print("结论：必需产物齐全，可以直接启动。")
        if missing_shap:
            print("提示：error_cases 未生成（SHAP 增强），前端\"误报漏报\"页会为空；"
                  "执行 `python run.py --build` 可补齐（约 5 分钟）。")
        return 0

    # ---- 6.3 构建 ----
    if args.build:
        mode = "full"
    elif args.build_core:
        mode = "core"
    else:
        mode = "auto"

    plan = decide_build_plan(state, mode)
    if plan:
        # 必需产物缺失时自动补齐；SHAP 只在显式 --build 时跑
        if not build(plan):
            return 1
        state = probe_artifacts()
    else:
        ok("产物齐全，跳过构建")

    if need_ui and not state["error_cases"]:
        warn("error_cases 表为空：前端\"误报漏报\"页会显示为空。"
             "需要的话执行 `python run.py --build`（约 5 分钟）。")

    # ---- 6.4 启动服务 ----
    hr("启动服务")
    procs: list[tuple[str, subprocess.Popen]] = []
    api_port = ui_port = None

    try:
        if need_api:
            api_port = pick_port(args.port_api)
            cmd = [
                sys.executable, "-m", "uvicorn", "app:app",
                "--app-dir", str(SRC_DIR),
                "--host", "127.0.0.1",
                "--port", str(api_port),
                "--log-level", "warning",
            ]
            procs.append(("API", start_process(cmd, "FastAPI 服务（模块5）")))

        if need_ui:
            ui_port = pick_port(args.port_ui)
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                str(SRC_DIR / "streamlit_app.py"),
                "--server.port", str(ui_port),
                "--server.headless", "true",       # 自己控制是否开浏览器
                "--browser.gatherUsageStats", "false",
            ]
            procs.append(("前端", start_process(cmd, "Streamlit 前端（模块6）")))

        # 等 API 就绪（前端不需要等，它直接读库）
        if need_api and api_port:
            print(f"\n  等待 API 就绪（最多 60 秒）…")
            if wait_for_api(api_port, api_key, timeout=60):
                ok("API 已就绪")
            else:
                warn("API 在 60 秒内没有响应，请看上面的报错日志")

        # 打印访问地址
        print()
        hr("服务地址")
        if api_port:
            print(f"  API        http://127.0.0.1:{api_port}")
            print(f"  交互式文档  http://127.0.0.1:{api_port}/docs")
            if api_key:
                print(f"  鉴权       已启用（请求头 X-API-Key: {'*' * 8}）")
                print(f"  试一下     curl -H \"X-API-Key: 你的密钥\" "
                      f"http://127.0.0.1:{api_port}/health")
            else:
                print("  鉴权       未启用（本地演示模式）")
        if ui_port:
            print(f"  前端        http://127.0.0.1:{ui_port}")

        if need_ui and not args.no_browser and ui_port:
            # 等前端把端口真正监听起来，否则浏览器可能打开过快拿到连接失败
            for _ in range(60):
                if not port_is_free(ui_port):
                    break
                time.sleep(0.5)
            try:
                webbrowser.open(f"http://127.0.0.1:{ui_port}")
                print("  已尝试打开浏览器（--no-browser 可关闭）")
            except Exception:            # noqa: BLE001 - 打不开浏览器不算失败
                pass

        print()
        hr()
        print("  服务运行中。按 Ctrl+C 停止。")
        hr()

        # 阻塞在这里，直到用户 Ctrl+C 或某个进程异常退出
        while True:
            time.sleep(0.5)
            for title, proc in procs:
                code = proc.poll()
                if code is not None:
                    bad(f"{title} 进程已退出（退出码 {code}）")
                    return 1

    except KeyboardInterrupt:
        print("\n\n收到 Ctrl+C，正在关闭服务…")
        return 0
    finally:
        if procs:
            stop_processes(procs)
            print("服务已全部关闭。")


if __name__ == "__main__":
    raise SystemExit(main())
