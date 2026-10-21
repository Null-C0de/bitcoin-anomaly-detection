@echo off
REM ===================================================================
REM  Windows 双击启动入口
REM  实际逻辑都在 run.py 里，这里只做两件事：
REM    1) 把控制台切到 UTF-8（否则打印 ✓ / ⚠ 会乱码或报错）
REM    2) 切到脚本所在目录后调用 run.py，并把参数原样透传
REM  例：start.bat --check     start.bat --api-only
REM ===================================================================
chcp 65001 >nul
cd /d "%~dp0"

python run.py %*

REM 服务退出后停一下，避免窗口一闪而过看不到报错
echo.
pause
