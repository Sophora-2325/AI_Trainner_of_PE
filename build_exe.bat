@echo off
chcp 65001 >nul
echo ========================================
echo   AI 健身教练 — 打包 exe
echo ========================================
echo.

cd /d "%~dp0"

echo [1/3] 检查 PyInstaller...
pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo 正在安装 PyInstaller...
    pip install pyinstaller
)

echo [2/3] 安装项目依赖...
pip install -r requirements.txt

echo [3/3] 开始打包（约 3-10 分钟，体积约 300-600 MB）...
pyinstaller fitness_coach.spec --noconfirm

if exist "dist\AI健身教练.exe" (
    echo.
    echo ✓ 打包成功: dist\AI健身教练.exe
    echo.
    echo 使用说明:
    echo   1. 将 test_squat.mp4 复制到 exe 同目录（可选，用于自动验收）
    echo   2. 双击 AI健身教练.exe 打开启动器
    echo   3. 大模型功能需另装 Ollama，或勾选「关闭大模型」
    echo   4. 首次运行 Windows 可能提示防火墙，允许本地 WebSocket 即可
) else (
    echo.
    echo ✗ 打包失败，请查看上方错误信息
)

pause
