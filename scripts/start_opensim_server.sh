#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# WSL2 OpenSim IK 服务启动脚本
#
# 使用方式 (在WSL2终端中):
#   chmod +x scripts/start_opensim_server.sh
#   ./scripts/start_opensim_server.sh
#
# 或:
#   bash scripts/start_opensim_server.sh
# ═══════════════════════════════════════════════════════════════

set -e

# ─── 配置 ─────────────────────────────────────────────────
CONDA_ENV="${CONDA_ENV:-fitness_coach}"
# 找到 conda 安装路径
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    CONDA_SH="$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    CONDA_SH="/opt/conda/etc/profile.d/conda.sh"
else
    echo "[ERROR] 未找到 conda 安装。请设置 CONDA_SH 环境变量。"
    exit 1
fi

# 项目路径 (Windows端，在WSL2中通过 /mnt 访问)
PROJECT_DIR="/mnt/d/冯老师项目/WorkPlace"

# OpenSim 模型路径
OPENSIM_MODEL_PATH="${PROJECT_DIR}/opensim_models/Rajagopal2015.osim"

# ─── 环境检查 ─────────────────────────────────────────────
echo "============================================"
echo "  AI 健身教练 — OpenSim IK 服务"
echo "============================================"
echo ""
echo "配置:"
echo "  Conda环境:   $CONDA_ENV"
echo "  项目路径:     $PROJECT_DIR"
echo "  OpenSim模型: $OPENSIM_MODEL_PATH"
echo ""

# 检查项目路径
if [ ! -d "$PROJECT_DIR" ]; then
    echo "[WARN] 项目目录不存在: $PROJECT_DIR"
    echo "请确保Windows项目在 D:\\冯老师项目\\WorkPlace"
fi

# ─── 激活Conda环境 ────────────────────────────────────────
echo "[1/3] 激活Conda环境..."
source "$CONDA_SH"
conda activate "$CONDA_ENV"

if [ $? -ne 0 ]; then
    echo "[ERROR] 无法激活Conda环境 '$CONDA_ENV'"
    echo "请先创建环境:"
    echo "  conda create -n fitness_coach python=3.10"
    echo "  conda activate fitness_coach"
    echo "  conda install -c opensim-org opensim"
    echo "  pip install -r ${PROJECT_DIR}/requirements_wsl.txt"
    exit 1
fi

echo "[OK] 环境已激活: $(which python)"

# ─── 检查OpenSim ──────────────────────────────────────────
echo "[2/3] 检查OpenSim安装..."
python -c "import opensim; print(f'[OK] OpenSim {opensim.GetVersion()} 可用')" 2>/dev/null || \
    echo "[WARN] OpenSim Python API 不可用，将使用几何IK回退方案"

# ─── 启动服务 ─────────────────────────────────────────────
echo "[3/3] 启动IK服务..."
echo ""
echo "服务监听:"
echo "  请求端口: 5000"
echo "  结果端口: 5001"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 切换目录并启动
cd "$PROJECT_DIR" 2>/dev/null || true
export OPENSIM_MODEL_PATH
python -m src.bridge.socket_server
