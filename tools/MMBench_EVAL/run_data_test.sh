#!/usr/bin/env bash
set -euo pipefail

# Network test configuration. Each value can be overridden when invoking this script.
TASK_NAME="${TASK_NAME:-mmbench_en_dev_static}"
TASK_DIR_NAME="${TASK_NAME//\//_}"
SERVER_IP="${SERVER_IP:-}"
SERVER_PORT="${SERVER_PORT:-8080}"
IMAGE_WIDTH="${IMAGE_WIDTH:-384}"
IMAGE_HEIGHT="${IMAGE_HEIGHT:-384}"

# The server's /v1/models model id. Normally this does not need changing.
MODEL_NAME="${MODEL_NAME:-Qwen3-VL-4B}"
API_KEY="${API_KEY:-no-key}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Activate the Conda or uv environment first. PYTHON_BIN may override it.
PYTHON_BIN="${PYTHON_BIN:-python}"
OUTPUT_DIR="$SCRIPT_DIR/results/${TASK_DIR_NAME}_$(date +%Y%m%d_%H%M%S)"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "未找到测试环境：$PYTHON_BIN"
    echo "请先激活 Conda 或 uv 环境，并按 README.md 完成安装。"
    echo "Python environment not found: $PYTHON_BIN"
    echo "Activate the Conda or uv environment, then follow README.md."
    exit 1
fi

if [[ -z "$SERVER_IP" ]]; then
    echo "SERVER_IP 不能为空，请填写板端网络 IP。"
    echo "SERVER_IP is required. Set it to the board's network IP."
    exit 1
fi

if ! [[ "$SERVER_PORT" =~ ^[1-9][0-9]*$ && "$IMAGE_WIDTH" =~ ^[1-9][0-9]*$ && "$IMAGE_HEIGHT" =~ ^[1-9][0-9]*$ ]]; then
    echo "SERVER_PORT、IMAGE_WIDTH、IMAGE_HEIGHT 必须是正整数。"
    echo "SERVER_PORT, IMAGE_WIDTH, and IMAGE_HEIGHT must be positive integers."
    exit 1
fi

if ! curl --fail --silent --show-error --max-time 10 "http://${SERVER_IP}:${SERVER_PORT}/health" >/dev/null; then
    echo "板端 /health 不可访问：http://${SERVER_IP}:${SERVER_PORT}/health"
    echo "Board /health is unavailable: http://${SERVER_IP}:${SERVER_PORT}/health"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

ARGS=(
    --task-name "$TASK_NAME"
    --server-ip "$SERVER_IP"
    --server-port "$SERVER_PORT"
    --image-width "$IMAGE_WIDTH"
    --image-height "$IMAGE_HEIGHT"
    --model-name "$MODEL_NAME"
    --api-key "$API_KEY"
    --output-path "$OUTPUT_DIR"
)

echo "开始数据集测试：$TASK_NAME，结果目录：$OUTPUT_DIR"
echo "Starting dataset evaluation: $TASK_NAME, result directory: $OUTPUT_DIR"
exec "$PYTHON_BIN" "$SCRIPT_DIR/run_data_test.py" "${ARGS[@]}"
