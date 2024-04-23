#!/bin/bash

file="./config/python_path.txt"
PYTHON_PATH=""
while IFS= read -r line; do
PYTHON_PATH="$PYTHON_PATH $line"
done < "$file"
echo "$PYTHON_PATH"


# 启动 ComfyUI 脚本
echo "启动 ComfyUI..."

# 指定Python环境和脚本路径
# PYTHON_PATH="./venv/bin/python"
SCRIPT_PATH="ComfyUI/main.py"

# 添加额外的参数
EXTRA_MODEL_PATHS_CONFIG="./extra_model_paths.yaml"
PORT=8188
LISTEN_ADDRESS="0.0.0.0"

# 执行Python脚本
$PYTHON_PATH $SCRIPT_PATH --extra-model-paths-config $EXTRA_MODEL_PATHS_CONFIG --port $PORT --listen $LISTEN_ADDRESS

echo "ComfyUI 已启动."