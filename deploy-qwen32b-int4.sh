#!/bin/bash

# ====================== 环境变量定义（标准化路径） ======================
USER="ubuntu"
USER_HOME="/home/${USER}"
NVIDIA_DRIVER_URL="https://us.download.nvidia.com/XFree86/Linux-x86_64/570.144.01/NVIDIA-Linux-x86_64-570.144.01.run"
CUDA_RUNFILE="cuda_12.8.1_530.54.03_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${CUDA_RUNFILE}"
CUDNN_DEB_URL="https://developer.download.nvidia.com/compute/cudnn/secure/8.9.6/cuda12/libcudnn8_8.9.6.24-1+cuda12_amd64.deb"  # 示例链接，需替换为实际下载地址
ANACONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/Anaconda3-2023.09-Linux-x86_64.sh"
XINFERENCE_MODEL="Qwen/Qwen-32B"
MODEL_DIR="${USER_HOME}/models/qwen-32b-int4"          # 模型存储绝对路径
CALIBRATION_DATA_PATH="${USER_HOME}/data/calibration.json"  # 校准数据绝对路径
RAGFLOW_CONFIG="${USER_HOME}/config/ragflow-tcm.yaml"    # RAGFlow配置绝对路径
XINFERENCE_BIN="${USER_HOME}/anaconda3/envs/tcm_env/bin/xinference"  # xinference绝对路径
SERVICE_NAME="qwen-32b-int4"
PORT=9900

# ====================== 路径存在性检查（健壮性增强） ======================
# 检查关键目录
for dir in "${MODEL_DIR}" "${USER_HOME}/data" "${USER_HOME}/chroma_db" "${USER_HOME}/config"; do
    mkdir -p "$dir" || { echo "错误：无法创建目录 $dir" ; exit 1; }
done

# 检查conda环境是否存在
if [ ! -d "${USER_HOME}/anaconda3/envs/tcm_env" ]; then
    echo "错误：conda环境'tcm_env'未创建，即将创建..."
else
    echo "conda环境已存在，跳过创建..."
fi

# ====================== 系统基础配置 ======================
sudo apt update && sudo apt upgrade -y -qq
sudo apt install -y wget curl build-essential python3-dev git nvidia-docker2 ufw libgl1-mesa-glx libopenblas-base --no-install-recommends -qq

# ====================== 显卡驱动安装 ======================
sudo systemctl stop lightdm
wget -q "${NVIDIA_DRIVER_URL}" -O nvidia-driver.run
sudo chmod +x nvidia-driver.run
sudo ./nvidia-driver.run --silent --no-opengl-files --no-x-check
rm nvidia-driver.run
sudo systemctl start lightdm

# ====================== CUDA安装 ======================
wget -q "${CUDA_URL}" -O cuda.run
sudo chmod +x cuda.run
sudo ./cuda.run --silent --toolkit --no-opengl-libs
rm cuda.run
echo "export PATH=/usr/local/cuda-12.8/bin:\$PATH" >> "${USER_HOME}/.bashrc"
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH" >> "${USER_HOME}/.bashrc"
source "${USER_HOME}/.bashrc"

# ====================== cuDNN安装（需提前获取下载权限） ======================
# 注意：此处需替换为通过NVIDIA开发者账号获取的实际下载链接
wget -q "${CUDNN_DEB_URL}" -O libcudnn8.deb
sudo dpkg -i libcudnn8.deb
rm libcudnn8.deb

# ====================== Anaconda环境搭建 ======================
wget -q "${ANACONDA_URL}" -O anaconda.sh
bash anaconda.sh -b -p "${USER_HOME}/anaconda3"
rm anaconda.sh
source "${USER_HOME}/.bashrc"
conda create -n tcm_env python=3.10 -y || echo "环境已存在，跳过创建"
conda activate tcm_env

# ====================== 安装依赖 ======================
pip install -U modelscope transformers accelerate bitsandbytes
pip install "xinference[quantization]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# ====================== 下载原始模型（需ModelScope权限） ======================
mkdir -p "${MODEL_DIR}/original"
xinference download --model-engine transformers --model-name "${XINFERENCE_MODEL}" --output-dir "${MODEL_DIR}/original"

# ====================== 执行4bit量化（GPTQ算法，使用绝对路径校准数据） ======================
python -m transformers.quantization.quantize \
    --model-path "${MODEL_DIR}/original" \
    --quantization-method gptq \
    --bits 4 \
    --output-dir "${MODEL_DIR}" \
    --calibration-data-path "${CALIBRATION_DATA_PATH}"  # 绝对路径

# ====================== 启动模型服务（前台测试，生产环境用systemd） ======================
# nohup ${XINFERENCE_BIN} launch \
#     --model-engine transformers \
#     --model-name "${SERVICE_NAME}" \
#     --model-path "${MODEL_DIR}" \
#     --size-in-billions 32 \
#     --quantization int4 \
#     --device cuda:0 \
#     --port "${PORT}" \
#     --max-batch-size 4 \
#     --max-sequence-length 8192 \
#     --load-in-8bit \
#     --trust-remote-code &

# ====================== RAGFlow配置（绝对路径定义） ======================
cat > "${RAGFLOW_CONFIG}" <<EOF
model_services:
  - name: ${SERVICE_NAME}
    type: xinference
    config:
      endpoint: http://localhost:${PORT}
      model: ${SERVICE_NAME}
      quantization: int4

vector_databases:
  - name: tcm-kb
    type: chroma
    config:
      path: ${USER_HOME}/chroma_db/tcm_knowledge  # 向量数据库绝对路径
      embedding_model: zhipuai/roberta-wwm-ext

document_parsers:
  - name: tcm-parser
    type: rule-based
    config:
      symptom_patterns: ["舌苔\\s*([^\n]+)", "脉象\\s*([^\w]+)"]
EOF

# ====================== systemd服务配置（绝对路径确保启动可靠性） ======================
cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=Qwen-32B-INT4 Medical Service
After=network.target
Requires=nvidia-persistenced.service

[Service]
User=${USER}
Environment="CUDA_VISIBLE_DEVICES=0"
WorkingDirectory=${MODEL_DIR}
ExecStart=${XINFERENCE_BIN} launch \
    --model-engine transformers \
    --model-name ${SERVICE_NAME} \
    --model-path ${MODEL_DIR} \
    --size-in-billions 32 \
    --quantization int4 \
    --device cuda:0 \
    --port ${PORT} \
    --max-sequence-length 8192 \
    --load-in-8bit \
    --trust-remote-code

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable ${SERVICE_NAME}
sudo systemctl start ${SERVICE_NAME}

# ====================== 环境验证 ======================
echo -e "\n===================== 部署完成 ==================="
echo "1. 模型服务状态："
systemctl status ${SERVICE_NAME} --no-pager

echo -e "\n2. 显存占用检查（预期18-20GB）："
nvidia-smi | grep -A 3 "Processes: GPU 0"

echo -e "\n3. 快速问诊测试（按Ctrl+C停止）："
while true; do
    read -p "输入问诊内容（输入exit退出）: " query
    if [ "$query" == "exit" ]; then
        break
    fi
    curl -s http://localhost:${PORT}/v1/chat/completions \
        -H "Content-Type: application/json" \
        -d "{\"model\": \"${SERVICE_NAME}\", \"messages\": [{\"role\": \"user\", \"content\": \"${query}\"}]}" \
        | jq .choices[0].message.content
done

# ====================== 清理临时文件 ======================
rm -rf nvidia-driver.run cuda.run anaconda.sh
conda clean -a -y
