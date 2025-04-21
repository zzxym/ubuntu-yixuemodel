#!/bin/bash

# ====================== 环境变量定义 ======================
USER="ubuntu"
USER_HOME="/home/${USER}"
NVIDIA_DRIVER_VERSION="570.144.01"
CUDA_VERSION="12.8"
CUDNN_VERSION="8.9.6"
ANACONDA_VERSION="Anaconda3-2023.09-Linux-x86_64.sh"
XINFERENCE_MODEL="Qwen/Qwen-32B"
MODEL_DIR="${USER_HOME}/models/qwen-32b-int4"
SERVICE_NAME="qwen-32b-int4"
PORT=9900

# ====================== 系统基础配置 ======================
sudo apt update && sudo apt upgrade -y -qq
sudo apt install -y wget curl build-essential python3-dev git nvidia-docker2 ufw libgl1-mesa-glx libopenblas-base --no-install-recommends -qq

# ====================== 显卡驱动安装 ======================
sudo systemctl stop lightdm
wget -q https://us.download.nvidia.com/XFree86/Linux-x86_64/${NVIDIA_DRIVER_VERSION}/NVIDIA-Linux-x86_64-${NVIDIA_DRIVER_VERSION}.run -O nvidia-driver.run
sudo chmod +x nvidia-driver.run
sudo ./nvidia-driver.run --silent --no-opengl-files --no-x-check
rm nvidia-driver.run
sudo systemctl start lightdm

# ====================== CUDA & cuDNN安装 ======================
# 安装CUDA
wget -q https://developer.download.nvidia.com/compute/cuda/${CUDA_VERSION}/local_installers/cuda_${CUDA_VERSION}_linux.run -O cuda.run
sudo sh cuda.run --silent --toolkit --no-opengl-libs
rm cuda.run
echo "export PATH=/usr/local/cuda-${CUDA_VERSION}/bin:\$PATH" >> ${USER_HOME}/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:\$LD_LIBRARY_PATH" >> ${USER_HOME}/.bashrc
source ${USER_HOME}/.bashrc

# 安装cuDNN（通过NVIDIA官方API下载，需提前配置认证）
# 注：实际需通过开发者账号获取下载链接，此处为示意
wget -q "https://developer.download.nvidia.com/compute/cudnn/secure/${CUDNN_VERSION}/cuda-${CUDA_VERSION}/libcudnn8_${CUDNN_VERSION}-1+cuda${CUDA_VERSION}_amd64.deb" -O libcudnn8.deb
wget -q "https://developer.download.nvidia.com/compute/cudnn/secure/${CUDNN_VERSION}/cuda-${CUDA_VERSION}/libcudnn8-dev_${CUDNN_VERSION}-1+cuda${CUDA_VERSION}_amd64.deb" -O libcudnn8-dev.deb
sudo dpkg -i libcudnn8*.deb
rm libcudnn8*.deb

# ====================== Anaconda环境搭建 ======================
wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/${ANACONDA_VERSION} -O anaconda.sh
bash anaconda.sh -b -p ${USER_HOME}/anaconda3
rm anaconda.sh
source ${USER_HOME}/.bashrc
conda create -n tcm_env python=3.10 -y
conda activate tcm_env

# ====================== 下载&量化Qwen-32B ======================
# 安装量化工具
pip install -U modelscope transformers accelerate bitsandbytes
pip install "xinference[quantization]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 创建模型目录
mkdir -p ${MODEL_DIR}

# 下载原始模型（需提前在ModelScope申请权限）
xinference download --model-engine transformers --model-name ${XINFERENCE_MODEL} --output-dir ${MODEL_DIR}/original

# 执行4bit量化（使用GPTQ算法，校准数据为中医问诊数据集）
python -m transformers.quantization.quantize \
    --model-path ${MODEL_DIR}/original \
    --quantization-method gptq \
    --bits 4 \
    --output-dir ${MODEL_DIR} \
    --calibration-data-path ./calibration_data.json  # 替换为实际校准数据路径

# ====================== 启动模型服务 ======================
xinference launch \
    --model-engine transformers \
    --model-name ${SERVICE_NAME} \
    --model-path ${MODEL_DIR} \
    --size-in-billions 32 \
    --quantization int4 \
    --device cuda:0 \
    --port ${PORT} \
    --max-batch-size 4 \  # 4090优化后的批量大小
    --max-sequence-length 8192 \
    --load-in-8bit \  # 混合精度加载，进一步优化显存
    --trust-remote-code &  # 允许加载Qwen特有的代码格式

# ====================== RAGFlow配置 ======================
cat > ${USER_HOME}/ragflow-config.yaml <<EOF
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
      path: ${USER_HOME}/chroma_db/tcm_knowledge
      embedding_model: zhipuai/roberta-wwm-ext

document_parsers:
  - name: tcm-parser
    type: rule-based
    config:
      symptom_patterns: ["舌苔\\s*([^\n]+)", "脉象\\s*([^\w]+)"]
EOF

# ====================== 开机自启动配置 ======================
# 创建systemd服务
cat > /etc/systemd/system/${SERVICE_NAME}.service <<EOF
[Unit]
Description=Qwen-32B-INT4 Medical Service
After=network.target
Requires=nvidia-persistenced.service

[Service]
User=${USER}
Environment="PATH=${USER_HOME}/anaconda3/envs/tcm_env/bin:\$PATH"
Environment="CUDA_VISIBLE_DEVICES=0"
WorkingDirectory=${MODEL_DIR}
ExecStart=$(which xinference) launch \
    --model-engine transformers \
    --model-name ${SERVICE_NAME} \
    --model-path ${MODEL_DIR} \
    --size-in-billions 32 \
    --quantization int4 \
    --device cuda:0 \
    --port ${PORT} \
    --max-sequence-length 8192 \
    --load-in-8bit

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

echo -e "\n2. 显存占用检查："
nvidia-smi | grep $(nvidia-smi --query-gpu=pid --format=csv,noheader)

echo -e "\n3. 快速问诊测试："
curl -X POST http://localhost:${PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "'${SERVICE_NAME}'", "messages": [{"role": "user", "content": "患者咳嗽痰黄，舌苔黄腻，脉滑数，开方建议"}]}' | jq .choices[0].message.content

# ====================== 性能优化 ======================
# 启用NVIDIA持久化模式（提升GPU利用率）
sudo nvidia-smi -pm 1

# 清理临时文件
conda clean -a -y
rm -rf nvidia-driver.run cuda.run anaconda.sh
