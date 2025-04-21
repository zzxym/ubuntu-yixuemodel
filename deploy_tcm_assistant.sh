#!/bin/bash

# 定义用户和路径变量
USER="ubuntu"
USER_HOME="/home/${USER}"
NVIDIA_DRIVER_URL="https://us.download.nvidia.com/XFree86/Linux-x86_64/570.144.01/NVIDIA-Linux-x86_64-570.144.01.run"
CUDA_RUNFILE="cuda_12.8.1_530.54.03_linux.run"
CUDA_URL="https://developer.download.nvidia.com/compute/cuda/12.8.1/local_installers/${CUDA_RUNFILE}"
# cuDNN相关信息
CUDNN_VERSION="cudnn-linux-x86_64-8.9.6.24_cuda12-archive"
CUDNN_URL="https://developer.download.nvidia.com/compute/cudnn/secure/${CUDNN_VERSION}/${CUDNN_VERSION}.tar.xz"
ANACONDA_VERSION="Anaconda3-2023.09-Linux-x86_64.sh"
ANACONDA_URL="https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/${ANACONDA_VERSION}"
XINFERENCE_DIR="${USER_HOME}/Xinference"
BIANCANG_MODEL_DIR="${USER_HOME}/models/biancang-tcm"
RAGFLOW_CONFIG="${USER_HOME}/ragflow-tcm-config.yaml"

# 系统基础配置
sudo apt update && sudo apt upgrade -y -qq
sudo apt install -y wget curl build-essential python3-dev git nvidia-docker2 ufw libgl1-mesa-glx libopenblas-base --no-install-recommends -qq

# 安装NVIDIA驱动
sudo systemctl stop lightdm
wget -q ${NVIDIA_DRIVER_URL} -O nvidia-driver.run
sudo chmod +x nvidia-driver.run
sudo ./nvidia-driver.run --silent --no-opengl-files --no-x-check
rm nvidia-driver.run
sudo systemctl start lightdm

# 安装CUDA
wget -q ${CUDA_URL} -O ${CUDA_RUNFILE}
sudo chmod +x ${CUDA_RUNFILE}
sudo ./${CUDA_RUNFILE} --silent --toolkit --no-opengl-libs
rm ${CUDA_RUNFILE}
echo "export PATH=/usr/local/cuda-12.8/bin:\$PATH" >> ${USER_HOME}/.bashrc
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:\$LD_LIBRARY_PATH" >> ${USER_HOME}/.bashrc
source ${USER_HOME}/.bashrc

# 安装cuDNN
wget -q ${CUDNN_URL} -O cudnn.tar.xz
tar -xJf cudnn.tar.xz
sudo cp ${CUDNN_VERSION}/include/cudnn*.h /usr/local/cuda/include
sudo cp ${CUDNN_VERSION}/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn*.h /usr/local/cuda/lib64/libcudnn*
rm -rf ${CUDNN_VERSION} cudnn.tar.xz

# 安装Anaconda
wget -q ${ANACONDA_URL} -O anaconda.sh
bash anaconda.sh -b -p ${USER_HOME}/anaconda3
rm anaconda.sh
source ${USER_HOME}/.bashrc
conda create -n tcm_env python=3.10 -y
conda activate tcm_env

# 安装Xinference
pip install "xinference[transformers]" -i https://pypi.tuna.tsinghua.edu.cn/simple

# 下载扁仓中医大模型（需提前获取权限）
mkdir -p ${BIANCANG_MODEL_DIR}
XINFERENCE_MODEL_SRC=modelscope xinference download \
    --model-engine transformers \
    --model-name biancang-tcm \
    --output-dir ${BIANCANG_MODEL_DIR}

# 启动扁仓中医大模型服务
xinference launch \
    --model-engine transformers \
    --model-name biancang-tcm \
    --model-path ${BIANCANG_MODEL_DIR} \
    --size-in-billions 7 \
    --quantization int8 \
    --device cuda:0 \
    --port 9900 \
    --max-batch-size 8 \
    --max-sequence-length 4096 &

# 配置RAGFlow
cat > ${RAGFLOW_CONFIG} <<EOF
model_services:
  - name: biancang-tcm
    type: xinference
    config:
      endpoint: http://localhost:9900
      model: biancang-tcm

vector_databases:
  - name: tcm-cases
    type: chroma
    config:
      path: ${USER_HOME}/chroma_db/tcm_cases
      embedding_model: nghuyong/ernie-3.0-nano-zh

document_parsers:
  - name: tcm-doc-parser
    type: custom
    config:
      parser_script: ${USER_HOME}/scripts/tcm_parser.py
      file_types: ["doc", "docx", "pdf"]
      metadata_fields: ["patient_id", "diagnosis_date", "chief_complaint"]
EOF

# 创建中医文档解析脚本
cat > ${USER_HOME}/scripts/tcm_parser.py <<EOF
import re
from ragflow import Document

def parse_tcm_document(file_path, metadata):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    patient_id = re.search(r'病历号：(\d{6,})', content).group(1)
    diagnosis = re.search(r'中医诊断：(.+?)\n', content, re.DOTALL).group(1)
    prescription = re.findall(r'【处方】(.+?)\n', content, re.DOTALL)
    
    return Document(
        content=content,
        metadata={
            **metadata,
            "patient_id": patient_id,
            "diagnosis": diagnosis,
            "prescription": "\n".join(prescription)
        }
    )
EOF

# 启动RAGFlow容器
docker run -d \
    -p 8000:8000 \
    -v ${RAGFLOW_CONFIG}:/app/config.yaml \
    -v ${USER_HOME}/scripts:/app/scripts \
    --name tcm-ragflow \
    --gpus all \
    --restart unless-stopped \
    ragflow/ragflow:latest

# 开机自启动配置
# Xinference服务自启动
cat > /etc/systemd/system/xinference-tcm.service <<EOF
[Unit]
Description=Biancang TCM Model Service
After=network.target docker.service
Requires=docker.service

[Service]
User=${USER}
WorkingDirectory=${XINFERENCE_DIR}
Environment="CONDA_HOME=${USER_HOME}/anaconda3"
Environment="PATH=${USER_HOME}/anaconda3/envs/tcm_env/bin:\$PATH"
ExecStart=${USER_HOME}/anaconda3/envs/tcm_env/bin/xinference launch \
    --model-engine transformers \
    --model-name biancang-tcm \
    --model-path ${BIANCANG_MODEL_DIR} \
    --size-in-billions 7 \
    --quantization int8 \
    --device cuda:0 \
    --port 9900 \
    --max-sequence-length 4096

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable xinference-tcm
sudo systemctl start xinference-tcm

# RAGFlow容器自启动
docker update --restart unless-stopped tcm-ragflow

# 环境验证
echo -e "\n===================== 中医诊疗系统部署完成 ==================="
echo "1. 模型服务状态："
systemctl status xinference-tcm --no-pager

echo -e "\n2. RAGFlow运行状态："
docker ps | grep tcm-ragflow

echo -e "\n3. 中医文档解析测试："
python ${USER_HOME}/scripts/tcm_parser.py /path/to/tcm_case.doc  # 替换为实际的中医病例文件路径

echo -e "\n===================== 开始使用 ==================="
echo "1. 访问RAGFlow界面：http://$(hostname -I | awk '{print $1}'):8000"
echo "2. 在知识库管理中上传中医病例文档，开始构建诊疗知识库"
echo "3. 测试问诊：输入『患者咳嗽痰黄，舌苔黄腻，脉滑数，开方建议』"

# 清理临时文件
rm -rf nvidia-driver.run cuda.run anaconda.sh
conda clean -a -y
