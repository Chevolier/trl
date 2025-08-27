#!/bin/bash

echo "安装依赖..."

# 安装指定版本的 transformers
echo "安装 transformers..."
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775

# 安装其他必要的包
echo "安装其他依赖..."
pip install accelerate
pip install qwen-vl-utils
pip install 'vllm==0.10.0'
pip install pandas
# Install flashinfer with CUDA support
CUDA_HOME=/usr/local/cuda pip install flashinfer-python==0.2.5 --no-build-isolation
pip install -U flash-attn --no-build-isolation

echo "依赖安装完成。"