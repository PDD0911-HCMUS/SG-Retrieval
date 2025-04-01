FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Tùy chọn: chuyển về noninteractive để tránh lỗi khi cài gói
ENV DEBIAN_FRONTEND=noninteractive

# Cài Python 3.10 và các thư viện hệ thống cơ bản
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-venv python3-pip \
    git wget curl vim \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Tạo alias cho python
# RUN ln -s /usr/bin/python3.10 /usr/bin/python && \
#     ln -s /usr/bin/pip3 /usr/bin/pip

RUN python3 --version
RUN pip3 --version

# Cập nhật pip, cài torch + các thư viện cần thiết
RUN pip install --upgrade pip setuptools

# Cài PyTorch (tương thích CUDA 11.8)
RUN pip install torch==2.2.2 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# (Optional) Thêm HuggingFace Transformers, tqdm, matplotlib,...
RUN pip install transformers tqdm matplotlib

# Tạo workspace
WORKDIR /workspace
# COPY requirements.txt .
# RUN pip install -r requirements.txt
