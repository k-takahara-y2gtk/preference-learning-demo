FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# 基本パッケージのインストール
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Pythonをデフォルトのpythonとして設定
RUN ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3

# 作業ディレクトリの設定
WORKDIR /workspace

# 必要なPythonパッケージのインストール
COPY requirements.txt /tmp/
RUN pip3 install --upgrade pip && \
    pip3 install -r /tmp/requirements.txt

# JCG-LLMプロジェクトのディレクトリ構造を保持
RUN mkdir -p /workspace/data/raw \
    /workspace/qa_benchmark/{scripts,datasets,evaluation,models} \
    /workspace/structure_db/{scripts,datasets,extraction,database}

# コンテナ起動時に実行されるコマンド
CMD ["/bin/bash"] 