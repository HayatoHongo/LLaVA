FROM ubuntu:22.04

# 基本ツール
RUN apt-get update && apt-get install -y \
    git curl wget vim build-essential python3.10 python3.10-venv python3.10-dev \
    && rm -rf /var/lib/apt/lists/*

# pip セットアップ
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# 仮想環境
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 依存関係（requirements.txt を使う場合）
WORKDIR /workspace
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# LLaVA 本体
COPY . /workspace
RUN pip install -e . && pip install -e .[train]

EXPOSE 8888
