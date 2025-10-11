# PyTorch + CUDA の公式イメージを使用
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-runtime

# 作業ディレクトリを設定
WORKDIR /home/kozaki/musicrec/BSARec

# 非対話モードで apt を実行
ENV DEBIAN_FRONTEND=noninteractive

# 必要なパッケージのインストール（curl, bzip2）
RUN apt-get update && \
    apt-get install -y curl bzip2 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Miniconda のダウンロードとインストール（curl を使用）
RUN rm -rf /opt/conda && \
    curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /miniconda.sh && \
    ls -l /miniconda.sh && \
    head -n 10 /miniconda.sh && \
    chmod +x /miniconda.sh && \
    bash -x /miniconda.sh -b -p /opt/conda && \
    rm /miniconda.sh


# conda のパスを通す
ENV PATH="/opt/conda/bin:$PATH"

# Conda 環境ファイルをコピーして環境を作成
COPY bsarec_env.yaml .
RUN conda update -n base -c defaults conda && \
    conda env create -f bsarec_env.yaml

# conda run を使って、指定環境下でコマンドを実行できるようにする
SHELL ["conda", "run", "-n", "bsarec", "/bin/bash", "-c"]

# プロジェクトファイルをコピー
COPY . .

