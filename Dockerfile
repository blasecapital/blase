FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

RUN apt-get update && apt-get install -y \
    wget \
    tzdata \
    sqlite3 \
    libsqlite3-dev \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py39_23.11.0-1-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py39_23.11.0-1-Linux-x86_64.sh

ENV PATH="/opt/conda/bin:$PATH"

RUN conda install -y pip && conda clean --all -y

RUN conda config --add channels conda-forge

RUN pip install --no-cache-dir tensorflow-gpu==2.5.0
RUN pip install --no-cache-dir torch==1.10.1+cu111 \
    torchvision==0.11.2+cu111 \
    torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN conda install -y \
    transformers \
    xgboost \
    lightgbm \
    jupyter \
    pyarrow \
    duckdb \
    polars \
    orjson \
    librosa \
    joblib \
    matplotlib \
    mplfinance \
    pickleshare \
    pillow \
    python-dateutil \
    python-dotenv \
    pytz \
    scikit-learn \
    scipy \
    threadpoolctl \
    tqdm \
    lime \
    pytest \
    "numpy=1.19.5" \
    "pandas=1.1.5" && \
    conda clean --all -y

ENV CONDA_DEFAULT_ENV=base
ENV PATH="/opt/conda/bin:$PATH"

RUN python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
RUN python -c "import torch; print('PyTorch:', torch.__version__)"
RUN python -c "import numpy as np; print('NumPy:', np.__version__)"
RUN python -c "import pandas as pd; print('Pandas:', pd.__version__)"

WORKDIR /workspace
