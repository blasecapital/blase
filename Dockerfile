FROM nvidia/cuda:12.5.1-cudnn-devel-ubuntu20.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
ENV PATH="/opt/conda/bin:$PATH"

# System dependencies
RUN apt-get update && apt-get install -y \
    wget \
    tzdata \
    sqlite3 \
    libsqlite3-dev \
    && ln -fs /usr/share/zoneinfo/Etc/UTC /etc/localtime && \
    echo "Etc/UTC" > /etc/timezone && \
    dpkg-reconfigure -f noninteractive tzdata

# Install Miniconda (Linux x86_64 version)
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py39_23.11.0-1-Linux-x86_64.sh && \
    bash Miniconda3-py39_23.11.0-1-Linux-x86_64.sh -b -p /opt/conda && \
    rm Miniconda3-py39_23.11.0-1-Linux-x86_64.sh

# Configure conda
RUN conda config --add channels conda-forge && \
    conda config --remove channels defaults && \
    conda install -y python=3.9 pip && \
    conda clean --all -y

RUN pip install --no-cache-dir \
    https://storage.googleapis.com/tensorflow/versions/2.19.0/tensorflow-2.19.0-cp39-cp39-manylinux_2_17_x86_64.manylinux2014_x86_64.whl

RUN conda install -y \
    pyarrow \
    duckdb \
    polars \
    orjson \
    librosa \
    joblib \
    matplotlib \
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
    more-itertools \
    numpy=2.0.2 \
    pandas && \
    conda clean --all -y

ENV CONDA_DEFAULT_ENV=base
ENV PATH="/opt/conda/bin:$PATH"

RUN python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
RUN python -c "import numpy as np; print('NumPy:', np.__version__)"
RUN python -c "import pandas as pd; print('Pandas:', pd.__version__)"

# -------- Rust + maturin setup --------
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    pkg-config \
    libssl-dev \
    && curl https://sh.rustup.rs -sSf | bash -s -- -y --default-toolchain 1.87.0 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.cargo/bin:${PATH}"

# Install maturin globally (used to build Rust Python bindings)
RUN pip install maturin

RUN python -c "import tensorflow as tf; print('TensorFlow:', tf.__version__)"
RUN python -c "import numpy as np; print('NumPy:', np.__version__)"
RUN python -c "import pandas as pd; print('Pandas:', pd.__version__)"
# -------- end Rust setup ---------

WORKDIR /workspace