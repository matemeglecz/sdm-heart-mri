FROM nvidia/cuda:11.7.1-runtime-ubuntu20.04

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH


# install pip
RUN apt-get update && apt-get install -y python3-pip

# install conda
RUN conda install mpi4py
COPY requirements2.txt requirements2.txt
RUN pip install -r requirements2.txt
WORKDIR /app

RUN pip list
RUN apt install nano
