# FROM registry.eps.surrey.ac.uk/tu/cuda:11.3.0-cudnn8-devel-ubuntu20.04
FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04
ENV DEBIAN_FRONTEND noninteractive

# rm nvidia repo if necessary
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list

# base packages
RUN apt-get update -y && apt-get install -y \
    ca-certificates \
    software-properties-common \
    sudo \
    git \
    wget \
    curl \
    bzip2 \
    libgtk2.0-dev \
    libopenblas-dev \
    g++ \
    cmake \
    sshfs \
    && apt-get -y purge manpages-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# sshfs - only needed if mount remote storage
RUN echo "user_allow_other" >> /etc/fuse.conf
RUN mkdir /mnt/fast && chmod 777 /mnt/fast


# imagemagik and opencv dependencies
RUN apt-get update -y && apt-get install -y \
    libmagickwand-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# librosa and pydub dependencies
RUN apt-get update -y && apt-get install -y libsndfile1-dev ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# custom user, useful if mount remote storage
ARG NB_USER="appuser"
ARG NB_UID="123456"
ARG NB_GID="1234567"

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV NB_UID=${NB_UID} \
    NB_GID=${NB_GID} \
    HOME="/home/${NB_USER}"
RUN useradd -l -m -s /bin/bash -N -u "${NB_UID}" "${NB_USER}" && \
    mkdir -p "${HOME}" && \
    chown "${NB_USER}:${NB_GID}" "${HOME}"
RUN adduser ${NB_USER} sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

USER ${NB_UID}

# conda
WORKDIR "${HOME}"
ARG CONDA_FILE="Miniconda3-py38_4.9.2-Linux-x86_64.sh"
ARG PYTHON_VER="3.8.5"
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=${HOME}/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/${CONDA_FILE} \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==${PYTHON_VER} pip==20.3 numpy==1.19.2 \
 && conda clean -ya

# CUDA and pytorch
RUN conda install -y pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 torchaudio=0.11.0 -c pytorch -c conda-forge \
    && conda clean -ya 

# jupyter lab
RUN conda install -y -c conda-forge jupyterlab \
    && conda clean -ya

# lmdb
RUN conda install -y lmdb python-lmdb \
    && conda clean -ya

# imagenet c with updated dependencies
RUN git clone https://github.com/TuBui/robustness.git && cd robustness/ImageNet-C/imagenet_c/ && pip install -e . 

# pip packages
ADD ./requirements.txt ./
ADD ./requirements2.txt ./
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r requirements2.txt
