#!/bin/bash

# ==================================================================
# Module list
# ------------------------------------------------------------------
# python                3.9.14           (apt)
# torch                 1.12.1           (pip)
# torchvision           0.13.1           (pip)
# torchaudio            0.12.1           (pip)
# tensorflow            2.9.2            (pip)
# jax                   0.3.17           (pip)
# transformers          4.21.3           (pip)
# datasets              2.4.0            (pip)
# jupyterlab            3.4.6            (pip)
# numpy                 1.23.2           (pip)
# scipy                 1.9.1            (pip)
# pandas                1.4.4            (pip)
# cloudpickle           2.1.0            (pip)
# scikit-image          0.19.3           (pip)
# scikit-learn          1.1.2            (pip)
# matplotlib            3.5.3            (pip)
# ipython               8.5.0            (pip)
# ipykernel             6.15.2           (pip)
# ipywidgets            8.0.2            (pip)
# cython                0.29.32          (pip)
# tqdm                  4.64.1           (pip)
# gdown                 4.5.1            (pip)
# xgboost               1.6.2            (pip)
# pillow                9.2.0            (pip)
# seaborn               0.12.0           (pip)
# sqlalchemy            1.4.40           (pip)
# spacy                 3.4.1            (pip)
# nltk                  3.7              (pip)
# boto3                 1.24.66          (pip)
# tabulate              0.8.10           (pip)
# future                0.18.2           (pip)
# gradient              2.0.6            (pip)
# jsonify               0.5              (pip)
# opencv-python         4.6.0.66         (pip)
# pyyaml                5.4.1            (pip)
# sentence-transformers 2.2.2            (pip)
# wandb                 0.13.4           (pip)
# nodejs                16.x latest      (apt)
# default-jre           latest           (apt)
# default-jdk           latest           (apt)


# ==================================================================
# Initial setup
# ------------------------------------------------------------------

    # Set ENV variables
    export APT_INSTALL="apt-get install -y --no-install-recommends"
    export PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
    export GIT_CLONE="git clone --depth 10"
    export DEBIAN_FRONTEND="noninteractive"

    # Update apt
    sudo apt update


# ==================================================================
# Tools
# ------------------------------------------------------------------

    # DEBIAN_FRONTEND=noninteractive \
    sudo $APT_INSTALL \
        gcc \
        make \
        pkg-config \
        apt-transport-https \
        build-essential \
        apt-utils \
        ca-certificates \
        wget \
        rsync \
        git \
        vim \
        mlocate \
        libssl-dev \
        curl \
        openssh-client \
        unzip \
        unrar \
        zip \
        awscli \
        csvkit \
        emacs \
        joe \
        jq \
        dialog \
        man-db \
        manpages \
        manpages-dev \
        manpages-posix \
        manpages-posix-dev \
        nano \
        iputils-ping \
        sudo \
        ffmpeg \
        libsm6 \
        libxext6 \
        libboost-all-dev \
        gnupg \
        cifs-utils

# ==================================================================
# Python
# ------------------------------------------------------------------

    # Based on https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa

    # Adding repository for python3.11
    DEBIAN_FRONTEND=noninteractive \
    sudo $APT_INSTALL software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y

    # Installing python3.11
    sudo $APT_INSTALL \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-distutils-extra

    # Add symlink so python and python3 commands use same python3.11 executable
    sudo ln -s /usr/bin/python3.11 /usr/local/bin/python3
    sudo ln -s /usr/bin/python3.11 /usr/local/bin/python

    # Installing pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
    export PATH=$PATH:/home/paperspace/.local/bin


# ==================================================================
# Installing CUDA packages (CUDA Toolkit 12.1.1 & CUDNN 8.9.4)
# ------------------------------------------------------------------

    # Based on https://developer.nvidia.com/cuda-toolkit-archive
    # Based on https://developer.nvidia.com/rdp/cudnn-archive

    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
    sudo sh cuda_12.1.1_530.30.02_linux.run --silent --toolkit
    export PATH=$PATH:/usr/local/cuda-12.1/bin
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64
    rm cuda_12.1.1_530.30.02_linux.run


    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    sudo $APT_INSTALL libcudnn8=8.9.4.*-1+cuda12.1
    sudo $APT_INSTALL libcudnn8-dev=8.9.4.*-1+cuda12.1


# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/
    
    # Stable version
    # $PIP_INSTALL torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
        
    # Nightly version
    $PIP_INSTALL --pre torch==2.2.0.dev20230921+cu121 torchvision==0.17.0.dev20230921+cu121 torchaudio==2.2.0.dev20230921+cu121 --index-url https://download.pytorch.org/whl/nightly/cu121


# ==================================================================
# JAX
# ------------------------------------------------------------------

    # Based on https://github.com/google/jax#pip-installation-gpu-cuda

    $PIP_INSTALL "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
        flax==0.7.4


# ==================================================================
# TensorFlow
# ------------------------------------------------------------------

    # Based on https://www.tensorflow.org/install/pip

    $PIP_INSTALL tensorflow==2.13.0


# ==================================================================
# Hugging Face
# ------------------------------------------------------------------
    
    # Based on https://huggingface.co/docs/transformers/installation
    # Based on https://huggingface.co/docs/datasets/installation

    $PIP_INSTALL transformers==4.33.2 \
        datasets==2.14.5 \
        peft==0.5.0 \
        tokenizers==0.13.3 \
        accelerate==0.23.0 \
        diffusers==0.21.3 \
        timm==0.9.7


# ==================================================================
# JupyterLab
# ------------------------------------------------------------------

    # Based on https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#pip

    $PIP_INSTALL jupyterlab==3.6.5


# ==================================================================
# Additional Python Packages
# ------------------------------------------------------------------

    $PIP_INSTALL \
        bitsandbytes==0.41.1 \
        numpy==1.24.3 \
        scipy==1.11.2 \
        pandas==2.1.0 \
        cloudpickle==2.2.1 \
        scikit-image==0.21.0 \
        scikit-learn==1.3.0 \
        matplotlib==3.7.3 \
        ipython==8.15.0 \
        ipykernel==6.25.2 \
        ipywidgets==8.1.1 \
        cython==3.0.2 \
        tqdm==4.66.1 \
        gdown==4.7.1 \
        xgboost==1.7.6 \
        pillow==10.0.1 \
        seaborn==0.12.2 \
        sqlalchemy==2.0.21 \
        spacy==3.6.1 \
        nltk==3.8.1 \
        boto3==1.28.51 \
        tabulate==0.9.0 \
        future==0.18.3 \
        gradient==2.0.6 \
        jsonify==0.5 \
        opencv-python==4.8.0.76 \
        pyyaml==5.4.1 \
        sentence-transformers==2.2.2 \
        wandb==0.15.10 \
        deepspeed==0.10.3 \
        cupy-cuda12x==12.2.0
       
        


# ==================================================================
# Installing JRE and JDK
# ------------------------------------------------------------------

    sudo $APT_INSTALL default-jre
    sudo $APT_INSTALL default-jdk


# ==================================================================
# CMake
# ------------------------------------------------------------------

    sudo $GIT_CLONE https://github.com/Kitware/CMake ~/cmake
    cd ~/cmake
    sudo ./bootstrap
    sudo make -j"$(nproc)" install


# ==================================================================
# Node.js and Jupyter Notebook Extensions
# ------------------------------------------------------------------
    
    # Node.js
    sudo mkdir -p /etc/apt/keyrings
    curl -fsSL https://deb.nodesource.com/gpgkey/nodesource-repo.gpg.key | sudo gpg --dearmor -o /etc/apt/keyrings/nodesource.gpg
    export NODE_MAJOR=20
    echo "deb [signed-by=/etc/apt/keyrings/nodesource.gpg] https://deb.nodesource.com/node_$NODE_MAJOR.x nodistro main" | sudo tee /etc/apt/sources.list.d/nodesource.list
    sudo apt-get update
    sudo $APT_INSTALL nodejs

    # Jupyter Notebook Extensions
    $PIP_INSTALL jupyter_contrib_nbextensions jupyterlab-git
    jupyter contrib nbextension install --user


# ==================================================================
# Config & Cleanup
# ------------------------------------------------------------------

    echo "export PATH=${PATH}" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ~/.bashrc
