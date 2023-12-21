#!/bin/bash

# ==================================================================
# Module list
# ------------------------------------------------------------------
# python                        3.11.6           (apt)
# pip3                          23.3.1           (apt)
# cuda toolkit                  12.1.1           (apt)
# cudnn                         8.9.3            (apt)
# torch                         2.1.1            (pip)
# torchvision                   0.16.1           (pip)
# torchaudio                    2.1.1            (pip)
# tensorflow                    2.15.0           (pip)
# transformers                  4.35.2           (pip)
# datasets                      2.14.5           (pip)
# peft                          0.6.2            (pip)
# tokenizers                    0.13.3           (pip)
# accelerate                    0.24.1           (pip)
# diffusers                     0.21.4           (pip)
# safetensors                   0.4.0            (pip)
# jupyterlab                    3.6.5            (pip)
# bitsandbytes                  0.41.2           (pip)
# cloudpickle                   2.2.1            (pip)
# scikit-image                  0.21.0           (pip)
# scikit-learn                  1.3.0            (pip)
# matplotlib                    3.7.3            (pip)
# ipywidgets                    8.1.1            (pip)
# cython                        3.0.2            (pip)
# tqdm                          4.66.1           (pip)
# gdown                         4.7.1            (pip)
# xgboost                       1.7.6            (pip)
# pillow                        9.5.0            (pip)
# seaborn                       0.12.2           (pip)
# sqlalchemy                    2.0.21           (pip)
# spacy                         3.6.1            (pip)
# nltk                          3.8.1            (pip)
# boto3                         1.28.51          (pip)
# tabulate                      0.9.0            (pip)
# future                        0.18.3           (pip)
# jsonify                       0.5              (pip)
# opencv-python                 4.8.0.76         (pip)
# pyyaml                        5.4.1            (pip)
# sentence-transformers         2.2.2            (pip)
# wandb                         0.15.10          (pip)
# deepspeed                     0.10.3           (pip)
# cupy-cuda12x                  12.2.0           (pip)
# timm                          0.9.7            (pip)
# omegaconf                     2.3.0            (pip)
# scipy                         1.11.2           (pip)
# gradient                      2.0.6            (pip) 
# attrs                         23.1.0           (pip)
# default-jre                   latest           (apt)
# default-jdk                   latest           (apt)
# nodejs                        20.x latest      (apt)
# jupyter_contrib_nbextensions  0.7.0            (pip)
# jupyterlab-git                0.43.0           (pip)


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
        cifs-utils \
        zlib1g


# ==================================================================
# Git-lfs
# ------------------------------------------------------------------
    
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
    sudo $APT_INSTALL git-lfs


# ==================================================================
# Python
# ------------------------------------------------------------------

    # Based on https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa

    # Adding repository for python3.11
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

    # Grant access for pip to install in ~/.local
    sudo chmod -R a+rwx /home/paperspace/.local
    
    # Install pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3
    export PATH=$PATH:/home/paperspace/.local/bin


# ==================================================================
# Installing CUDA packages (CUDA Toolkit 12.1.1 & CUDNN 8.9.3)
# ------------------------------------------------------------------

    # Based on https://developer.nvidia.com/cuda-toolkit-archive
    # Based on https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

    sudo $APT_INSTALL nvidia-driver-535=535.129.03-0ubuntu1 \
        nvidia-fabricmanager-535=535.129.03-0ubuntu0.22.04.1 \
        cuda-drivers=535.129.03-1 \
        nvtop nvidia-modprobe=535.129.03-0ubuntu1 \
        nvidia-settings=535.129.03-0ubuntu1 \
        libnvidia-egl-wayland1 \
        cuda=12.1.1-1 \
        cuda-toolkit-12-config-common=12.1.105-1 \
        cuda-toolkit-config-common=12.1.105-1 \
        libnvidia-container-tools=1.13.2-1 \
        libnvidia-container1=1.13.2-1 \
        nvidia-container-toolkit=1.13.2-1 \
        nvidia-container-toolkit-base=1.13.2-1 \
        libnccl-dev=2.18.3-1+cuda12.1 \
        libnccl2=2.18.3-1+cuda12.1 \
        libxnvctrl0=535.129.03-0ubuntu1 \
        libcudnn8=8.9.3.28-1+cuda12.1 \
        libcudnn8-dev=8.9.3.28-1+cuda12.1 

    export PATH=$PATH:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_HOME=/usr/local/cuda


# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/
    
    $PIP_INSTALL torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1
    

# ==================================================================
# TensorFlow
# ------------------------------------------------------------------

    # Based on https://www.tensorflow.org/install/pip

    $PIP_INSTALL tensorflow==2.15.0


# ==================================================================
# Hugging Face Libraries
# ------------------------------------------------------------------
    
    $PIP_INSTALL transformers==4.35.2 \
        datasets==2.14.5 \
        peft==0.6.2 \
        tokenizers==0.13.3 \
        accelerate==0.24.1 \
        diffusers==0.21.4 \
        safetensors==0.4.0
        


# ==================================================================
# JupyterLab
# ------------------------------------------------------------------

    # Based on https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#pip

    $PIP_INSTALL jupyterlab==3.6.5


# ==================================================================
# Additional Python Packages
# ------------------------------------------------------------------

    $PIP_INSTALL \
        bitsandbytes==0.41.2 \
        cloudpickle==2.2.1 \
        scikit-image==0.21.0 \
        scikit-learn==1.3.0 \
        matplotlib==3.7.3 \
        ipywidgets==8.1.1 \
        cython==3.0.2 \
        tqdm==4.66.1 \
        gdown==4.7.1 \
        xgboost==1.7.6 \
        pillow==9.5.0 \
        seaborn==0.12.2 \
        sqlalchemy==2.0.21 \
        spacy==3.6.1 \
        nltk==3.8.1 \
        boto3==1.28.51 \
        tabulate==0.9.0 \
        future==0.18.3 \
        jsonify==0.5 \
        opencv-python==4.8.0.76 \
        pyyaml==5.4.1 \
        sentence-transformers==2.2.2 \
        wandb==0.15.10 \
        deepspeed==0.10.3 \
        cupy-cuda12x==12.2.0 \
        timm==0.9.7 \
        omegaconf==2.3.0 \
        scipy==1.11.2 \
        gradient==2.0.6

        $PIP_INSTALL attrs==23.1.0
       

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
    $PIP_INSTALL jupyter_contrib_nbextensions==0.7.0 jupyterlab-git==0.43.0
    jupyter contrib nbextension install --user


# ==================================================================
# Config & Cleanup
# ------------------------------------------------------------------

    echo "export PATH=${PATH}" >> ~/.bashrc
    echo "export LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" >> ~/.bashrc
    echo "export CUDA_HOME=${CUDA_HOME}" >> ~/.bashrc
