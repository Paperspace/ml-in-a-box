#!/bin/bash

# ==================================================================
# Module list
# ------------------------------------------------------------------
# python                        3.11.5           (apt)
# torch                         2.1.0            (pip)
# torchvision                   0.16.0           (pip)
# torchaudio                    2.1.0            (pip)
# tensorflow                    2.13.0           (pip)
# transformers                  4.33.2           (pip)
# datasets                      2.14.5           (pip)
# peft                          0.5.0            (pip)
# tokenizers                    0.14.1           (pip)
# accelerate                    0.23.0           (pip)
# diffusers                     0.21.4           (pip)
# timm                          0.9.7            (pip)
# jupyterlab                    3.6.5            (pip)
# bitsandbytes                  0.41.1           (pip)
# numpy                         1.24.3           (pip)
# scipy                         1.11.2           (pip)
# pandas                        2.1.0            (pip)
# cloudpickle                   2.2.1            (pip)
# scikit-image                  0.21.0           (pip)
# scikit-learn                  1.3.0            (pip)
# matplotlib                    3.7.3            (pip)
# ipython                       8.15.0           (pip)
# ipykernel                     6.25.2           (pip)
# ipywidgets                    8.1.1            (pip)
# cython                        3.0.2            (pip)
# tqdm                          4.66.1           (pip)
# gdown                         4.7.1            (pip)
# xgboost                       1.7.6            (pip)
# pillow                        10.0.1           (pip)
# seaborn                       0.12.2           (pip)
# sqlalchemy                    2.0.21           (pip)
# spacy                         3.6.1            (pip)
# nltk                          3.8.1            (pip)
# boto3                         1.28.51          (pip)
# tabulate                      0.9.0            (pip)
# future                        0.18.3           (pip)
# gradient                      2.0.6            (pip)
# jsonify                       0.5              (pip)
# opencv-python                 4.8.0.76         (pip)
# pyyaml                        5.4.1            (pip)
# sentence-transformers         2.2.2            (pip)
# wandb                         0.15.10          (pip)
# deepspeed                     0.10.3           (pip)
# cupy-cuda12x                  12.2.0           (pip)
# safetensors                   0.4.0            (pip)
# omegaconf                     2.3.0            (pip) 
# jupyter_contrib_nbextensions  0.7.0            (pip)
# jupyterlab-git                0.43.0           (pip)
# nodejs                        20.x latest      (apt)
# default-jre                   latest           (apt)
# default-jdk                   latest           (apt)


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
    
    curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash \
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
# Installing CUDA packages (CUDA Toolkit 12.1.1 & CUDNN 8.9.4)
# ------------------------------------------------------------------

    # Based on https://developer.nvidia.com/cuda-toolkit-archive
    # Based on https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html

    wget https://developer.download.nvidia.com/compute/cuda/12.1.1/local_installers/cuda_12.1.1_530.30.02_linux.run
    sudo sh cuda_12.1.1_530.30.02_linux.run --silent --toolkit
    export PATH=$PATH:/usr/local/cuda/bin
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_HOME=/usr/local/cuda
    rm cuda_12.1.1_530.30.02_linux.run

    # When cudnn-local*.deb file is on machine:
    sudo dpkg -i cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb
    sudo cp /var/cudnn-local-repo-*/cudnn-local-*-keyring.gpg /usr/share/keyrings/
    sudo apt-get update
    sudo $APT_INSTALL libcudnn8=8.9.5.29-1+cuda12.2
    sudo $APT_INSTALL libcudnn8-dev=8.9.5.29-1+cuda12.2
    rm cudnn-local-repo-ubuntu2204-8.9.5.29_1.0-1_amd64.deb


# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/
    
    $PIP_INSTALL torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
    

# ==================================================================
# JAX
# ------------------------------------------------------------------

    # Based on https://github.com/google/jax#pip-installation-gpu-cuda

    # $PIP_INSTALL "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \
    #     flax==0.7.4


# ==================================================================
# TensorFlow
# ------------------------------------------------------------------

    # Based on https://www.tensorflow.org/install/pip

    $PIP_INSTALL tensorflow==2.13.0


# ==================================================================
# Hugging Face Libraries
# ------------------------------------------------------------------
    
    $PIP_INSTALL transformers==4.33.3 \
        datasets==2.14.5 \
        peft==0.5.0 \
        tokenizers==0.13.3 \
        accelerate==0.23.0 \
        diffusers==0.21.4 \
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
        pillow==9.5.0 \
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
        cupy-cuda12x==12.2.0 \
        safetensors==0.4.0 \
        omegaconf==2.3.0 \
        attrs==23.1.0
       

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
