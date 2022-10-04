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
# nodejs                16.x latest      (apt)
# default-jre           latest           (apt)
# default-jdk           latest           (apt)


# ==================================================================
# Initial setup
# ------------------------------------------------------------------

    # Set ENV variables
    export APT_INSTALL="apt-get install -y --no-install-recommends"
    export PIP_INSTALL="python -m pip --no-cache-dir install --upgrade"
    # export CONDA_INSTALL="conda install -y"
    export GIT_CLONE="git clone --depth 10"

    # Update apt
    sudo apt update


# ==================================================================
# Tools
# ------------------------------------------------------------------

    DEBIAN_FRONTEND=noninteractive \
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
        libboost-all-dev


# ==================================================================
# Conda
# ------------------------------------------------------------------

    #Based on https://docs.anaconda.com/anaconda/install/linux/

    # sudo $APT_INSTALL libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
    # sudo wget https://repo.anaconda.com/archive/Anaconda3-2022.05-Linux-x86_64.sh
    # sudo bash ~/Anaconda3-2022.05-Linux-x86_64.sh -b -p $HOME/anaconda3
    
    # sudo chown -R $USER:$USER $HOME/anaconda3
    # sudo chmod -R +x $HOME/anaconda3
    
    # source $HOME/anaconda3/bin/activate
    # conda init bash
    # conda deactivate
    
    # export PATH=$HOME/anaconda3/bin:${PATH}
    
    # $PIP_INSTALL pip
    
    # rm -f ~/Anaconda3-2022.05-Linux-x86_64.sh


# ==================================================================
# Python
# ------------------------------------------------------------------

    #Based on https://launchpad.net/~deadsnakes/+archive/ubuntu/ppa

    # Adding repository for python3.9
    DEBIAN_FRONTEND=noninteractive \
    sudo $APT_INSTALL software-properties-common
    sudo add-apt-repository ppa:deadsnakes/ppa -y

    # Installing python3.9
    DEBIAN_FRONTEND=noninteractive sudo $APT_INSTALL \
    python3.10 \
    python3.10-distutils
    # python3.10-dev \
    # python3.10-distutils-extra

    # Add symlink so python and python3 commands use same python3.10 executable
    sudo ln -s /usr/bin/python3.10 /usr/local/bin/python3
    sudo ln -s /usr/bin/python3.10 /usr/local/bin/python
    sudo ln -sf /usr/bin/python3.10 /usr/bin/python3

    # Installing pip
    curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
    export PATH=$PATH:/home/paperspace/.local/bin

    # export PATH=$PATH:/home/paperspace/.local/bin
    # wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py 
    # python ~/get-pip.py
    # rm ~/get-pip.py
    
    # sudo ln -sf /home/paperspace/.local/bin/pip /usr/bin/pip
    # sudo ln -sf /home/paperspace/.local/bin/pip3 /usr/bin/pip3
    # sudo ln -sf /home/paperspace/.local/bin/pip3.10 /usr/bin/pip3.10

    # Installing pip
    # DEBIAN_FRONTEND=noninteractive \
    # sudo $APT_INSTALL python3-pip

    # wget -O ~/get-pip.py https://bootstrap.pypa.io/get-pip.py 
    # python ~/get-pip.py
    # rm ~/get-pip.py

# ==================================================================
# Installing CUDA packages (CUDA Toolkit 11.7.1 & CUDNN 8.5.0)
# ------------------------------------------------------------------

    # $CONDA_INSTALL -c nvidia/label/cuda-11.7.1 cuda

    # wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    # sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600

    # wget https://developer.download.nvidia.com/compute/cuda/11.3.0/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb
    # sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.0-465.19.01-1_amd64.deb

    # sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
    # sudo apt-get update

    # sudo apt-get -y install cuda
    # export PATH=/usr/local/cuda-11.3/bin${PATH:+:${PATH}}
    # export LD_LIBRARY_PATH=/usr/local/cuda-11.3/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}


    # sudo wget -O /etc/apt/preferences.d/cuda-repository-pin-600 https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    # sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub
    # sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"

    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
    sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
    sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
    sudo add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /"
    sudo apt-get update
    sudo apt-get install cuda #New
    sudo apt-get install libcudnn8=8.5.0.*-1+cuda11.7
    sudo apt-get install libcudnn8-dev=8.5.0.*-1+cuda11.7


# ==================================================================
# PyTorch
# ------------------------------------------------------------------

    # Based on https://pytorch.org/get-started/locally/

    $PIP_INSTALL torch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
        

# ==================================================================
# JAX
# ------------------------------------------------------------------

    # Based on https://github.com/google/jax#pip-installation-gpu-cuda

    $PIP_INSTALL "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html


# ==================================================================
# TensorFlow
# ------------------------------------------------------------------

    # Based on https://www.tensorflow.org/install/pip

    # export LD_LIBRARY_PATH=${HOME}/anaconda3/lib
    $PIP_INSTALL tensorflow==2.9.2


# ==================================================================
# Hugging Face
# ------------------------------------------------------------------
    
    # Based on https://huggingface.co/docs/transformers/installation
    # Based on https://huggingface.co/docs/datasets/installation

    $PIP_INSTALL transformers==4.21.3 datasets==2.4.0


# ==================================================================
# JupyterLab
# ------------------------------------------------------------------

    # Based on https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html#pip

    $PIP_INSTALL jupyterlab==3.4.6


# ==================================================================
# Additional Python Packages
# ------------------------------------------------------------------

    $PIP_INSTALL \
        numpy==1.23.2 \
        scipy==1.9.1 \
        pandas==1.4.4 \
        cloudpickle==2.1.0 \
        scikit-image==0.19.3 \
        scikit-learn==1.1.2 \
        matplotlib==3.5.3 \
        ipython==8.5.0 \
        ipykernel==6.15.2 \
        ipywidgets==8.0.2 \
        cython==0.29.32 \
        tqdm==4.64.1 \
        gdown==4.5.1 \
        xgboost==1.6.2 \
        pillow==9.2.0 \
        seaborn==0.12.0 \
        sqlalchemy==1.4.40 \
        spacy==3.4.1 \
        nltk==3.7 \
        boto3==1.24.66 \
        tabulate==0.8.10 \
        future==0.18.2 \
        gradient==2.0.6 \
        jsonify==0.5 \
        opencv-python==4.6.0.66 \
        pyyaml==5.4.1 \
        sentence-transformers==2.2.2


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

    sudo curl -sL https://deb.nodesource.com/setup_16.x | sudo bash
    sudo $APT_INSTALL nodejs
    $PIP_INSTALL jupyter_contrib_nbextensions jupyterlab-git && \
    DEBIAN_FRONTEND=noninteractive jupyter contrib nbextension install --sys-prefix


# ==================================================================
# Config & Cleanup
# ------------------------------------------------------------------

    # rm $HOME/anaconda3/lib/libtinfo.so.6
    # rm $HOME/anaconda3/lib/libncursesw.so.6

    echo "export PATH=${PATH}" >> ~/.bashrc
    # echo "export LD_LIBRARY_PATH=${HOME}/anaconda3/lib" >> ~/.bashrc

    echo "export PATH=${PATH}" >> ~/.profile
    # echo "export LD_LIBRARY_PATH=${HOME}/anaconda3/lib" >> ~/.profile

    echo "export PATH=${PATH}" >> ~/.bash_profile
    # echo "export LD_LIBRARY_PATH=${HOME}/anaconda3/lib" >> ~/.bash_profile


