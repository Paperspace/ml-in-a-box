#!/usr/bin/env bash

sudo apt-get update
sudo apt-get -y upgrade
sudo apt-get -y install \
  gcc \
  make \
  pkg-config \
  apt-transport-https \
  ca-certificates

# Blacklist nouveau and rebuild kernel initramfs
echo "blacklist nouveau
options nouveau modeset=0" >> blacklist-nouveau.conf
sudo mv blacklist-nouveau.conf /etc/modprobe.d/blacklist-nouveau.conf
sudo update-initramfs -u
sudo reboot

# Install NVIDIA Linux toolkit 510.54
wget https://us.download.nvidia.com/XFree86/Linux-x86_64/510.54/NVIDIA-Linux-x86_64-510.54.run
chmod +x NVIDIA-Linux-x86_64-510.54.run
sudo bash ./NVIDIA-Linux-x86_64-510.54.run
rm NVIDIA-Linux-x86_64-510.54.run

# Install CUDA toolkit 11.3 Upgrade 1 for Ubuntu 20.04 LTS
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.3.1/local_installers/cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-3-local_11.3.1-465.19.01-1_amd64.deb
sudo apt-key add /var/cuda-repo-ubuntu2004-11-3-local/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda

# Install cuDNN 8.2.1 for CUDA 11.x
# authed download, so just ended up rsyncing it up to the machine from my local
# wget https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.2.1.32/11.3_06072021/Ubuntu20_04-x64/libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
sudo dpkg -i libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb
sudo apt-get update
sudo apt-get -y install libcudnn8
rm libcudnn8_8.2.1.32-1+cuda11.3_amd64.deb

# Install Docker Engine
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
sudo apt-get update
sudo apt-get -y install \
  docker-ce \
  docker-ce-cli \
  containerd.io

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
   && curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add - \
   && curl -s -L https://nvidia.github.io/nvidia-docker/${distribution}/nvidia-docker.list | \
   sudo tee /etc/apt/sources.list.d/nvidia-docker.list
curl -s -L https://nvidia.github.io/nvidia-container-runtime/experimental/${distribution}/nvidia-container-runtime.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-runtime.list
sudo apt-get update
sudo apt-get -y install nvidia-docker2
sudo systemctl restart docker

# Install pip
wget https://bootstrap.pypa.io/get-pip.py
python3 ./get-pip.py

# Install package dependencies
python3 -m pip install \
  numpy \
  pandas \
  matplotlib \
  jupyterlab \
  tabulate \
  future \
  scikit-learn

# Install PyTorch
python3 -m pip install \
  torch==1.10.2+cu113 \
  torchvision==0.11.3+cu113 \
  torchaudio==0.10.2+cu113 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install TensorFlow
python3 -m pip install tensorflow==2.5.0

# Install H2O
python3 -m pip install \
  -f https://h2o-release.s3.amazonaws.com/h2o/latest_stable_Py.html \
  h2o

# Install NVIDIA Rapids
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh
bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p $HOME/src/miniconda
export PATH=${PATH}:$HOME/src/miniconda/bin
conda create -y -n rapids-21.08 -c rapidsai -c nvidia -c conda-forge rapids-blazing=21.08 python=3.8 cudatoolkit=11.3
rm Miniconda3-py38_4.10.3-Linux-x86_64.sh


# Update PATH
export PATH=${PATH}:/home/paperspace/.local/bin
echo "export PATH=${PATH}:/home/paperspace/.local/bin" >> ~/.bashrc
echo "export PATH=${PATH}:/home/paperspace/src/miniconda/bin" >> ~/.bashrc
