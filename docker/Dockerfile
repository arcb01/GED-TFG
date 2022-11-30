# Load pytorch img
FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
COPY . /tfg_code
WORKDIR /tfg_code

# Fixes NVIDIA issues
RUN apt-key del 7fa2af80
ADD https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb .
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Update and install git
RUN apt-get update && apt-get install -y git && apt-get install -y wget

# Clone mmf repo
RUN git clone https://github.com/facebookresearch/mmf.git 

# Install anaconda
RUN cd $HOME && wget https://repo.anaconda.com/miniconda/Miniconda3-py38_4.10.3-Linux-x86_64.sh && bash Miniconda3-py38_4.10.3-Linux-x86_64.sh -b -p $HOME/miniconda
COPY environment.yml .
COPY inst.sh .
RUN chmod +x inst.sh

# Create env
RUN $HOME/miniconda/bin/conda init bash
RUN $HOME/miniconda/bin/conda env create -f environment.yml
# Activate conda environment on startup
RUN echo "export PATH=$HOME/miniconda/bin:$PATH" >> $HOME/.bashrc
RUN echo "conda init bash" >> $HOME/.bashrc
RUN echo "conda activate mmf_env" >> $HOME/.bashrc
SHELL ["/bin/bash", "-c"]


