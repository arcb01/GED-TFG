#!/bin/bash
cd mmf/

pip install --editable .
pip uninstall -y torch torchvision torchaudio
pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==1.6.0
pip install future
pip install numpy==1.20.3 --upgrade
pip install pytest

