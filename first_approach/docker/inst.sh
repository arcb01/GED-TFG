#!/bin/bash
cd mmf/

python -m pip install --editable .
python -m pip uninstall -y torch torchvision torchaudio
python -m pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install pytorch-lightning==1.6.0
python -m pip install future
python -m pip install numpy==1.20.3 --upgrade
python -m pip install pytest

