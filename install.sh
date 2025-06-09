#!/bin/bash
# Install PyTorch first
pip install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.7.0 torchaudio==2.7.0 torchvision==0.22.0

# this is needed as torch geometircs depends on already having torch installed

# Install everything else
pip install -r requirements.txt