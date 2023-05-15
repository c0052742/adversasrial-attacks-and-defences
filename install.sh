#!/bin/bash

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install adversarial-robustness-toolbox==1.13.1