#!/bin/bash

cd ../latex2sympy
pip install -U pip setuptools

pip install torch 
pip install xformers==0.0.27.post2 --index-url https://download.pytorch.org/whl/cu121
pip install torchvision torchaudio

pip install -e .
cd ..
pip install -r requirements.txt 
pip install vllm==0.5.4 --no-build-isolation
pip install transformers==4.48.2
# pip install transformers==4.42.3