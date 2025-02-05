#!/bin/bash

cd ../latex2sympy
pip install -e .
cd ..
pip install -r requirements_test.txt 
pip install vllm==0.5.1 --no-build-isolation
pip install transformers==4.42.3