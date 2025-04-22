#!/bin/bash

pip install -r requirements-base.txt

if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Installing PyTorch packages for Mac..."
    pip install -r requirements-mac.txt
else
    echo "Installing PyTorch packages with CUDA support..."
    pip install --no-cache-dir -r requirements-cuda.txt
fi

pip install -e .