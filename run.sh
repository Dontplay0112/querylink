#!/bin/bash

export CUDA_VISIBLE_DEVICES="5,6,7"
export PYTHONDONTWRITEBYTECODE=1
find . -type d -name "__pycache__" -exec rm -r {} +

# relative path to src/
LOCOMO_CONFIG_PATH="config/locomo.toml"
LONGMEMEVAL_CONFIG_PATH="config/longmemeval.toml"

# read arguments
if [ "$1" == "locomo" ]; then
    CONFIG_PATH=$LOCOMO_CONFIG_PATH
elif [ "$1" == "longmemeval" ]; then
    CONFIG_PATH=$LONGMEMEVAL_CONFIG_PATH
else
    echo "Usage: $0 [locomo|longmemeval]"
    exit 1
fi

cd src
# uv run prepare.py
uv run main.py $CONFIG_PATH