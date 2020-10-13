#!/bin/bash

set -e

echo "start example for MNIST"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/MNIST ]
then
    echo "MNIST already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P analytics-zoo-data/data/MNIST/raw
fi
unzip -q analytics-zoo-data/data/MNIST/raw/mnist.zip -d analytics-zoo-data/data/MNIST/raw

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/mnist/main.py --dir analytics-zoo-data/data

