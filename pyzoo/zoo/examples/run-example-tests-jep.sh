#!/bin/bash

set -e

echo "start example for MNIST"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/MNIST ]
then
    echo "MNIST already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/mnist/main.py --dir analytics-zoo-data/data

echo "start example for orca MNIST"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/MNIST ]
then
    echo "MNIST already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/train-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-labels-idx1-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
    wget -nv $FTP_URI/analytics-zoo-data/mnist/t10k-images-idx3-ubyte.gz -P analytics-zoo-data/data/MNIST/raw
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/mnist/lenet_mnist.py --dir analytics-zoo-data/data

now=${data "+%s")
time1=$((now-start))
echo "#1 mnist example time used:$time1 seconds"

echo "start example for Cifar10"
#timer
start=$(date "+%s")
if [ -d ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/data ]
then
    echo "Cifar10 already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10
    unzip ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.zip
fi
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.py

now=${data "+%s")
time2=$((now-start))
echo "#2 cifar10 example time used:$time2 seconds"
