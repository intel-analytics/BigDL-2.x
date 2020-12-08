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

export ZOO_NUM_MKLTHREADS=4
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/mnist/main.py --dir analytics-zoo-data/data
unset ZOO_NUM_MKLTHREADS

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

export ZOO_NUM_MKLTHREADS=4
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/mnist/lenet_mnist.py --dir analytics-zoo-data/data
unset ZOO_NUM_MKLTHREADS

echo "start example for imagenet"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/imagenet-1k.tar.gz ]
then
    echo "imagenet-1k.tar.gz already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/imagenet-1k.tar.gz -P analytics-zoo-data/data
    tar zxf analytics-zoo-data/data/imagenet-1k.tar.gz -C analytics-zoo-data/data/
fi

export ZOO_NUM_MKLTHREADS=4
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/imagenet/main.py analytics-zoo-data/data/imagenet-small --batch-size 16 --epochs 1
unset ZOO_NUM_MKLTHREADS

echo "start example for resnet_finetune"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-data/data/dogs_cats.zip ]
then
    echo "dogs_cats.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/dogs_cats.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/dogs_cats.zip -d analytics-zoo-data/data/dogs_cats
fi

export ZOO_NUM_MKLTHREADS=4
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/resnet_finetune/resnet_finetune.py analytics-zoo-data/data/dogs_cats
unset ZOO_NUM_MKLTHREADS
