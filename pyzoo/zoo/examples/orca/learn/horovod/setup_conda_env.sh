#!/usr/bin/env bash

if [[ ! -f ${EXAMPLE_CONDA_NAME} ]]; then
    EXAMPLE_CONDA_NAME=orca-horovod-example
fi

conda create -y --name ${EXAMPLE_CONDA_NAME} python==3.6.7
conda activate ${EXAMPLE_CONDA_NAME}
conda install -y pytorch==1.4.0 torchvision==0.5.0 cpuonly -c pytorch
conda install -y cmake==3.14.0
conda install -y gxx_linux-64==7.3.0
HOROVOD_WITH_PYTORCH=1; HOROVOD_WITH_GLOO=1; pip install --no-cache-dir horovod==0.19.1

if [[ ! -f ${ANALYTICS_ZOO_WHL} ]]; then
    ANALYTICS_ZOO_WHL=analytics-zoo
fi
pip install ${ANALYTICS_ZOO_WHL}[ray]
