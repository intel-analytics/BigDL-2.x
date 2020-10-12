#!/bin/bash

# export SPARK_HOME=$SPARK_HOME
# export MASTER=local[4]
# export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
# export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
# export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
# export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
# export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
# export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH

set -e

echo "#1 start example for MNIST"
#timer
start=$(date "+%s")
if [ -f ${ANALYTICS_ZOO_ROOT}/data/mnist.zip ]
then
    echo "mnist.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P analytics-zoo-data/data
fi
unzip -q analytics-zoo-data/data/mnist.zip -d analytics-zoo-data/data

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/mnist/main.py --dir analytics-zoo-data/data

