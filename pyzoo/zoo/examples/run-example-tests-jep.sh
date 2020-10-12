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
if [ -f analytics-zoo-data/data/mnist ]
then
    echo "analytics-zoo-data/data/mnist already exists" 
else
    wget -nv http://10.239.45.10:8081/repository/raw/analytics-zoo-data/mnist -P analytics-zoo-data/data
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/mnist/main.py --dir analytics-zoo-data/data/mnist

