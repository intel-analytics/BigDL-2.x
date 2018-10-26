#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH

set -e

echo "#6 start example test for tensorflow"
#timer
start=$(date "+%s")
    
echo "start example test for tensorflow distributed_training"
if [ ! -d analytics-zoo-model ]
then
    mkdir analytics-zoo-model
fi
sed "s%/tmp%analytics-zoo-model%g"
if [ -d analytics-zoo-model/model ]
then
    echo "analytics-zoo-model/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
    git clone https://github.com/tensorflow/models/ analytics-zoo-model
    export PYTHONPATH=$PYTHONPATH:`pwd`/analytics-zoo-model/model/research:`pwd`/analytics-zoo-models/model/research/slim
 fi
${SPARK_HOME}/bin/spark-submit \
    --master ${master} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/train_lenet.py \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/evaluate_lenet.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/train_lenet.py \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/evaluate_lenet.py \
    
now=$(date "+%s")
time6=$((now-start))
echo "#6 tensorflow time used:$time6 seconds"
