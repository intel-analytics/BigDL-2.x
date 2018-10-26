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
if [ ! -d analytics-zoo-tensorflow-models ]
then
    mkdir analytics-zoo-tensorflow-models
    mkdir -p analytics-zoo-tensorflow-models/mnist
    mkdir -p analytics-zoo-tensorflow-models/az_lenet
    mkdir -p analytics-zoo-tensorflow-models/lenet
fi

sed "s/end_trigger=MaxEpoch(5)/end_trigger=MaxEpoch(2)/g;s%/tmp%analytics-zoo-tensorflow-models%g;s%models/slim%slim%g"
if [ -d analytics-zoo-tensorflow-models/slim ]
then
    echo "analytics-zoo-tensorflow-models/slim already exists."
else
    echo "Downloading research/slim"
   
   wget $FTP_URI/analytics-zoo-tensorflow-models/models/research/slim.tar.gz -P analytics-zoo-tensorflow-models
   tar -zxvf analytics-zoo-tensorflow-models/slim.tar.gz -C analytics-zoo-tensorflow-models
   
   echo "Finished downloading research/slim"
   export PYTHONPATH=`pwd`/analytics-zoo-tensorflow-models/slim:$PYTHONPATH
 fi

echo "start example test for tensorflow distributed_training train_lenet 1"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/train_lenet.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/train_lenet.py \

sed "s%/tmp%analytics-zoo-tensorflow-models%g;s%models/slim%slim%g"
if [ -d analytics-zoo-tensorflow-models/slim ]
then
    echo "analytics-zoo-tensorflow-models/slim already exists."
else
    echo "Downloading research/slim"
   
   wget $FTP_URI/analytics-zoo-tensorflow-models/models/research/slim.tar.gz -P analytics-zoo-tensorflow-models
   tar -zxvf analytics-zoo-tensorflow-models/slim.tar.gz -C analytics-zoo-tensorflow-models
   
   echo "Finished downloading research/slim"
   export PYTHONPATH=`pwd`/analytics-zoo-tensorflow-models/slim:$PYTHONPATH
 fi

echo "start example test for tensorflow distributed_training evaluate_lenet 2"
${SPARK_HOME}/bin/spark-submit \
    --master ${MASTER} \
    --driver-memory 200g \
    --executor-memory 200g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/evaluate_lenet.py \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/evaluate_lenet.py \
    
now=$(date "+%s")
time6=$((now-start))
echo "#6 tensorflow time used:$time6 seconds"
