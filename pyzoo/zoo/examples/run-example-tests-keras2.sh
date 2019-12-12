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

echo "#1 start example test for attention"
#timer
start=$(date "+%s")
sed "s/hidden_size=128/hidden_size=32/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/transformer.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/tmp.py

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 100g \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/tmp.py \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/transformer.py

now=$(date "+%s")
time1=$((now-start))
echo "#1 attention time used:$time1 seconds"