#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist

set -e

echo "#1 start example test for attention"
#timer
start=$(date "+%s")
sed "s/max_features = 20000/max_features = 100/g;s/max_len = 200/max_len = 10/g;s/hidden_size=128/hidden_size=8/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/transformer.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/tmp.py

${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
    --conf spark.executor.extraJavaOptions="-Xss512m" \
    --conf spark.driver.extraJavaOptions="-Xss512m" \
    --master ${MASTER} \
    --driver-memory 20g \
    --executor-memory 100g \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/tmp.py

now=$(date "+%s")
time1=$((now-start))
echo "#1 attention time used:$time1 seconds"
