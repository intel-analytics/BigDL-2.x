#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_HOME
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

set -e

if [ $RUN_PART3 = 1 ]; then
echo "#15 start app test for pytorch face-generation"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation
sed -i '/get_ipython()/d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
sed -i '/plt./d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
${ANALYTICS_ZOO_HOME}/bin/spark-submit-python-with-zoo.sh \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
now=$(date "+%s")
time15=$((now-start))
echo "#15 pytorch face-generation time used:$time15 seconds"
fi
