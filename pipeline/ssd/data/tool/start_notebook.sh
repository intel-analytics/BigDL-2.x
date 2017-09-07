#!/bin/bash

#setup pathes
SPARK_HOME=$HOME/spark-2.1.0-bin-hadoop2.7/
SSD_HOME=$HOME/code/analytics-zoo/pipeline/ssd
BigDL_HOME=$HOME/BigDL
MASTER="local[2]"

PYTHON_API_ZIP_PATH=${BigDL_HOME}/lib/bigdl-0.2.1-SNAPSHOT-python-api.zip
SSD_JAR_PATH=${SSD_HOME}/target/pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=../../notebook --ip=* --no-browser"

${SPARK_HOME}/bin/pyspark \
    --master ${MASTER} \
    --driver-cores 1  \
    --driver-memory 10g  \
    --total-executor-cores 3  \
    --executor-cores 1  \
    --executor-memory 20g \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
    --jars ${SSD_JAR_PATH} \
    --conf spark.driver.extraClassPath=${SSD_JAR_PATH} \
    --conf spark.executor.extraClassPath=pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar