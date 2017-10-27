#!/bin/bash

#setup pathes
SPARK_HOME=$HOME/spark-2.1.0-bin-hadoop2.7/
SSD_HOME=$HOME/code/analytics-zoo/pipeline/ssd
VISION_HOME=$SSD_HOME/../../transform/vision
BigDL_HOME=$HOME/BigDL
MASTER="local[2]"

PYTHON_API_ZIP_PATH=${VISION_HOME}/target/vision-0.1-SNAPSHOT-python-api.zip,${SSD_HOME}/target/pipeline-0.1-SNAPSHOT-python-api.zip,${BigDL_HOME}/lib/bigdl-0.3.0-SNAPSHOT-python-api.zip
SSD_JAR_PATH=${SSD_HOME}/target/pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=../../notebook --ip=* --no-browser"

${SPARK_HOME}/bin/pyspark \
    --master ${MASTER} \
    --driver-cores 1  \
    --driver-memory 10g  \
    --total-executor-cores 2  \
    --executor-cores 1  \
    --executor-memory 30g \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
    --jars ${SSD_JAR_PATH} \
    --conf spark.driver.extraClassPath=${SSD_JAR_PATH} \
    --conf spark.executor.extraClassPath=pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar