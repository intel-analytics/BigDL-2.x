#!/bin/bash

#setup pathes
curr=$PWD
echo $curr
SPARK_HOME=$HOME/spark-2.1.0-bin-hadoop2.7
analytics_zoo=$HOME/code/analytics-zoo
LIB_HOME=${analytics_zoo}/pipeline/segmentation

PYTHON_API_ZIP_PATH=$LIB_HOME/target/bigdl-segmentation-0.1-SNAPSHOT-python-api.zip
JAR_PATH=${LIB_HOME}/target/segmentation-0.1-SNAPSHOT-jar-with-dependencies.jar

# build model zoo

if [ ! -f $PYTHON_API_ZIP_PATH ]
then
  cd $LIB_HOME
  echo $LIB_HOME
  bash build.sh
  cd $curr
fi



# when you build models jar, you should have download BigDL
cd $LIB_HOME
BigDL_HOME=$(find . -type d -name  'dist-spark-*')
cd $BigDL_HOME
BigDL_HOME=$PWD

cd $curr

MASTER="local[1]"

export PYTHONPATH=${PYTHON_API_ZIP_PATH}:$PYTHONPATH
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=* --no-browser"

echo ${PYTHON_API_ZIP_PATH}
echo ${BigDL_HOME}/conf/spark-bigdl.conf
echo ${JAR_PATH}

${SPARK_HOME}/bin/pyspark \
    --master ${MASTER} \
    --driver-cores 1  \
    --driver-memory 10g  \
    --total-executor-cores 1  \
    --executor-cores 1  \
    --executor-memory 20g \
    --py-files ${PYTHON_API_ZIP_PATH} \
    --properties-file ${BigDL_HOME}/conf/spark-bigdl.conf \
    --jars ${JAR_PATH} \
    --conf spark.driver.extraClassPath=${JAR_PATH} \
    --conf spark.driver.maxResultSize=4g \
    --conf spark.executor.extraClassPath=segmentation-0.1-SNAPSHOT-jar-with-dependencies.jar
