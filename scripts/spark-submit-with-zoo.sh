#!/bin/bash

# Check environment variables
if [ -z "${ZOO_HOME}" ]; then
    echo "Please set ZOO_HOME environment variable"
    exit 1
fi

if [ -z "${SPARK_HOME}" ]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

#setup pathes
export ZOO_JAR_NAME=`ls ${ZOO_HOME}/lib/ | grep jar-with-dependencies.jar`
export ZOO_JAR="${ZOO_HOME}/lib/$ZOO_JAR_NAME"
export ZOO_PY_ZIP_NAME=`ls ${ZOO_HOME}/lib/ | grep python-api.zip`
export ZOO_PY_ZIP="${ZOO_HOME}/lib/$ZOO_PY_ZIP_NAME"
export ZOO_CONF=${ZOO_HOME}/conf/spark-bigdl.conf

# Check files
if [ ! -f ${ZOO_CONF} ]; then
    echo "Cannot find ${ZOO_CONF}"
    exit 1
fi

if [ ! -f ${ZOO_PY_ZIP} ]; then
    echo ${ZOO_PY_ZIP}
    echo "Cannot find ${ZOO_PY_ZIP}"
    exit 1
fi

if [ ! -f $ZOO_JAR ]; then
    echo "Cannot find $ZOO_JAR"
    exit 1
fi

${SPARK_HOME}/bin/spark-submit \
  --properties-file ${ZOO_CONF} \
  --py-files ${ZOO_PY_ZIP} \
  --jars ${ZOO_JAR} \
  --conf spark.driver.extraClassPath=${ZOO_JAR} \
  --conf spark.executor.extraClassPath=${ZOO_JAR} \
  $*
