#!/bin/bash

# Check environment variables
if [ -z "${ANALYTICS_ZOO_HOME}" ]; then
    echo "Please set ANALYTICS_ZOO_HOME environment variable"
    exit 1
fi

if [ -z "${SPARK_HOME}" ]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

# setup paths
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PY_ZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf

# Check files
if [ ! -f ${ANALYTICS_ZOO_CONF} ]; then
    echo "Cannot find ${ANALYTICS_ZOO_CONF}"
    exit 1
fi

if [ ! -f ${ANALYTICS_ZOO_PY_ZIP} ]; then
    echo "Cannot find ${ANALYTICS_ZOO_PY_ZIP}"
    exit 1
fi

if [ ! -f ${ANALYTICS_ZOO_JAR} ]; then
    echo "Cannot find ${ANALYTICS_ZOO_JAR}"
    exit 1
fi

${SPARK_HOME}/bin/spark-submit \
  --properties-file ${ANALYTICS_ZOO_CONF} \
  --py-files ${ANALYTICS_ZOO_PY_ZIP} \
  --jars ${ANALYTICS_ZOO_JAR} \
  --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
  --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
  $*
