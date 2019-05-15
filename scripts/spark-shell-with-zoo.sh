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
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export KMP_AFFINITY=granularity=fine,compact,1,0

if [ -z "${ZOO_NUM_MKLTHREADS}" ]; then
    export OMP_NUM_THREADS=1
    export KMP_BLOCKTIME=0
else
    export OMP_NUM_THREADS=${ZOO_NUM_MKLTHREADS}
if

# verbose for OpenMP
if [[ $* == *"verbose"* ]]; then
    export KMP_SETTINGS=1
    export KMP_AFFINITY=${KMP_AFFINITY},verbose
if

# Check files
if [ ! -f ${ANALYTICS_ZOO_CONF} ]; then
    echo "Cannot find ${ANALYTICS_ZOO_CONF}"
    exit 1
fi

if [ ! -f ${ANALYTICS_ZOO_JAR} ]; then
    echo "Cannot find ${ANALYTICS_ZOO_JAR}"
    exit 1
fi

${SPARK_HOME}/bin/spark-shell \
	--properties-file ${ANALYTICS_ZOO_CONF} \
	--jars ${ANALYTICS_ZOO_JAR} \
	--conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
	--conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
	$*
