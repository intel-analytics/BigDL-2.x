#!/bin/bash

SPARK_VERSION=2.4.3

if [[ -z "${SPARK_CMD}" ]]; then
    echo "Please set SPARK_CMD environment variable"
    exit 1
fi

# Check files
if [[ ! -f ${ANALYTICS_ZOO_CONF} ]]; then
    echo "Cannot find ${ANALYTICS_ZOO_CONF}"
    exit 1
fi

if [[ ! -f ${ANALYTICS_ZOO_PY_ZIP} ]]; then
    echo "Cannot find ${ANALYTICS_ZOO_PY_ZIP}"
    exit 1
fi

if [[ ! -f ${ANALYTICS_ZOO_JAR} ]]; then
    echo "Cannot find ${ANALYTICS_ZOO_JAR}"
    exit 1
fi

function version_ge() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" == "$1"; }

if version_ge $SPARK_VERSION 2.4.4;
then
    ${SPARK_HOME}/bin/${SPARK_CMD} \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --jars local://${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    "$@"
else
    ${SPARK_HOME}/bin/${SPARK_CMD} \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    "$@"
fi

