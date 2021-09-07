#!/bin/bash

SPARK_VERSION=2.4.3

# Check environment variables
if [[ -z "${ANALYTICS_ZOO_HOME}" ]]; then
    echo "Please set ANALYTICS_ZOO_HOME environment variable"
    exit 1
fi

if [[ -z "${SPARK_HOME}" ]]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

source ${ANALYTICS_ZOO_HOME}/bin/analytics-zoo-env.sh

export SPARK_CMD=spark-submit

function version_ge() { test "$(echo "$@" | tr " " "\n" | sort -rV | head -n 1)" == "$1"; }

if version_ge $SPARK_VERSION 2.4.4;
then
    bash ${ANALYTICS_ZOO_HOME}/bin/analytics-zoo-base.sh \
        --py-files local://${ANALYTICS_ZOO_PY_ZIP} \
        "$@"
else
    bash ${ANALYTICS_ZOO_HOME}/bin/analytics-zoo-base.sh \
        --py-files ${ANALYTICS_ZOO_PY_ZIP} \
        "$@"
fi
