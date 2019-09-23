#!/bin/bash

# Check environment variables
if [[ -z "${ANALYTICS_ZOO_HOME}" ]]; then
    echo "Please set ANALYTICS_ZOO_HOME environment variable"
    exit 1
fi

if [[ -z "${SPARK_HOME}" ]]; then
    echo "Please set SPARK_HOME environment variable"
    exit 1
fi

# setup paths
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"

source ${ANALYTICS_ZOO_HOME}/bin/analytics-zoo-env.sh

export SPARK_CMD=pyspark

bash ${ANALYTICS_ZOO_HOME}/bin/analytics-zoo-base.sh \
    --conf spark.sql.catalogImplementation='in-memory' \
    --py-files ${ANALYTICS_ZOO_PY_ZIP} \
    $*
