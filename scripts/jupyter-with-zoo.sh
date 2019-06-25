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
export PYSPARK_DRIVER_PYTHON=jupyter
export PYSPARK_DRIVER_PYTHON_OPTS="notebook --notebook-dir=./ --ip=0.0.0.0 --no-browser --allow-root --NotebookApp.token=''"
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PY_ZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PY_ZIP}:${PYTHONPATH}

if [ -z "${KMP_AFFINITY}" ]; then
    export KMP_AFFINITY=granularity=fine,compact,1,0
fi

if [ -z "${OMP_NUM_THREADS}" ]; then
    if [ -z "${ZOO_NUM_MKLTHREADS}" ]; then
        export OMP_NUM_THREADS=1
    else
        if [ `echo $ZOO_NUM_MKLTHREADS | tr '[A-Z]' '[a-z]'` == "all" ]; then
            export OMP_NUM_THREADS=`nproc`
        else
            export OMP_NUM_THREADS=${ZOO_NUM_MKLTHREADS}
        fi
    fi
fi

if [ -z "${KMP_BLOCKTIME}" ]; then
    export KMP_BLOCKTIME=0
fi

# verbose for OpenMP
if [[ $* == *"verbose"* ]]; then
    export KMP_SETTINGS=1
    export KMP_AFFINITY=${KMP_AFFINITY},verbose
fi

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

${SPARK_HOME}/bin/pyspark \
    --properties-file ${ANALYTICS_ZOO_CONF} \
    --py-files ${ANALYTICS_ZOO_PY_ZIP} \
    --jars ${ANALYTICS_ZOO_JAR} \
    --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
    --conf spark.sql.catalogImplementation='in-memory' \
    $*
