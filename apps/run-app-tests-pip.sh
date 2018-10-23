#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

echo "#1 start app test for sentiment-analysis"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment
# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "sentiment analysis failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "sentiment-analysis time used:$time1 seconds"

# This should be done at the very end after all tests finish.
clear_up
