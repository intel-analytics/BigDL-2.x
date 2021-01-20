#!/bin/bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}


echo "#1 start test for orca tf transfer_learning"
#timer
start=$(date "+%s")
#run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/transfer_learning/transfer_learning.py --cluster_mode yarn
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "orca tf transfer_learning failed"
    exit $exit_status
fi
now=$(date "+%s")
time1=$((now-start))

echo "#2 start test for orca tf basic_text_classification"
#timer
start=$(date "+%s")
#run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/basic_text_classification/basic_text_classification.py --cluster_mode yarn
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "orca tf basic_text_classification failed"
    exit $exit_status
fi
now=$(date "+%s")
time2=$((now-start))


echo "#1 orca tf transfer_learning time used:$time1 seconds"
echo "#2 orca tf basic_text_classification time used:$time2 seconds"
