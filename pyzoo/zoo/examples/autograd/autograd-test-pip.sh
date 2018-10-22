#!/usr/bin/env bash

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

echo "start example test for autograd"
start=$(date "+%s")

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/custom.py \
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/customloss.py \
    --nb_epoch 2 \
    --batch_size 112 \
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "autograd failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "autograd time used:$time1 seconds"

# This should be done at the very end after all tests finish.
clear_up
