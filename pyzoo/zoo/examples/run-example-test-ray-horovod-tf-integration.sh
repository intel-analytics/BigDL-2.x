#!/usr/bin/env bash
set -e
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

echo "Start ray horovod tf example tests"

echo "#1 Start tf2 estimator lenet"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf2/mnist/lenet_mnist_keras.py --cluster_mode local --max_epoch 1
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tf2 estimator lenet failed"
    exit $exit_status
fi
now=$(date "+%s")
time1=$((now-start))

echo "Ray example tests finished"
echo "#1 tf2 estimator lenet used:$time1 seconds"
