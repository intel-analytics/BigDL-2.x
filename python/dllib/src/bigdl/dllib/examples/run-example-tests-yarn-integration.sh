#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling dllib"
  pip uninstall -y bigdl-dllib
  pip uninstall -y pyspark
}

echo "#1 start test for orca tf transfer_learning"

#timer
start=$(date "+%s")
#run the example
export SPARK_DRIVER_MEMORY=3g
hadoop fs -get ${HDFS_URI}/mnist /tmp/mnist
python ${BIGDL_ROOT}/python/dllib/src/bigdl/dllib/models/lenet/lenet5.py --on-yarn
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca tf transfer_learning failed"
  exit $exit_status
fi
now=$(date "+%s")
time1=$((now - start))

