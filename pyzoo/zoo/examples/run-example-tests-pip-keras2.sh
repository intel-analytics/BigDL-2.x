#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

echo "start example test for attention"
start=$(date "+%s")
sed "s/hidden_size=128/hidden_size=32/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/transformer.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/tmp.py
export SPARK_DRIVER_MEMORY=20g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/attention/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "attention failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "attention time used:$time1 seconds"