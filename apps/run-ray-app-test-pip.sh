#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

echo "#1 start app test for ray paramater-server"
#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/ray/parameter_server/sharded_parameter_server
python sharded_parameter_server.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "ray paramater-server failed"
    exit $exit_status
fi
now=$(date "+%s")
time1=$((now-start))
echo "#1 ray paramater-server time used:$time1 seconds"