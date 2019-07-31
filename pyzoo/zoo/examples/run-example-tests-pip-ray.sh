#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}
#if image exist this two dependency, remove below
echo "Install gym and atari"
pip install gym
pip install gym[atari]

#start execute
echo "Start pong example"
start=$(date "+%s")
export SPARK_DRIVER_MEMORY=4g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/rl_pong/rl_pong.py \
      --iterations 10
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "rl_pong failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "rl_pong time used:$time1 seconds"

echo "Start async_parameter example"
start=$(date "+%s")
export SPARK_DRIVER_MEMORY=4g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/parameter_server/async_parameter_server.py \
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "async_parameter_server failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time2=$((now-start))
echo "async_parameter_server time used:$time2 seconds"


echo "Start sync_parameter_server example"
start=$(date "+%s")
export SPARK_DRIVER_MEMORY=4g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/parameter_server/sync_parameter_server.py \
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "aync_parameter_server failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time3=$((now-start))
echo "sync_parameter_server time used:$time3 seconds"


echo "Start multiagent example"
start=$(date "+%s")
export SPARK_DRIVER_MEMORY=4g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/rllib/multiagent_two_trainers.py \
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "async_parameter_server failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time4=$((now-start))
echo "async_parameter_server time used:$time4 seconds"

pip uninstall gym
pip uninstall gym[artri]
clear_up