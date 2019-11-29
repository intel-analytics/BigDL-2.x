#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}
#if image exist this two dependency, remove below
execute_ray_test(){
    echo "start example $1"
    start=$(date "+%s")
    python $2
    exit_status=$?
    if [ $exit_status -ne 0 ];
    then
        clear_up
        echo "$1 failed"
        exit $exit_status
    fi
    now=$(date "+%s")
    return $((now-start))
}

execute_ray_test rl_pong ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/rl_pong/rl_pong.py  --iterations 10
time1=$?

execute_ray_test sync_parameter_server ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/parameter_server/sync_parameter_server.py  --iterations 10
time2=$?

execute_ray_test async_parameter_server ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/parameter_server/async_parameter_server.py  --iterations 10
time3=$?

execute_ray_test multiagent_two_trainers ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray/rllib/multiagent_two_trainers.py
time4=$?

echo "#1 rl_pong time used:$time1 seconds"
echo "#2 sync_parameter_server time used:$time2 seconds"
echo "#3 async_parameter_server time used:$time3 seconds"
echo "#4 multiagent_two_trainers time used:$time4 seconds"

clear_up