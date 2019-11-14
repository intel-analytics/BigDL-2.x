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

execute_ray_test rl_pong ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/rayexample/rl_pong/rl_pong.py  --iterations 10
time9=$?

execute_ray_test sync_parameter_server ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/rayexample/parameter_server/sync_parameter_server.py  --iterations 10
time10=$?

execute_ray_test async_parameter_server ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/rayexample/parameter_server/async_parameter_server.py  --iterations 10
time11=$?

execute_ray_test multiagent_two_trainers ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/rayexample/rllibexample/multiagent_two_trainers.py
time12=$?

echo "#9 rl_pong time used:$time9 seconds"
echo "#10 sync_parameter_server time used:$time10 seconds"
echo "#11 async_parameter_server time used:$time11 seconds"
echo "#12 multiagent_two_trainers time used:$time12 seconds"

clear_up