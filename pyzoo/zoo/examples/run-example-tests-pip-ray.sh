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

execute_ray_test rl_pong "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/rl_pong/rl_pong.py  --iterations 10"
time1=$?

#execute_ray_test sync_parameter_server "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/parameter_server/sync_parameter_server.py  --iterations 10"
#time2=$?

execute_ray_test async_parameter_server "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/parameter_server/async_parameter_server.py  --iterations 10"
time3=$?

execute_ray_test multiagent_two_trainers ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/rllib/multiagent_two_trainers.py
time4=$?

# get_mnist_iterator in MXNet requires the data to be placed in the `data` folder of the running directory.
# The running directory of integration test is ${ANALYTICS_ZOO_ROOT}.
if [ -f ${ANALYTICS_ZOO_ROOT}/data/mnist.zip ]
then
    echo "mnist.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P ${ANALYTICS_ZOO_ROOT}/data
fi
unzip -q ${ANALYTICS_ZOO_ROOT}/data/mnist.zip -d ${ANALYTICS_ZOO_ROOT}/data

execute_ray_test lenet_mnist "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/mxnet/lenet_mnist.py -e 1 -b 256"
time5=$?

if [ -d ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/data ]
then
    echo "fashion-mnist dataset already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/fashion-mnist.zip -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/
    unzip ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion-mnist.zip
fi

sed "s/epochs=5/epochs=1/g;s/batch_size=4/batch_size=256/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py

execute_ray_test fashion_mnist "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py --backend torch_distributed"
time6=$?

if [ ! -f BSDS300-images.tgz ]; then
  wget -nv $FTP_URI/analytics-zoo-data/BSDS300-images.tgz
fi
if [ ! -d dataset/BSDS300/images ]; then
  mkdir dataset
  tar -xzf BSDS300-images.tgz -C dataset
fi

execute_ray_test super_resolution_BSDS3000 "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/super_resolution/super_resolution.py --backend torch_distributed"
time7=$?

if [ -d ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/data ]; then
  echo "Cifar10 already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10
  unzip ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.zip
fi

execute_ray_test cifar10 "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.py --backend torch_distributed"
time8=$?

if [ -f ${ANALYTICS_ZOO_ROOT}/data/airline_14col.data ]
then
    echo "airline_14col.data already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/airline_14col.data -P ${ANALYTICS_ZOO_ROOT}/data/
fi

execute_ray_test auto-xgboost-classifier "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/AutoXGBoostClassifier.py -p ${ANALYTICS_ZOO_ROOT}/data/airline_14col.data"
time9=$?

if [ -f ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/incd.csv ]
then
    echo "incd.csv already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/incd.csv -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/
fi

execute_ray_test auto-xgboost-regressor "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/AutoXGBoostRegressor.py -p ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/incd.csv"
time10=$?

execute_ray_test autoecastimator-pytorch "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoestimator/autoestimator_pytorch.py --trials 5 --epochs 2"
time11=$?

if [ ! -f ~/.chronos/dataset/nyc_taxi/nyc_taxi.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ~/.chronos/dataset/nyc_taxi/
fi

execute_ray_test autolstm_nyc_taxi "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/autolstm_nyc_taxi.py"
time12=$?

if [ ! -f {ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/nyc_taxi.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/
fi

execute_ray_test autoprophet_nyc_taxi "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/autoprophet_nyc_taxi.py --datadir ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/auto_model/nyc_taxi.csv"
time13=$?

if [ ! -f {ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/quickstart/nyc_taxi.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/quickstart/
fi

execute_ray_test autots_nyc_taxi "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/quickstart/autots_nyc_taxi.py --datadir ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/quickstart/nyc_taxi.csv"
time14=$?

if [ ! -f {ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/simulator/data_train_small.npz ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/doppelGANger_data/data_train_small.npz -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/simulator/
fi

execute_ray_test dpgansimulator_wwt "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/simulator/dpgansimulator_wwt.py --datadir ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/simulator/data_train_small.npz --epoch 1 --plot_figures False"
time15=$?

if [ ! -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
  mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/network_traffic_data.csv
fi

execute_ray_test distributed_training_network_traffic "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/distributed/distributed_training_network_traffic.py"
time16=$?

if [ ! -f ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/apps/nyc-taxi/nyc_taxi.csv -P ~/.chronos/dataset/nyc_taxi/
  mv ~/.chronos/dataset/nyc_taxi/nyc_taxi.csv ~/.chronos/dataset/nyc_taxi/nyc_taxi_data.csv
fi

execute_ray_test onnx_autotsestimator_nyc_taxi "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/onnx/onnx_autotsestimator_nyc_taxi.py"
time17=$?

if [ ! -f ~/.chronos/dataset/network_traffic/network_traffic_data.csv ]; then
  wget -nv $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ~/.chronos/dataset/network_traffic/
  mv ~/.chronos/dataset/network_traffic/data.csv ~/.chronos/dataset/network_traffic/network_traffic_data.csv
fi

execute_ray_test onnx_forecaster_network_traffic "${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/chronos/examples/onnx/onnx_forecaster_network_traffic.py"
time18=$?

echo "#1 rl_pong time used:$time1 seconds"
echo "#2 sync_parameter_server time used:$time2 seconds"
echo "#3 async_parameter_server time used:$time3 seconds"
echo "#4 multiagent_two_trainers time used:$time4 seconds"
echo "#5 mxnet_lenet time used:$time5 seconds"
echo "#6 fashion-mnist time used:$time6 seconds"
echo "#7 super-resolution time used:$time7 seconds"
echo "#8 cifar10 time used:$time8 seconds"
echo "#9 auto-xgboost-classifier time used:$time9 seconds"
echo "#10 auto-xgboost-regressor time used:$time10 seconds"
echo "#11 autoecastimator-pytorch time used:$time11 seconds"
echo "#12 autolstm_nyc_taxi time used:$time12 seconds"
echo "#13 autoprophet_nyc_taxi time used:$time13 seconds"
echo "#14 autots_nyc_taxi time used:$time14 seconds"
echo "#15 dpgansimulator_wwt time used:$time15 seconds"
echo "#16 distributed_training_network_traffic time used:$time16 seconds"
echo "#17 onnx_autotsestimator time used:$time17 seconds"
echo "#18 onnx_forecaster_network_traffic used:$time18 seconds"

clear_up
