#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH
export BIGDL_JARS=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

set -e

ray stop -f

echo "#start orca ray example tests"
echo "#1 Start rl_pong example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/rl_pong/rl_pong.py --iterations 10
now=$(date "+%s")
time1=$((now-start))

echo "#2 Start multiagent example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/rllib/multiagent_two_trainers.py --iterations 5
now=$(date "+%s")
time2=$((now-start))

echo "#3 Start async_parameter example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/parameter_server/async_parameter_server.py --iterations 10
now=$(date "+%s")
time3=$((now-start))

echo "#4 Start sync_parameter example"
#start=$(date "+%s")
#python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/ray_on_spark/parameter_server/sync_parameter_server.py --iterations 10
#now=$(date "+%s")
#time4=$((now-start))

echo "#5 Start mxnet lenet example"
start=$(date "+%s")

# get_mnist_iterator in MXNet requires the data to be placed in the `data` folder of the running directory.
# The running directory of integration test is ${ANALYTICS_ZOO_ROOT}.
if [ -f ${ANALYTICS_ZOO_ROOT}/data/mnist.zip ]
then
    echo "mnist.zip already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/mnist.zip -P ${ANALYTICS_ZOO_ROOT}/data
fi
unzip -q ${ANALYTICS_ZOO_ROOT}/data/mnist.zip -d ${ANALYTICS_ZOO_ROOT}/data

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/mxnet/lenet_mnist.py -e 1 -b 256
now=$(date "+%s")
time5=$((now-start))

echo "#6 Start fashion_mnist example with Tensorboard visualization"
start=$(date "+%s")

if [ -d ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/data ]
then
    echo "fashion-mnist already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/data/fashion-mnist.zip -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/
    unzip ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion-mnist.zip
fi

sed "s/epochs=5/epochs=1/g;s/batch_size=4/batch_size=256/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/fashion_mnist/fashion_mnist_tmp.py --backend torch_distributed
now=$(date "+%s")
time6=$((now-start))


echo "#7 start example for orca super-resolution"
start=$(date "+%s")

if [ ! -f BSDS300-images.tgz ]; then
  wget -nv $FTP_URI/analytics-zoo-data/BSDS300-images.tgz
fi
if [ ! -d dataset/BSDS300/images ]; then
  mkdir dataset
  tar -xzf BSDS300-images.tgz -C dataset
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/super_resolution/super_resolution.py --backend torch_distributed

now=$(date "+%s")
time7=$((now-start))


echo "#8 start example for orca cifar10"
start=$(date "+%s")

if [ -d ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/data ]; then
  echo "Cifar10 already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/cifar10.zip -P ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10
  unzip ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.zip
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/pytorch/cifar10/cifar10.py --backend torch_distributed

now=$(date "+%s")
time8=$((now-start))

echo "#9 start example for orca auto-xgboost-classifier"
start=$(date "+%s")

if [ -f ${ANALYTICS_ZOO_ROOT}/data/airline_14col.data ]
then
    echo "airline_14col.data already exists"
else
    wget -nv $FTP_URI/analytics-zoo-data/airline_14col.data -P ${ANALYTICS_ZOO_ROOT}/data/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/AutoXGBoostClassifier.py

now=$(date "+%s")
time9=$((now-start))


echo "#10 start example for orca auto-xgboost-regressor"
start=$(date "+%s")

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/AutoXGBoostRegressor.py \
 -p ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoxgboost/incd.csv

now=$(date "+%s")
time10=$((now-start))


echo "#11 start example for orca autoestimator-pytorch"
start=$(date "+%s")

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/automl/autoestimator/autoestimator_pytorch.py \
    --trials 5 --epochs 2

now=$(date "+%s")
time11=$((now-start))


echo "Ray example tests finished"
echo "#1 orca rl_pong time used:$time1 seconds"
echo "#2 orca async_parameter_server time used:$time2 seconds"
echo "#3 orca sync_parameter_server time used:$time3 seconds"
echo "#4 orca multiagent_two_trainers time used:$time4 seconds"
echo "#5 mxnet_lenet time used:$time5 seconds"
echo "#6 fashion-mnist time used:$time6 seconds"
echo "#7 orca super-resolution example time used:$time7 seconds"
echo "#8 orca cifar10 example time used:$time8 seconds"
echo "#9 orca auto-xgboost-classifier time used:$time9 seconds"
echo "#10 orca auto-xgboost-regressor time used:$time10 seconds"
echo "#11 orca autoestimator-pytorch time used:$time11 second"
