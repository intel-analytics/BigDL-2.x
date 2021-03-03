#!/bin/bash
clear_up() {
  echo "Clearing up environment. Uninstalling analytics-zoo"
  pip uninstall -y analytics-zoo
  pip uninstall -y bigdl
  pip uninstall -y pyspark
}

echo "#1 start test for orca tf transfer_learning"
#timer
start=$(date "+%s")
#run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/transfer_learning/transfer_learning.py --cluster_mode yarn
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca tf transfer_learning failed"
  exit $exit_status
fi
now=$(date "+%s")
time1=$((now - start))

echo "#2 start test for orca tf basic_text_classification"
#timer
start=$(date "+%s")
sed "s/epochs=100/epochs=10/g" \
  ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/basic_text_classification/basic_text_classification.py \
  >${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/basic_text_classification/tmp.py
#run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/tf/basic_text_classification/tmp.py --cluster_mode yarn
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca tf basic_text_classification failed"
  exit $exit_status
fi
now=$(date "+%s")
time2=$((now - start))

echo "#3 start test for orca bigdl attention"
#timer
start=$(date "+%s")
#run the example
start=$(date "+%s")
sed "s/max_features = 20000/max_features = 200/g;s/max_len = 200/max_len = 20/g;s/hidden_size=128/hidden_size=8/g;s/memory=\"100g\"/memory=\"20g\"/g;s/driver_memory=\"20g\"/driver_memory=\"3g\"/g" \
  ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/attention/transformer.py \
  >${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/attention/tmp.py
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/attention/tmp.py --cluster_mode yarn
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca bigdl attention failed"
  exit $exit_status
fi
now=$(date "+%s")
time3=$((now - start))

echo "#4 start test for orca bigdl resnet-finetune"
#timer
start=$(date "+%s")
hadoop fs -test -e dogs_cats
if [ $? -ne 0 ]; then
  echo "dogs_cats not exists"
  #prepare dataset
  wget $FTP_URI/analytics-zoo-data/data/cats_and_dogs_filtered.zip -P analytics-zoo-data/data
  unzip -q analytics-zoo-data/data/cats_and_dogs_filtered.zip -d analytics-zoo-data/data
  mkdir analytics-zoo-data/data/cats_and_dogs_filtered/samples
  cp analytics-zoo-data/data/cats_and_dogs_filtered/train/cats/cat.7* analytics-zoo-data/data/cats_and_dogs_filtered/samples
  cp analytics-zoo-data/data/cats_and_dogs_filtered/train/dogs/dog.7* analytics-zoo-data/data/cats_and_dogs_filtered/samples
  hdfs dfs -put analytics-zoo-data/data/cats_and_dogs_filtered/samples dogs_cats
fi
#run the example
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/resnet_finetune/resnet_finetune.py \
  --cluster_mode yarn --imagePath dogs_cats
exit_status=$?
if [ $exit_status -ne 0 ]; then
  clear_up
  echo "orca bigdl resnet-finetune"
  exit $exit_status
fi
now=$(date "+%s")
time4=$((now - start))

echo "#5 start test for orca bigdl imageInference"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]; then
  echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
  wget -nv $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P analytics-zoo-models
fi

#run the example
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/bigdl/imageInference/imageInference.py \
  -m analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
  -f ${HDFS_URI}/kaggle/train_100 \
  --cluster_mode yarn
exit_status=$?
if [ $exit_status -ne 0 ]; then
  echo "orca bigdl imageInference failed"
  exit $exit_status
fi
now=$(date "+%s")
time5=$((now - start))

echo "#start orca ray example tests"

echo "#6 Start rl_pong example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/ray_on_spark/rl_pong/rl_pong.py --iterations 10 --cluster_mode yarn
now=$(date "+%s")
time6=$((now-start))

echo "#7 Start multiagent example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/ray_on_spark/rllib/multiagent_two_trainers.py --iterations 5 --cluster_mode yarn
now=$(date "+%s")
time7=$((now-start))

echo "#8 Start async_parameter example"
if [ ! -f MNIST_data.zip ]; then
  wget $FTP_URI/analytics-zoo-data/MNIST_data.zip
fi
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/ray_on_spark/parameter_server/async_parameter_server.py --iterations 10 --num_workers 2 --cluster_mode yarn
now=$(date "+%s")
time8=$((now-start))

echo "#9 Start sync_parameter example"
start=$(date "+%s")
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/orca/learn/ray_on_spark/parameter_server/sync_parameter_server.py --iterations 10 --num_workers 2 --cluster_mode yarn
now=$(date "+%s")
time9=$((now-start))
clear_up

echo "#1 orca tf transfer_learning time used:$time1 seconds"
echo "#2 orca tf basic_text_classification time used:$time2 seconds"
echo "#3 orca bigdl attention time used:$time3 seconds"
echo "#4 orca bigdl resnet-finetune time used:$time4 seconds"
echo "#5 orca bigdl imageInference time used:$time5 seconds"
echo "#6 orca rl_pong time used:$time6 seconds"
echo "#7 orca multiagent time used:$time7 seconds"
echo "#8 orca async_parameter_server time used:$time8 seconds"
echo "#9 orca sync_parameter_server time used:$time9 seconds"
