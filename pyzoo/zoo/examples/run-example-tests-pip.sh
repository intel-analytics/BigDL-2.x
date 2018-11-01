#!/usr/bin/env bash

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

echo "start example test for textclassification"
start=$(date "+%s")

# Data preparation
if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
    echo "analytics-zoo-data/data/glove.6B.zip already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi
if [ -f analytics-zoo-data/data/20news-18828.tar.gz ]
then
    echo "analytics-zoo-data/data/20news-18828.tar.gz already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P analytics-zoo-data/data
    tar zxf analytics-zoo-data/data/20news-18828.tar.gz -C analytics-zoo-data/data/
fi

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/textclassification/text_classification.py \
    --nb_epoch 2 \
    --batch_size 112 \
    --data_path analytics-zoo-data/data/20news-18828 \
    --embedding_path analytics-zoo-data/data/glove.6B
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "textclassification failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "textclassification time used:$time1 seconds"

echo "start example test for image-classification"
#timer
start=$(date "+%s")
echo "check if model directory exists"
if [ ! -d analytics-zoo-models ]
then
    mkdir analytics-zoo-models
fi
if [ -f analytics-zoo-models/image-classification/analytics-zoo_squeezenet_imagenet_0.1.0.model ]
then
    echo "analytics-zoo-models/image-classification/analytics-zoo_squeezenet_imagenet_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models/image-classification/analytics-zoo_squeezenet_imagenet_0.1.0.model\
    -P analytics-zoo-models
fi
export SPARK_DRIVER_MEMORY=10g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/imageclassification/predict.py \
    -f hdfs://172.168.2.181:9000/kaggle/train_100 \
    --model analytics-zoo-models/analytics-zoo_squeezenet_imagenet_0.1.0.model \
    --topN 5
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "imageclassification failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time2=$((now-start))
echo "imageclassification time used:$time2 seconds"

echo "start example test for autograd"
#timer
start=$(date "+%s")

export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/custom.py 
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "autograd-custom failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/autograd/customloss.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "autograd_customloss failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time3=$((now-start))
echo "autograd time used:$time3 seconds"

echo "start example test for objectdetection"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model ]
then
    echo "analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model \
    -P analytics-zoo-models
fi

export SPARK_DRIVER_MEMORY=10g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/objectdetection/predict.py \
    analytics-zoo-models/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model \
    hdfs://172.168.2.181:9000/kaggle/train_100 \
    /tmp
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "objectdetection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time4=$((now-start))
echo "objectdetection time used:$time4 seconds"

echo "start example test for nnframes"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model ]
then
   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
   wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model \
    -P analytics-zoo-models
fi

if [ -f analytics-zoo-data/data/dogs-vs-cats/train.zip ]
then
   echo "analytics-zoo-data/data/dogs-vs-cats/train.zip already exists."
else
   # echo "Downloading dogs and cats images"
   wget  $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip\
    -P analytics-zoo-data/data/dogs-vs-cats
   unzip analytics-zoo-data/data/dogs-vs-cats/train.zip -d analytics-zoo-data/data/dogs-vs-cats
   mkdir -p analytics-zoo-data/data/dogs-vs-cats/samples
   cp analytics-zoo-data/data/dogs-vs-cats/train/cat.7* analytics-zoo-data/data/dogs-vs-cats/samples
   cp analytics-zoo-data/data/dogs-vs-cats/train/dog.7* analytics-zoo-data/data/dogs-vs-cats/samples
   # echo "Finished downloading images"
fi

# total batch size: 32 should be divided by total core number: 28
sed "s/setBatchSize(32)/setBatchSize(56)/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/image_finetuning_example.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/tmp.py
sed "s/setBatchSize(32)/setBatchSize(56)/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/ImageTransferLearningExample.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/tmp.py
sed "s/setBatchSize(4)/setBatchSize(56)/g" \
    ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/ImageInferenceExample.py \
    > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/tmp.py

export SPARK_DRIVER_MEMORY=20g

echo "start example test for nnframes imageInference"
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageInference/tmp.py \
    analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
    hdfs://172.168.2.181:9000/kaggle/train_100

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "nnframes_imageInference failed"
    exit $exit_status
fi

echo "start example test for nnframes finetune"
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/finetune/tmp.py \
    analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
    analytics-zoo-data/data/dogs-vs-cats/samples

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "nnframes_finetune failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/imageTransferLearning/tmp.py \
    analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model \
    analytics-zoo-data/data/dogs-vs-cats/samples

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "nnframes_imageTransferLearning failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time5=$((now-start))
echo "nnframes time used:$time5 seconds"

echo "start example test for tensorflow tfnet"
#timer
start=$(date "+%s")

if [ -f analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz ]
then
   echo "analytics-zoo-models/bigdl_inception-v1_imagenet_0.4.0.model already exists."
else
   wget http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz \
    -P analytics-zoo-models
   tar zxf analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17.tar.gz -C analytics-zoo-models/
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfnet/predict.py \
    --image hdfs://172.168.2.181:9000/kaggle/train_100 \
    --model analytics-zoo-models/ssd_mobilenet_v1_coco_2017_11_17/frozen_inference_graph.pb

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tfnet failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time6=$((now-start))
echo "tensorflow tfnet time used:$time6 seconds"

echo "start example test for tensorflow distributed_training"
#timer
start=$(date "+%s")

if [ -d analytics-zoo-models/model ]
then
    echo "analytics-zoo-models/model/research/slim already exists."
else
    git clone https://github.com/tensorflow/models/ analytics-zoo-models
    export PYTHONPATH=$PYTHONPATH:`pwd`/analytics-zoo-models/model/research:`pwd`/analytics-zoo-models/model/research/slim
fi

export SPARK_DRIVER_MEMORY=20g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/train_lenet.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tensorflow distributed_training train_lenet failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/distributed_training/evaluate_lenet.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tensorflow distributed_training evaluate_lenet failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time7=$((now-start))
echo "tensorflow distributed_training time used:$time7 seconds"

# This should be done at the very end after all tests finish.
clear_up
