#!/usr/bin/env bash
clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

echo "#1 start example test for textclassification"
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

echo "#2 start example test for image-classification"
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

echo "#3 start example test for autograd"
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

echo "#4 start example test for objectdetection"
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

echo "#5 start example test for nnframes"
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

   mkdir -p analytics-zoo-data/data/dogs-vs-cats/demo/cats
   mkdir -p analytics-zoo-data/data/dogs-vs-cats/demo/dogs
   cp analytics-zoo-data/data/dogs-vs-cats/train/cat.71* analytics-zoo-data/data/dogs-vs-cats/demo/cats
   cp analytics-zoo-data/data/dogs-vs-cats/train/dog.71* analytics-zoo-data/data/dogs-vs-cats/demo/dogs
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

echo "start example test for nnframes tensorflow SimpleTraining"
export MASTER=local[1]
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/nnframes/tensorflow/SimpleTraining.py
exit_status=$?
unset MASTER
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "nnframes tensorflow SimpleTraining failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time5=$((now-start))

echo "#6 start example test for inceptionv1 training"
#timer
start=$(date "+%s")
export MASTER=local[4]
export SPARK_DRIVER_MEMORY=20g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/inception/inception.py \
   --maxIteration 20 \
   -b 8 \
   -f hdfs://172.168.2.181:9000/imagenet-mini
exit_status=$?
unset MASTER
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "inceptionv1 training failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time6=$((now-start))

echo "#7 start example test for pytorch"
#timer
start=$(date "+%s")
echo "start example test for pytorch SimpleTrainingExample"
export MASTER=local[1]
export SPARK_DRIVER_MEMORY=5g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/SimpleTrainingExample.py
exit_status=$?
unset MASTER
unset SPARK_DRIVER_MEMORY
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "pytorch SimpleTrainingExample failed"
    exit $exit_status
fi

echo "start example test for pytorch mnist training"
export SPARK_DRIVER_MEMORY=10g
export MASTER=local[1]
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/Lenet_mnist.py
exit_status=$?
unset MASTER
unset SPARK_DRIVER_MEMORY
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "pytorch mnist training failed"
    exit $exit_status
fi

echo "start example test for pytorch resnet finetune"
export SPARK_DRIVER_MEMORY=20g
export MASTER=local[4]
export ZOO_NUM_MKLTHREADS=all
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/pytorch/train/resnet_finetune/resnet_finetune.py \
    analytics-zoo-data/data/dogs-vs-cats/samples
exit_status=$?
unset MASTER
unset ZOO_NUM_MKLTHREADS
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "pytorch resnet finetune failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time7=$((now-start))

echo "#8 start example test for tensorflow"
#timer
start=$(date "+%s")
echo "start example test for tensorflow tfnet"
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

echo "start example test for tensorflow distributed_training"

if [ -d analytics-zoo-models/model ]
then
    echo "analytics-zoo-models/model/research/slim already exists."
else
    git clone https://github.com/tensorflow/models/ analytics-zoo-models
    export PYTHONPATH=$PYTHONPATH:`pwd`/analytics-zoo-models/model/research:`pwd`/analytics-zoo-models/model/research/slim
fi

export SPARK_DRIVER_MEMORY=20g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train_lenet.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tensorflow distributed_training train_lenet failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate_lenet.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tensorflow distributed_training evaluate_lenet failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/train_mnist_keras.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tensorflow distributed_training train_mnist_keras failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/tf_optimizer/evaluate_mnist_keras.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "tensorflow distributed_training evaluate_mnist_keras failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/keras/keras_dataset.py 1

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark keras keras_dataset failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/keras/keras_ndarray.py 1

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark keras keras_ndarray failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/estimator/estimator_dataset.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark estimator estimator_dataset failed"
    exit $exit_status
fi

python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/tensorflow/tfpark/estimator/estimator_inception.py \
    --image-path analytics-zoo-data/data/dogs-vs-cats/demo --num-classes 2

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "TFPark estimator estimator_inception failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time8=$((now-start))

echo "#9 start test for anomalydetection"
#timer
start=$(date "+%s")
# prepare data
if [ -f analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv ]
then
    echo "analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/NAB/nyc_taxi/nyc_taxi.csv \
    -P analytics-zoo-data/data/NAB/nyc_taxi/
fi
sed "s/model.predict(test)/model.predict(test, batch_per_thread=56)/" ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection.py > ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection2.py

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/anomalydetection/anomaly_detection2.py \
    --nb_epoch 1 \
    -b 1008 \
    --input_dir analytics-zoo-data//data/NAB/nyc_taxi/nyc_taxi.csv
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomalydetection failed"
    exit $exit_status
fi
now=$(date "+%s")
time9=$((now-start))


echo "#10 start example test for qaranker"
start=$(date "+%s")

if [ -f analytics-zoo-data/data/glove.6B.zip ]
then
    echo "analytics-zoo-data/data/glove.6B.zip already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove.6B.zip -d analytics-zoo-data/data/glove.6B
fi
if [ -f analytics-zoo-data/data/WikiQAProcessed.zip ]
then
    echo "analytics-zoo-data/data/WikiQAProcessed.zip already exists"
else
    wget https://s3.amazonaws.com/analytics-zoo-data/WikiQAProcessed.zip -P analytics-zoo-data/data
    unzip analytics-zoo-data/data/WikiQAProcessed.zip -d analytics-zoo-data/data/
fi

# Run the example
export SPARK_DRIVER_MEMORY=3g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/qaranker/qa_ranker.py \
    --nb_epoch 2 \
    --batch_size 112 \
    --data_path analytics-zoo-data/data/WikiQAProcessed \
    --embedding_file analytics-zoo-data/data/glove.6B/glove.6B.50d.txt
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "qaranker failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time10=$((now-start))

echo "#12 start example test for vnni/openvino"
start=$(date "+%s")
if [ -d analytics-zoo-models/vnni ]
then
   echo "analytics-zoo-models/resnet_v1_50.xml already exists."
else
   wget $FTP_URI/analytics-zoo-models/openvino/vnni/resnet_v1_50.zip \
    -P analytics-zoo-models
    unzip -q analytics-zoo-models/resnet_v1_50.zip -d analytics-zoo-models/vnni
fi
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data
fi
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/vnni/openvino/predict.py \
    --model analytics-zoo-models/vnni/resnet_v1_50.xml \
    --image analytics-zoo-data/data/object-detection-coco

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "vnni/openvino failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time12=$((now-start))

echo "#13 start example test for streaming Object Detection"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
fi

if [ -f analytics-zoo-models/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model ]
then
    echo "analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
     -P analytics-zoo-models
fi

mkdir output
mkdir stream
export SPARK_DRIVER_MEMORY=2g
while true
do
   temp1=$(find analytics-zoo-data/data/object-detection-coco -type f|wc -l)
   temp2=$(find ./output -type f|wc -l)
   temp3=$(($temp1+$temp1))
   if [ $temp3 -eq $temp2 ];then
       kill -9 $(ps -ef | grep streaming_object_detection | grep -v grep |awk '{print $2}')
       rm -r output
       rm -r stream
   break
   fi
done  &
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/objectdetection/streaming_object_detection.py \
    --streaming_path ./stream \
    --model analytics-zoo-models/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
    --output_path ./output  &
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/objectdetection/image_path_writer.py \
    --streaming_path ./stream \
    --img_path analytics-zoo-data/data/object-detection-coco

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "streaming Object Detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time13=$((now-start))

echo "#14 start example test for streaming Text Classification"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/streaming/text-model ]
then
    echo "analytics-zoo-data/data/streaming/text-model already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/streaming/text-model.zip -P analytics-zoo-data/data/streaming/
    unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/
fi
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_ROOT}/pyzoo/zoo/examples/streaming/textclassification/streaming_text_classification.py \
    --model analytics-zoo-data/data/streaming/text-model/text_classifier.model \
    --index_path analytics-zoo-data/data/streaming/text-model/word_index.txt \
    --input_file analytics-zoo-data/data/streaming/text-model/textfile/ > 1.log &
while :
do
echo "I am strong and I am smart" >> analytics-zoo-data/data/streaming/text-model/textfile/s
if [ -n "$(grep "top-5" 1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep streaming_text_classification | grep -v grep |awk '{print $2}')
    rm 1.log
    sleep 1s
    break
fi
done
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "streaming Text Classification failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time14=$((now-start))

# This should be done at the very end after all tests finish.
clear_up

echo "#1 textclassification time used: $time1 seconds"
echo "#2 imageclassification time used: $time2 seconds"
echo "#3 autograd time used: $time3 seconds"
echo "#4 objectdetection time used: $time4 seconds"
echo "#5 nnframes time used: $time5 seconds"
echo "#6 inceptionV1 training time used: $time6 seconds"
echo "#7 pytorch time used: $time7 seconds"
echo "#8 tensorflow time used: $time8 seconds"
echo "#9 anomalydetection time used: $time9 seconds"
echo "#10 qaranker time used: $time10 seconds"
echo "#12 vnni/openvino time used: $time12 seconds"
echo "#13 streaming Object Detection time used: $time13 seconds"
echo "#14 streaming text classification time used: $time14 seconds"
