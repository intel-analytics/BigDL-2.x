#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
set -e

echo "#1 start example test for tfnet"

if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data/
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
fi

if [ -d analytics-zoo-models/tfnet ]
then
    echo "analytics-zoo-model/tfnet already exists"
else
    wget $FTP_URI/analytics-zoo-models/tfnet/tfnet.zip -P analytics-zoo-models/tfnet/
    unzip -q analytics-zoo-models/tfnet/tfnet.zip -d analytics-zoo-models/tfnet/
fi

#timer
start=$(date "+%s")

bash ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master $MASTER \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--class com.intel.analytics.zoo.examples.tensorflow.tfnet.Predict \
--image analytics-zoo-data/data/object-detection-coco \
--model analytics-zoo-models/tfnet/tfnet/frozen_inference_graph.pb \
--partition 4

now=$(date "+%s")
time1=$((now-start))
echo "#1 Tfnet time used:$time1 seconds"

echo "#2 start example test for LocalEstimator"

if [ -d analytics-zoo-data/data/mnist ]
then
    echo "analytics-zoo-data/data/mnist already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/mnist.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/mnist.zip -d analytics-zoo-data/data/
fi

if [ -d analytics-zoo-data/data/cifar10 ];then
    echo "analytics-zoo-data/data/cifar10 already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/cifar10.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/cifar10.zip -d analytics-zoo-data/data/
fi

if [ -d analytics-zoo-models/localestimator/saved_model4 ];then
    echo "analytics-zoo-models/localestimator/saved_model4 already exists"
else
    wget $FTP_URI/analytics-zoo-models/localestimator/saved_model4.zip  -P analytics-zoo-models/localestimator
    unzip -q analytics-zoo-models/localestimator/saved_model4.zip -d analytics-zoo-models/localestimator/
fi

echo "##2.1 LenetEstimator testing"

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.LenetLocalEstimator \
-d analytics-zoo-data/data/mnist -b 128 -e 1 -t 4

now=$(date "+%s")
time2=$((now-start))
echo "#2.1 LocalEstimator:LenetEstimator time used:$time2 seconds"

echo "##2.2 ResnetEstimator testing"

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.ResnetLocalEstimator \
-d analytics-zoo-data/data/cifar10 -b 128 -e 1 -t 4

now=$(date "+%s")
time3=$((now-start))
echo "#2.2 LocalEstimator:ResnetEstimator time used:$time3 seconds"

echo "##2.3 TransferLearning testing"

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.TransferLearning \
-d analytics-zoo-data/data/cifar10 \
-m analytics-zoo-models/localestimator/saved_model4 \
-i "resnet50_input:0" -o "resnet50/activation_48/Relu:0" -b 132 -e 20 -t 10

now=$(date "+%s")
time4=$((now-start))
echo "#2.3 LocalEstimator:TransferLearning time used:$time4 seconds"

echo "#3 start example test for Streaming Test"
echo "#3.1 start example test for streaming Object Detection"

if [ -d analytics-zoo-data/data/object-detection-coco ];then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
fi

if [ -f analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model ];then
    echo "analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model already exists"
else
    wget ${FTP_URI}/analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model -P analytics-zoo-models/object-detection/
fi

mkdir output
mkdir stream

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master local[20] \
--driver-memory 4g \
--executor-memory 5g \
--class com.intel.analytics.zoo.examples.streaming.objectdetection.StreamingObjectDetection \
--streamingPath ./stream --model analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
--output ./output > 1.log &

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master local[2] \
--driver-memory 2g \
--executor-memory 5g \
--class com.intel.analytics.zoo.examples.streaming.objectdetection.ImagePathWriter \
--imageSourcePath analytics-zoo-data/data/object-detection-coco --streamingPath ./stream

while true
do
   cp ./stream/0.txt ./stream/1.txt
   temp1=$(find analytics-zoo-data/data/object-detection-coco -type f|wc -l)
   temp2=$(find ./output -type f|wc -l)
   temp3=$(($temp1+$temp1))
   if [ $temp3 -le $temp2 ];then
       kill -9 $(ps -ef | grep StreamingObjectDetection | grep -v grep |awk '{print $2}')
       rm -r output
       rm -r stream
       rm 1.log
       echo "Finished streaming"
       break
   fi
done

now=$(date "+%s")
time5=$((now-start))
echo "#3.1 Streaming:Object Detection time used:$time5 seconds"

echo "#3.2 start example test for streaming Text Classification"

if [ -d analytics-zoo-data/data/streaming/text-model ]
then
    echo "analytics-zoo-data/data/streaming/text-model already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/streaming/text-model.zip -P analytics-zoo-data/data/streaming/
    unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/
fi

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 2g \
--executor-memory 5g \
--class com.intel.analytics.zoo.examples.streaming.textclassification.StreamingTextClassification \
--model analytics-zoo-data/data/streaming/text-model/text_classifier.model \
--indexPath analytics-zoo-data/data/streaming/text-model/word_index.txt \
--inputFile analytics-zoo-data/data/streaming/text-model/textfile/ > 1.log &

while :
do
echo "I am strong and I am smart" >> analytics-zoo-data/data/streaming/text-model/textfile/s
if [ -n "$(grep "top-5" 1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep StreamingTextClassification | grep -v grep |awk '{print $2}')
    rm 1.log
    sleep 1s
    break
fi
done

now=$(date "+%s")
time6=$((now-start))
echo "#3.2 Streaming:Text Classification time used:$time6 seconds"

echo "#4 start example test for chatbot"

if [ -d analytics-zoo-data/data/chatbot_short ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/chatbot_short.zip -P analytics-zoo-data/data
    unzip analytics-zoo-data/data/chatbot_short.zip -d analytics-zoo-data/data/
fi

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.chatbot.Train \
-f analytics-zoo-data/data/chatbot_short/ -b 32 -e 2

now=$(date "+%s")
time7=$((now-start))
echo "#4 Chatbot time used:$time7 seconds"

echo "#5 start example test for resnet training"

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--executor-cores 4 --total-executor-cores 4 \
--driver-memory 50g \
--class com.intel.analytics.zoo.examples.resnet.TrainImageNet \
-f hdfs://172.168.2.181:9000/imagenet-zl \
--batchSize 32 --nEpochs 2 --learningRate 0.1 --warmupEpoch 1 \
--maxLr 3.2 --cache /cache  --depth 50 --classes 1000

now=$(date "+%s")
time8=$((now-start))
echo "#5 Resnet time used:$time8 seconds"

echo "#6 start example test for vnni"

if [ -d analytics-zoo-data/data/imagenet_val ]
then
    echo "analytics-zoo-data/data/imagenet_val already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/imagenet_val.zip -P analytics-zoo-data/data/
    unzip -q analytics-zoo-data/data/imagenet_val.zip -d analytics-zoo-data/data/
fi

if [ -d analytics-zoo-data/data/opencvlib/lib ]
then
    echo "analytics-zoo-data/data/opencvlib/lib already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/opencvlib/opencv_4.0.0_ubuntu_lib.tar -P analytics-zoo-data/data/opencvlib/
    tar -xvf analytics-zoo-data/data/opencvlib/opencv_4.0.0_ubuntu_lib.tar -C analytics-zoo-data/data/opencvlib/
fi

if [ -f analytics-zoo-models/openVINO_model/resnet_v1_50.ckpt ]
then
    echo "analytics-zoo-models/flink_model/resnet_v1_50.ckpt already exists"
else
    wget ${FTP_URI}/analytics-zoo-models/flink_model/resnet_v1_50.ckpt -P analytics-zoo-models/openVINO_model/
fi

if [ -f analytics-zoo-models/bigdl_model/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model ]
then
    echo "analytics-zoo-models/bigdl_model/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model already exists"
else
    wget ${FTP_URI}/analytics-zoo-models/bigdl_model/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model -P analytics-zoo-models/bigdl_model/
fi

echo "#6.1 start OpenVINO Int8 Resnet example"

#timer
start=$(date "+%s")
echo "Prepare model and data"

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* \
com.intel.analytics.zoo.examples.vnni.openvino.PrepareOpenVINOResNet \
-m analytics-zoo-models/openVINO_model \
-v analytics-zoo-data/data/imagenet_val -l analytics-zoo-data/data/opencvlib/lib

echo "OpenVINO Perf"

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* \
com.intel.analytics.zoo.examples.vnni.openvino.Perf \
-m analytics-zoo-models/openVINO_model/resnet_v1_50_inference_graph.xml \
-w analytics-zoo-models/openVINO_model/resnet_v1_50_inference_graph.bin

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh  \
--master ${MASTER} --driver-memory 4g \
--class com.intel.analytics.zoo.examples.vnni.openvino.Perf \
-m analytics-zoo-models/openVINO_model/resnet_v1_50_inference_graph.xml \
-w analytics-zoo-models/openVINO_model/resnet_v1_50_inference_graph.bin --onSpark

echo "OpenVINO ImageNetEvaluation"

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} --driver-memory 100g \
--class com.intel.analytics.zoo.examples.vnni.openvino.ImageNetEvaluation \
-f hdfs://172.168.2.181:9000/imagenet-zl/val/imagenet-seq-0_0.seq \
-m analytics-zoo-models/openVINO_model/resnet_v1_50_inference_graph.xml \
-w analytics-zoo-models/openVINO_model/resnet_v1_50_inference_graph.bin

echo "OpenVINO Predict"

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} --driver-memory 10g \
--class com.intel.analytics.zoo.examples.vnni.openvino.Predict \
-f zoo/src/test/resources/imagenet/n04370456/ \
-m analytics-zoo-models/openVINO_model/resnet_v1_50_inference_graph.xml \
-w analytics-zoo-models/openVINO_model/resnet_v1_50_inference_graph.bin

now=$(date "+%s")
time9=$((now-start))
echo "#6.1 OpenVINO Resnet time used:$time9 seconds"

echo "#6.2 start BigDL Resnet example"

#timer
start=$(date "+%s")
echo "BigDL Perf"

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* \
com.intel.analytics.zoo.examples.vnni.bigdl.Perf \
-m analytics-zoo-models/bigdl_model/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model \
-b 64 -i 20

echo "BigDL ImageNetEvaluation"

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--class com.intel.analytics.zoo.examples.vnni.bigdl.ImageNetEvaluation \
-f hdfs://172.168.2.181:9000/imagenet-zl/val/imagenet-seq-0_0.seq \
-m analytics-zoo-models/bigdl_model/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model

echo "BigDL Predict"

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--class com.intel.analytics.zoo.examples.vnni.bigdl.Predict \
-f zoo/src/test/resources/imagenet/n04370456/ \
-m analytics-zoo-models/bigdl_model/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model

now=$(date "+%s")
time10=$((now-start))
echo "#6.2 BigDL Resnet time used:$time10 seconds"

echo "Scala Examples"
echo "#1 tfnet time used:$time1 seconds"
echo "#2.1 LocalEstimator:LenetEstimator time used:$time2 seconds"
echo "#2.2 LocalEstimator:ResnetEstimator time used:$time3 seconds"
echo "#2.3 LocalEstimator:TransferLearning used:$time4 seconds"
echo "#3.1 Streaming:Object Detection time used:$time5 seconds"
echo "#3.2 Streaming:Text Classification time used:$time6 seconds"
echo "#4 chatbot time used:$time7 seconds"
echo "#5 Resnet time used:$time8 seconds"
echo "#6.1 OpenVINO Resnet time used:$time9 seconds"
echo "#6.2 BigDL Resnet time used:$time10 seconds"
