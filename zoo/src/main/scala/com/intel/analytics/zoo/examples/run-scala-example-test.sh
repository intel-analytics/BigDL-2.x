#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist

set -e

echo "#5 start example test for resnet training"

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--executor-cores 4 --total-executor-cores 4 \
--driver-memory 50g \
--class com.intel.analytics.zoo.examples.resnet.TrainImageNet \
-f hdfs://172.168.2.181:9000/imagenet-zl \
--batchSize 2048 --nEpochs 2 --learningRate 0.1 --warmupEpoch 1 \
--maxLr 3.2 --cache /cache  --depth 50 --classes 1000

now=$(date "+%s")
time15=$((now-start))
echo "#5 Resnet time used:$time15 seconds"

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
    wget ${FTP_URI}/analytics-zoo-models/bigdl_model/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model -P analytics-zoo-model/bigdl_model/
fi

echo "#6.1 start OpenVINO Int8 Resnet example"

#timer
start=$(date "+%s")
echo "Prepare model and data"

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* \
com.intel.analytics.zoo.examples.vnni.openvino.PrepareOpenVINOResNet \
-m analytics-zoo-models/openVINO_model/resnet_v1_50.ckpt \
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
time16=$((now-start))
echo "#6.1 OpenVINO Resnet time used:$time16 seconds"

echo "#6.2 start BigDL Resnet example"

#timer
start=$(date "+%s")
echo "BigDL Perf"

java -cp ${ANALYTICS_ZOO_JAR}:${SPARK_HOME}/jars/* \
com.intel.analytics.zoo.examples.vnni.bigdl.Perf \
-m analytics-zoo-models/bigdl_model/analytics-zoo_resnet-50-int8_imagenet_0.5.0.model -b 64

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
time17=$((now-start))
echo "#6.2 BigDL Resnet time used:$time17 seconds"

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
--class com.intel.analytics.zoo.examples.tfnet.Predict \
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

echo "----------------------------------------------"
echo "App[Model-inference-example] Test"
echo "# Test 1 text-classification-training"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/
mkdir "models"

if [ -d analytics-zoo-data/data/ ]
then
    echo "analytics-zoo-data/data/ already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
    wget $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip -P analytics-zoo-data/data/glove
    unzip -q analytics-zoo-data/data/glove/glove.6B.zip -d analytics-zoo-data/data/glove/glove
    wget $FTP_URI/analytics-zoo-data/data/news20/20news-18828.tar.gz -P analytics-zoo-data/data/news20/
    tar -zxvf analytics-zoo-data/data/news20/20news-18828.tar.gz -C analytics-zoo-data/data/news20/
fi

cd text-classification-training
mvn clean
mvn clean package
mvn install
#return model-inference-examples/
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/

#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--jars ./text-classification-training/target/text-classification-training-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--conf spark.executor.memory="20g" \
--class com.intel.analytics.zoo.apps.textclassfication.training.TextClassificationTrainer \
--batchSize 2000 --nbEpoch 2 \
--trainDataDir "analytics-zoo-data/data/news20/20news-18828" \
--embeddingFile "analytics-zoo-data/data/glove/glove/glove.6B.300d.txt" \
--modelSaveDirPath "models/text-classification.bigdl"

now=$(date "+%s")
time8=$((now-start))
echo "#App[Model-inference-example] Test 1: text-classification-training time used:$time8 seconds"

echo "# Test Apps -- 2.text-classification-inference"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/text-classification-inference
mvn clean
mvn clean package

echo "# Test 2.1 text-classification-inference:SimpleDriver"
#timer
start=$(date "+%s")

java -cp target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DEMBEDDING_FILE_PATH=${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
-DMODEL_PATH=${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/models/text-classification.bigdl \
com.intel.analytics.zoo.apps.textclassfication.inference.SimpleDriver

now=$(date "+%s")
time9=$((now-start))
echo "#App[Model-inference-example] Test 2.1: text-classification-inference:SimpleDriver time used:$time9 seconds"

echo "# Test 2.2 text-classification-inference:WebServiceDriver"
#timer
start=$(date "+%s")

mvn spring-boot:run -DEMBEDDING_FILE_PATH=${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
-DMODEL_PATH=${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/models/text-classification.bigdl >1.log &
curl -d hello -x "" http://localhost:8080/predict &
while :
do
if [ -n "$(grep "class" ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/text-classification-inference/1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep text-classification | grep -v grep |awk '{print $2}')
    sleep 1s
    break
fi
done

now=$(date "+%s")
time10=$((now-start))
echo "#App[Model-inference-example] Test 2.2: text-classification-inference:WebServiceDriver time used:$time10 seconds"

echo "# Test 3.recommendation-inference"

#recommendation
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/recommendation-inference
mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples

if [ -f analytics-zoo-models/recommendation/ncf.bigdl ]
then
    echo "analytics-zoo-models/recommedation/ncf.bigdl already exists"
else
    wget $FTP_URI/analytics-zoo-models/recommendation/ncf.bigdl -P analytics-zoo-models/recommendation/
fi
echo "# Test 3.1 recommendation-inference:SimpleScalaDriver"
#timer
start=$(date "+%s")

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DMODEL_PATH=./analytics-zoo-models/recommendation/ncf.bigdl \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleScalaDriver

now=$(date "+%s")
time11=$((now-start))
echo "#App[Model-inference-example] Test 3.1: recommendation-inference:SimpleScalaDriver time used:$time11 seconds"

echo "# Test 3.2 recommendation-inference:SimpleDriver[Java]"
#timer
start=$(date "+%s")

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DMODEL_PATH=./analytics-zoo-models/recommendation/ncf.bigdl \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleDriver

now=$(date "+%s")
time12=$((now-start))
echo "#App[Model-inference-example] Test 3.2: recommendation-inference:SimpleDriver time used:$time12 seconds"


echo "# Test 4.model-inference-flink"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/model-inference-flink
mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples

if [ -f ./flink-1.7.2/bin/start-cluster.sh ]
then
    echo "flink-1.7.2/bin/start-cluster.sh already exists"
else
    wget $FTP_URI/flink-1.7.2.zip
    unzip flink-1.7.2.zip
fi

./flink-1.7.2/bin/start-cluster.sh

echo "# Test 4.1 model-inference-flink:Text Classification"
#timer
start=$(date "+%s")

./flink-1.7.2/bin/flink run \
./model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--inputFile ${ANALYTICS_ZOO_ROOT}/analytics-zoo-data/data/streaming/text-model/2.log \
--embeddingFilePath analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
--modelPath models/text-classification.bigdl \
--parallelism 1

now=$(date "+%s")
time13=$((now-start))
echo "#App[Model-inference-example] Test 4.1: model-inference-flink:Text Classification time used:$time13 seconds"

./flink-1.7.2/bin/stop-cluster.sh

wget ${FTP_URI}/analytics-zoo-models/flink_model/resnet_v1_50.ckpt

./flink-1.7.2/bin/start-cluster.sh

echo "# Test 4.2 model-inference-flink:Resnet50 Image Classification"
#timer
start=$(date "+%s")

./flink-1.7.2/bin/flink run \
-m localhost:8081 -p 1 \
-c com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification.ImageClassificationStreaming  \
${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
--modelType resnet_v1_50 --checkpointPath resnet_v1_50.ckpt  \
--image ${ANALYTICS_ZOO_ROOT}/zoo/src/test/resources/imagenet/n04370456/ \
--classes ${ANALYTICS_ZOO_ROOT}/zoo/src/main/resources/imagenet_classname.txt  \
--inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1

now=$(date "+%s")
time14=$((now-start))
echo "#App[Model-inference-example] Test 4.1: model-inference-flink:Resnet50 Image Classification time used:$time14 seconds"


./flink-1.7.2/bin/stop-cluster.sh

echo "Finish auto scala example test"

echo "Scala Examples"
echo "#1 tfnet time used:$time1 seconds"
echo "#2.1 LocalEstimator:LenetEstimator time used:$time2 seconds"
echo "#2.2 LocalEstimator:ResnetEstimator time used:$time3 seconds"
echo "#2.3 LocalEstimator:TransferLearning used:$time4 seconds"
echo "#3.1 Streaming:Object Detection time used:$time5 seconds"
echo "#3.2 Streaming:Text Classification time used:$time6 seconds"
echo "#4 chatbot time used:$time7 seconds"
echo "#5 Resnet time used:$time15 seconds"
echo "#6.1 OpenVINO Resnet time used:$time16 seconds"
echo "#6.2 BigDL Resnet time used:$time17 seconds"
echo "App Part"
echo "#1 text-classification-training time used:$time8 seconds"
echo "#2.1 text-classification-inference:SimpleDriver time used:$time9 seconds"
echo "#2.2 text-classification-inference:WebServiceDriver time used:$time10 seconds"
echo "#3.1 recommendation-inference:SimpleScalaDriver time used:$time11 seconds"
echo "#3.2 recommendation-inference:SimpleDriver time used:$time12 seconds"
echo "#4.1 model-inference-flink:Text Classification time used:$time13 seconds"
echo "#4.2 model-inference-flink:Resnet50 Image Classification time used:$time14 seconds"




