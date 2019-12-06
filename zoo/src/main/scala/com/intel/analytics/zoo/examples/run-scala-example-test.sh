#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist

set -e

echo "#1 start example test for tfnet"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/object-detection-coco
fi

bash ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master $MASTER \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--class com.intel.analytics.zoo.examples.tfnet.Predict \
--image analytics-zoo-data/data/object-detection-coco \
--model analytics-zoo-model/tfnet/ \
--partition 4

now=$(date "+%s")
time1=$((now-start))

echo "#2 start example test for chatbot"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/chatbot-data ]
then
    echo "analytics-zoo-data/data/object-detection-coco.zip already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    tar -xvzf analytics-zoo-data/data/chatbot-data.tar.gz
fi

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.chatbot.Train \
-f analytics-zoo-data/data/chatbot-data

now=$(date "+%s")
time1=$((now-start))


echo "#3 start example test for LocalEstimator"

if [ -d analytics-zoo-data/data/mnist ]
then
    echo "analytics-zoo-data/data/mnist already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/mnist.zip -d analytics-zoo-data/data/mnist
fi

echo "#3.1 LenetEstimator testing"

#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.LenetLocalEstimator \
-d analytics-zoo-data/data/mnist -b 128 -e 1 -t 4

now=$(date "+%s")
time1=$((now-start))

echo "#3.2 ResnetEstimator testing"

#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.ResnetLocalEstimator \
-d analytics-zoo-data/data/mnist -b 128 -e 1 -t 4

echo "#3.3 ResnetEstimator testing"

if [ -d analytics-zoo-data/data/cifar10 ]
then
    echo "analytics-zoo-data/data/cifar10 already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/cifar10.zip -d analytics-zoo-data/data/cifar10
fi

#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.TransferLearning \
-d analytics-zoo-data/data/cifar10 \
-m analytics-zoo-data/model/saved-model4 \
-i "resnet50_input:0" -o "resnet50/activation_48/Relu:0" -b 132 -e 20 -t 10


now=$(date "+%s")
time1=$((now-start))

echo "#4 start example test for streaming Object Detection"
#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/object-detection-coco ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/object-detection-coco
fi

if [ -f analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model ]
then
    echo "analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model already exists"
else
    wget $FTP_URI/analytics-zoo-models/object-detection/ -P analytics-zoo-models/object-detection/
fi

mkdir output
mkdir stream
while true
do
   temp1=$(find analytics-zoo-data/data/object-detection-coco -type f|wc -l)
   temp2=$(find ./output -type f|wc -l)
   temp3=$(($temp1+$temp1))
   if [ $temp3 -eq $temp2 ];then
       kill -9 $(ps -ef | grep StreamingObjectDetection | grep -v grep |awk '{print $2}')
   break
   fi
done  &
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 4g \
--executor-memory 5g \
--class com.intel.analytics.zoo.examples.streaming.objectdetection.StreamingObjectDetection \
--streamingPath ./stream --model analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
--output ./output  &
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 2g \
--executor-memory 5g \
--class com.intel.analytics.zoo.examples.streaming.objectdetection.ImagePathWriter \
--streamingPath ./stream --imageSourcePath analytics-zoo-data/data/object-detection-coco

rm -r output
rm -r stream

bash ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master $MASTER \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--class com.intel.analytics.zoo.examples.tfnet.Predict \
--image analytics-zoo-data/data/object-detection-coco \
--model analytics-zoo-models/object-detections/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model --partition 4

now=$(date "+%s")
time1=$((now-start))

echo "#4 start example test for streaming Text Classification"

#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/streaming/text-model ]
then
    echo "analytics-zoo-data/data/streaming/text-model already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/text-model
fi

nc -lk 9000 < analytics-zoo-data/data/streaming/text-model/2.log &
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 2g \
--executor-memory 5g \
--class com.intel.analytics.zoo.examples.streaming.textclassification.StreamingTextClassification \
--model analytics-zoo-data/data/streaming/text-model/text_classifier.model \
--indexPath word_index.txt --port 9000 >>1.log &
while :
do
if [ -n "$(grep "top-5" 1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep StreamingTextClassification | grep -v grep |awk '{print $2}')
    kill -9 $(ps -ef | grep "nc -lk" | grep -v grep |awk '{print $2}')
    sleep 1s
    break
fi
done

rm 1.log

now=$(date "+%s")
time1=$((now-start))

echo "# Test Apps -- text-classification-training"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/
mkdir "models"

if [ -d analytics-zoo-data/data/ ]
then
    echo "analytics-zoo-data/data/ already exists"
    unzip -q analytics-zoo-data/data/glove/glove.6B.zip -d analytics-zoo-data/data/glove/glove
    tar -xvzf analytics-zoo-data/data/news20/20news-18828.tar.gz
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/glove/glove.6B.zip -d analytics-zoo-data/data/glove/glove
    tar -xvzf analytics-zoo-data/data/news20/20news-18828.tar.gz
fi

cd text-classification-training

mvn clean
mvn clean package
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

mvn clean install

echo "# Test Apps -- text-classification-inference"

#change files
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/text-classification-inference
mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples

java -cp ./text-classification-inference/target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.textclassfication.inference.SimpleDriver \
--EMBEDDING_FILE_PATH analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
--MODEL_PATH models/text-classification.bigdl


#WebServiceDriver has some problem!
#java -cp /home/jieru/PycharmProjects/analytics-zoo/apps/model-inference-examples/text-classification-inference/target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
#com.intel.analytics.zoo.apps.textclassfication.inference.WebServiceDriver

mvn cleam

echo "# Test Apps -- recommendation-inference"
#recommendation
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/recommendation-inference

mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples

if [ -f analytics-zoo-models/models/apps/recommedation/ncf.bigdl ]
then
    echo "analytics-zoo-models/models/apps/recommedation/ncf.bigdl already exists"
else
    wget $FTP_URI/analytics-zoo-models/models/apps/recommedation/ -P analytics-zoo-models/models/apps/recommedation/
fi

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleScalaDriver \
--MODEL_PATH=./analytics-zoo-models/models/apps/recommedation/ncf.bigdl

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleDriver \
--MODEL_PATH=./analytics-zoo-models/models/apps/recommedation/ncf.bigdl

echo "# Test Apps -- model-inference-flink"
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/model-inference-flink

mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples


if [ -f flink-1.7.2/bin/start-cluster.sh ]
then
    echo "flink-1.7.2/bin/start-cluster.sh already exists"
else
    wget http://mirrors.tuna.tsinghua.edu.cn/apache/flink/flink-1.7.2/flink-1.7.2-bin-scala_2.11.tgz
    tar zxvf flink-1.7.2-bin-scala_2.11.tgz
fi

./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/start-cluster.sh

nc -l 9000 < analytics-zoo-data/data/streaming/text-model/2.log &
./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/flink run \
./model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--port 9000 --embeddingFilePath analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
--modelPath models/text-classification.bigdl \
--parallelism 2 > 1.log &
while :
do
if [ -n "$(grep "top-5" 1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep StreamingTextClassification | grep -v grep |awk '{print $2}')
    kill -9 $(ps -ef | grep "nc -lk" | grep -v grep |awk '{print $2}')
    sleep 1s
    break
fi
done


#./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/flink run \
#    -m localhost:8081 -p 2 \
#    -c com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification.ImageClassificationStreaming  \
#    ${ANALYTICS_ZOO_HOME}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
#    --modelType resnet_v1_50 --checkpointPath ~/download/models  \
#    --inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1