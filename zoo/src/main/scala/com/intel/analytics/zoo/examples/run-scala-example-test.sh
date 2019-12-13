#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_ROOT=$ANALYTICS_ZOO_ROOT
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_ROOT/dist

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

bash ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master $MASTER \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--class com.intel.analytics.zoo.examples.tfnet.Predict \
--image analytics-zoo-data/data/object-detection-coco \
--model analytics-zoo-models/tfnet/tfnet/frozen_inference_graph.pb \
--partition 4

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

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.LenetLocalEstimator \
-d analytics-zoo-data/data/mnist -b 128 -e 1 -t 4

echo "##2.2 ResnetEstimator testing"

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.ResnetLocalEstimator \
-d analytics-zoo-data/data/cifar10 -b 128 -e 1 -t 4

echo "##2.3 ResnetEstimator testing"

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.TransferLearning \
-d analytics-zoo-data/data/cifar10 \
-m analytics-zoo-models/localestimator/saved_model4 \
-i "resnet50_input:0" -o "resnet50/activation_48/Relu:0" -b 132 -e 20 -t 10

echo "#3 start example test for streaming Object Detection"

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


echo "#4 start example test for streaming Text Classification"

if [ -d analytics-zoo-data/data/streaming/text-model ]
then
    echo "analytics-zoo-data/data/streaming/text-model already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/streaming/text-model.zip -P analytics-zoo-data/data/streaming/
    unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/
fi

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
echo "I am strong and I am smart" > analytics-zoo-data/data/streaming/text-model/textfile/s
if [ -n "$(grep "top-5" 1.log)" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep StreamingTextClassification | grep -v grep |awk '{print $2}')
    rm 1.log
    sleep 1s
    break
fi
done


echo "#5 start example test for chatbot"

if [ -d analytics-zoo-data/data/chatbot_short ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/chatbot_short.zip -P analytics-zoo-data/data
    unzip analytics-zoo-data/data/chatbot_short.zip -d aalytics-zoo-data/data/
fi

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.chatbot.Train \
-f analytics-zoo-data/data/chatbot_short/ -b 256 -e 2


echo "----------------------------------------------"
echo "# Test Apps -- 1.text-classification-training"

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

echo "# Test Apps -- 2.text-classification-inference"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/text-classification-inference
mvn clean
mvn clean package

java -cp target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DEMBEDDING_FILE_PATH=../analytics-zoo-data/data/glove/glove.6B.300d.txt \
-DMODEL_PATH=../models/text-classification.bigdl \
com.intel.analytics.zoo.apps.textclassfication.inference.SimpleDriver

mvn spring-boot:run -DEMBEDDING_FILE_PATH=../analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
-DMODEL_PATH=../models/text-classification.bigdl >1.log &
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


echo "# Test Apps -- 3.recommendation-inference"

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

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DMODEL_PATH=./analytics-zoo-models/recommendation/ncf.bigdl \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleScalaDriver

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
-DMODEL_PATH=./analytics-zoo-models/recommendation/ncf.bigdl \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleDriver

echo "# Test Apps -- 4.model-inference-flink"

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

./flink-1.7.2/bin/flink run \
./model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--inputFile ../../analytics-zoo-data/data/streaming/text-model/2.log \
--embeddingFilePath analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
--modelPath models/text-classification.bigdl \
--parallelism 1  &
while :
do
if [ -n "$(find . -type f -name "flink*taskexecutor*.out" | xargs grep -i "can you see it")" ];then
    echo "----Find-----"
    ./flink-1.7.2/bin/stop-cluster.sh
    sleep 1s
    break
fi
done

./flink-1.7.2/bin/stop-cluster.sh

wget ${FTP_URI}/analytics-zoo-models/flink_model/resnet_v1_50.ckpt

./flink-1.7.2/bin/start-cluster.sh

./flink-1.7.2/bin/flink run \
-m localhost:8081 -p 1 \
-c com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification.ImageClassificationStreaming  \
${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
--modelType resnet_v1_50 --checkpointPath resnet_v1_50.ckpt  \
--image ${ANALYTICS_ZOO_ROOT}/zoo/src/test/resources/imagenet/n04370456/ \
--classes ${ANALYTICS_ZOO_ROOT}/zoo/src/main/resources/imagenet_classname.txt  \
--inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1 > 1.log &
while :
do
if [ -n "$(grep "Printing result" 1.log)" ];then
    echo "----Find-----"
    ./flink-1.7.2/bin/stop-cluster.sh
    rm 1.log
    sleep 1s
    break
fi
done

./flink-1.7.2/bin/stop-cluster.sh

echo "Finish test"



