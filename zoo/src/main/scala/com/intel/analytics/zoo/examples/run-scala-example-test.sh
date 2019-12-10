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
--model analytics-zoo-models/tfnet/ \
--partition 4


echo "#2 start example test for chatbot"

if [ -d analytics-zoo-data/data/chatbot-data ]
then
    echo "analytics-zoo-data/data/object-detection-coco already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/chatbot-data.zip -P analytics-zoo-data/data
    tar -xvzf analytics-zoo-data/data/chatbot-data.tar.gz
fi

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.chatbot.Train \
-f analytics-zoo-data/data/chatbot-data


echo "#3 start example test for LocalEstimator"

if [ -d analytics-zoo-data/data/mnist ]
then
    echo "analytics-zoo-data/data/mnist already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/mnist.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/mnist.zip -d analytics-zoo-data/data/
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

if [ -d analytics-zoo-data/data/cifar10 ];then
    echo "analytics-zoo-data/data/cifar10 already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/cifar10.zip -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/cifar10.zip -d analytics-zoo-data/data/
fi

if [ -d analytics-zoo-models/localestimator/saved_model4 ];then
    echo "analytics-zoo-model/localestimator/saved_model4 already exists"
else
    wget $FTP_URI/analytics-zoo-models/localestimator/saved_model4.zip  -P analytics-zoo-model/localestimator
    unzip -q analytics-zoo-models/localestimator/saved_model4.zip -d analytics-zoo-model/localestimator/
fi

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--class com.intel.analytics.zoo.examples.localEstimator.TransferLearning \
-d analytics-zoo-data/data/cifar10 \
-m analytics-zoo-models/localestimator/saved-model4 \
-i "resnet50_input:0" -o "resnet50/activation_48/Relu:0" -b 132 -e 20 -t 10

echo "#4 start example test for streaming Object Detection"

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
--streamingPath stream --model analytics-zoo-models/object-detection/analytics-zoo_ssd-vgg16-300x300_COCO_0.1.0.model \
--output output  &
${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 2g \
--executor-memory 5g \
--class com.intel.analytics.zoo.examples.streaming.objectdetection.ImagePathWriter \
--streamingPath stream --imageSourcePath analytics-zoo-data/data/object-detection-coco

rm -r output
rm -r stream

echo "#4 start example test for streaming Text Classification"

if [ -d analytics-zoo-data/data/streaming/text-model ]
then
    echo "analytics-zoo-data/data/streaming/text-model already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/streaming/text-model.zip -P analytics-zoo-data/data/streaming/
    unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/
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

echo "----------------------------------------------"
echo "# Test Apps -- 1.text-classification-training"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/
mkdir "models"

if [ -d analytics-zoo-data/data/ ]
then
    echo "analytics-zoo-data/data/ already exists"
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
    unzip -q analytics-zoo-data/data/glove/glove.6B.zip -d analytics-zoo-data/data/glove/glove
    tar -xvzf analytics-zoo-data/data/news20/20news-18828.tar.gz
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data/
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

echo "# Test Apps -- 2.text-classification-inference"

#change files
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/text-classification-inference
mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples

java -cp ./text-classification-inference/target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.textclassfication.inference.SimpleDriver \
--EMBEDDING_FILE_PATH analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
--MODEL_PATH models/text-classification.bigdl

mvn spring-boot:run -DEMBEDDING_FILE_PATH=analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
-DMODEL_PATH=models/text-classification.bigdl >1.log &
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

mvn clean

echo "# Test Apps -- 3.recommendation-inference"
#recommendation
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/recommendation-inference

mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples

if [ -f analytics-zoo-models/recommedation/ncf.bigdl ]
then
    echo "analytics-zoo-models/recommedation/ncf.bigdl already exists"
else
    wget $FTP_URI/analytics-zoo-models/recommedation/ -P analytics-zoo-models/recommedation/
fi

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleScalaDriver \
--MODEL_PATH=./analytics-zoo-models/recommedation/ncf.bigdl

java -cp ./recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleDriver \
--MODEL_PATH=./analytics-zoo-models/models/apps/recommedation/ncf.bigdl

echo "# Test Apps -- 4.model-inference-flink"
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/model-inference-flink

mvn clean
mvn clean package
cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples


if [ -f flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/start-cluster.sh ]
then
    echo "flink-1.7.2/bin/start-cluster.sh already exists"
else
    wget http://mirrors.tuna.tsinghua.edu.cn/apache/flink/flink-1.7.2/flink-1.7.2-bin-scala_2.11.tgz
    tar zxvf flink-1.7.2-bin-scala_2.11.tgz
fi

#modify flink conf
#rm ./flink-1.7.2-bin-scala_2.11/flink-1.7.2/conf/flink-conf.yaml
#wget -P ./flink-1.7.2-bin-scala_2.11/flink-1.7.2/conf/ ${FTP_URI}/analytics-zoo-data/apps/flink/flink-conf.yaml

./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/start-cluster.sh

nc -l 9000 < analytics-zoo-data/data/streaming/text-model/2.log &
./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/flink run \
./model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--port 9000 --embeddingFilePath analytics-zoo-data/data/glove/glove/glove.6B.300d.txt \
--modelPath models/text-classification.bigdl \
--parallelism 1  &
while :
do
if [ -n "$(find . -type f -name "flink*taskexecutor*.out" | xargs grep -i "Zoo")" ];then
    echo "----Find-----"
    kill -9 $(ps -ef | grep "nc -l" | grep -v grep |awk '{print $2}')
    ./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/stop-cluster.sh
    sleep 1s
    break
fi
done

./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/stop-cluster.sh

wget ${FTP_URL}/analytics-zoo-models/flink-models/resnet_v1_50.ckpt
./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/start-cluster.sh

./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/flink run \
-m localhost:8081 -p 1 \
-c com.intel.analytics.zoo.apps.model.inference.flink.Resnet50ImageClassification.ImageClassificationStreaming  \
${ANALYTICS_ZOO_HOME}/apps/model-inference-examples/model-inference-flink/target/model-inference-flink-0.1.0-SNAPSHOT-jar-with-dependencies.jar  \
--modelType resnet_v1_50 --checkpointPath resnet_v1_50.ckpt  \
--image analytics-zoo-data/data/object-detection-coco \
--classes ${ANALYTICS_ZOO_ROOT}/zoo/src/main/resource/coco_classname.txt  \
--inputShape "1,224,224,3" --ifReverseInputChannels true --meanValues "123.68,116.78,103.94" --scale 1 > 1.log &
while :
do
if [ -n "$(grep "********" 1.log)" ];then
    echo "----Find-----"
    ./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/stop-cluster.sh
    rm 1.log
    sleep 1s
    break
fi
done

./flink-1.7.2-bin-scala_2.11/flink-1.7.2/bin/stop-cluster.sh

echo "Finish test"
