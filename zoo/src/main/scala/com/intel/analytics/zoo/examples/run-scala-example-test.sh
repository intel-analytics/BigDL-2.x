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
--model analytics-zoo-data/model/tfnet/ \
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

bash ${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master $MASTER \
--conf spark.executor.cores=1 \
--total-executor-cores 4 \
--class com.intel.analytics.zoo.examples.tfnet.Predict \
--image analytics-zoo-data/data/object-detection-coco \
--model analytics-zoo-data/data/streaming/object-detection --partition 4

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

nc -lk 9000 < 2.log &
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

now=$(date "+%s")
time1=$((now-start))

echo "# Test Apps -- text-classification-training"

cd ${ANALYTICS_ZOO_ROOT}/apps/model-inference-examples/text-classification-training

mvn clean
mvn clean package

#timer
start=$(date "+%s")
if [ -d analytics-zoo-data/data/streaming/text-model ]
then
    echo "analytics-zoo-data/data/streaming/text-model already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/ -P analytics-zoo-data/data
    unzip -q analytics-zoo-data/data/streaming/text-model.zip -d analytics-zoo-data/data/streaming/text-model
fi

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--jars ./target/text-classification-training-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--conf spark.executor.memory="20g" \
--class com.intel.analytics.zoo.apps.textclassfication.training.TextClassificationTrainer \
--batchSize 2000 --nbEpoch 2 \
--trainDataDir "/home/jieru/download/20news-18828" \
--embeddingFile "/home/jieru/download/glove/glove.6B.300d.txt" \
--modelSaveDirPath "/home/jieru/download/models/text-classification.bigdl"

mvn clean install

#change files
mvn clean
mvn clean package

${ANALYTICS_ZOO_HOME}/bin/spark-shell-with-zoo.sh \
--master ${MASTER} \
--driver-memory 20g \
--executor-memory 20g \
--jars /home/jieru/PycharmProjects/analytics-zoo/apps/model-inference-examples/text-classification-inference/target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--class com.intel.analytics.zoo.apps.textclassfication.inference.WebServiceDriver

java -cp /home/jieru/PycharmProjects/analytics-zoo/apps/model-inference-examples/text-classification-inference/target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.textclassfication.inference.SimpleDriver

java -cp /home/jieru/PycharmProjects/analytics-zoo/apps/model-inference-examples/text-classification-inference/target/text-classification-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.textclassfication.inference.WebServiceDriver

mvn cleam

#recommendation

java -cp /home/jieru/PycharmProjects/analytics-zoo/apps/model-inference-examples/recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleScalaDriver \
--MODEL_PATH=/home/jieru/download/models/ncf.bigdl

java -cp /home/jieru/PycharmProjects/analytics-zoo/apps/model-inference-examples/recommendation-inference/target/recommendation-inference-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
com.intel.analytics.zoo.apps.recommendation.inference.SimpleDriver \
--MODEL_PATH=/home/jieru/download/models/ncf.bigdl


