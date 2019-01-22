#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_HOME
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

set -e

RUN_PART1=0
RUN_PART2=0
RUN_PART3=0
if [ $1 = 1 ]; then
RUN_PART1=1
RUN_PART2=0
RUN_PART3=0
elif [ $1 = 2 ]; then
RUN_PART1=0
RUN_PART2=1
RUN_PART3=0
elif [ $1 = 3 ]; then
RUN_PART1=0
RUN_PART2=0
RUN_PART3=1
else
RUN_PART1=1
RUN_PART2=1
RUN_PART3=1
fi

echo "#12 start app test for image_classification_inference"
#timer
start=$(date "+%s")
 ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/tfnet/image_classification_inference
 sed "s%/path/to/yourdownload%${ANALYTICS_ZOO_HOME}/apps/tfnet%g;s%file:///path/toyourdownload/dogs-vs-cats/train%${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain%g;s%test.jpg%${ANALYTICS_ZOO_HOME}/apps/tfnet/test.jpg%g;s%imagenet_class_index.json%${ANALYTICS_ZOO_HOME}/apps/tfnet/imagenet_class_index.json%g" ${ANALYTICS_ZOO_HOME}/apps/tfnet/image_classification_inference.py > ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
 ModelPath="${ANALYTICS_ZOO_HOME}/apps/tfnet/models/"
 rm -rf "$ModelPath"
 echo "Downloading model"
 git clone https://github.com/tensorflow/models/ ${ANALYTICS_ZOO_HOME}/apps/tfnet/models
 echo "Finished downloading model"
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint/inception_v1.ckpt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading inception_v1 checkpoint"
   
   wget http://download.tensorflow.org/models/inception_v1_2016_08_28.tar.gz -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint
   tar -zxvf ${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint/inception_v1_2016_08_28.tar.gz -C ${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint
   
   echo "Finished downloading checkpoint"
fi
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading dogs and cats images"
   
   wget $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/minitrain.zip -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/data
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain ${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain.zip
   #wget $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/data
   #unzip -d ${ANALYTICS_ZOO_HOME}/apps/tfnet/data ${ANALYTICS_ZOO_HOME}/apps/tfnet/data/train.zip
    echo "Finished downloading images"
fi
 ${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
 now=$(date "+%s")
time12=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
echo "#12 image_classification_inference time used:$time12 seconds"
