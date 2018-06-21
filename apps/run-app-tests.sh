#!/bin/bash

export SPARK_HOME=$SPARK_HOME
export MASTER=local[4]
export FTP_URI=$FTP_URI
export ANALYTICS_ZOO_HOME=$ANALYTICS_ZOO_HOME
export ANALYTICS_ZOO_HOME_DIST=$ANALYTICS_ZOO_HOME/dist
export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME_DIST}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`
export ANALYTICS_ZOO_PYZIP=`find ${ANALYTICS_ZOO_HOME_DIST}/lib -type f -name "analytics-zoo*python-api.zip"`
export ANALYTICS_ZOO_CONF=${ANALYTICS_ZOO_HOME_DIST}/conf/spark-analytics-zoo.conf
export PYTHONPATH=${ANALYTICS_ZOO_PYZIP}:$PYTHONPATH

chmod +x ./apps/ipynb2py.sh

set -e

echo "#1 start app test for anomaly-detection-nyc-taxi"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/anomaly-detection/anomaly-detection-nyc-taxi

chmod +x $ANALYTICS_ZOO_HOME/scripts/data/NAB/nyc_taxi/get_nyc_taxi.sh

$ANALYTICS_ZOO_HOME/scripts/data/NAB/nyc_taxi/get_nyc_taxi.sh

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi.py
now=$(date "+%s")
time1=$((now-start))

echo "#2 start app test for object-detection"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/object-detection/object-detection

FILENAME="$ANALYTICS_ZOO_HOME/apps/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model"

if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists" 
else
    wget $FTP_URI/analytics-zoo-models-new/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model -P $ANALYTICS_ZOO_HOME/apps/object-detection/
fi 
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists" 
else
    wget https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model -P $ANALYTICS_ZOO_HOME/apps/object-detection/
fi 
FILENAME="$ANALYTICS_ZOO_HOME/apps/object-detection/train_dog.mp4"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists" 
else
    wget $FTP_URI/analytics-zoo-data/apps/object-detection/train_dog.mp4 -P $ANALYTICS_ZOO_HOME/apps/object-detection/
fi 
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists" 
else
    wget https://s3.amazonaws.com/analytics-zoo-data/train_dog.mp4 -P $ANALYTICS_ZOO_HOME/apps/object-detection/
fi 
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/object-detection/object-detection.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/object-detection/object-detection.py
now=$(date "+%s")
time2=$((now-start))

echo "#3 start app test for ncf-explicit-feedback"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/recommendation/ncf-explicit-feedback

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/recommendation/ncf-explicit-feedback.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/recommendation/ncf-explicit-feedback.py
now=$(date "+%s")
time3=$((now-start))

echo "#4 start app test for wide_n_deep"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/recommendation/wide_n_deep

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/recommendation/wide_n_deep.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/recommendation/wide_n_deep.py
now=$(date "+%s")
time4=$((now-start))

echo "#5 start app test for using_variational_autoencoder_and_deep_feature_loss_to_generate_faces"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces

sed -i "s/data_files\[\:100000\]/data_files\[\:5000\]/g" ./apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/bigdl_vgg-16_imagenet_0.4.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG model"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational-autoencoder/bigdl_vgg-16_imagenet_0.4.0.model --no-host-directories
   echo "Finished"
fi

FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi
        
${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py,${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/utils.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
now=$(date "+%s")
time5=$((now-start))

echo "#6 start app test for using_variational_autoencoder_to_generate_faces"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces

sed -i "s/data_files\[\:100000\]/data_files\[\:5000\]/g" ./apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py,${ANALYTICS_ZOO_HOME}/apps/variational_autoencoder/utils.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
now=$(date "+%s")
time6=$((now-start))

echo "#7 start app test for using_variational_autoencoder_to_generate_digital_numbers"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py
now=$(date "+%s")
time7=$((now-start))

echo "#8 start app test for sentiment-analysis"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/sentiment-analysis/sentiment

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-cores 2  \
        --driver-memory 12g  \
        --total-executor-cores 2  \
        --executor-cores 2  \
        --executor-memory 12g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment.py
now=$(date "+%s")
time8=$((now-start))

echo "#9 start app test for image-augmentation"
#timer
start=$(date "+%s")
./apps/ipynb2py.sh ./apps/image-augmentation/image-augmentation

${SPARK_HOME}/bin/spark-submit \
        --master ${MASTER} \
        --driver-memory 1g  \
        --executor-memory 1g \
        --conf spark.akka.frameSize=64 \
        --py-files ${ANALYTICS_ZOO_PYZIP},${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py  \
        --properties-file ${ANALYTICS_ZOO_CONF} \
        --jars ${ANALYTICS_ZOO_JAR} \
        --conf spark.driver.extraClassPath=${ANALYTICS_ZOO_JAR} \
        --conf spark.executor.extraClassPath=${ANALYTICS_ZOO_JAR} \
        ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py
now=$(date "+%s")
time9=$((now-start))

echo "#1 anomaly-detection-nyc-taxi time used:$time1 seconds"
echo "#2 object-detection time used:$time2 seconds"
echo "#3 ncf-explicit-feedback time used:$time3 seconds"
echo "#4 wide_n_deep time used:$time4 seconds"
echo "#5 using_variational_autoencoder_and_deep_feature_loss_to_generate_faces time used:$time5 seconds"
echo "#6 using_variational_autoencoder_to_generate_faces time used:$time6 seconds"
echo "#7 using_variational_autoencoder_to_generate_digital_numbers time used:$time7 seconds"
echo "#8 sentimentAnalysis time used:$time8 seconds"
echo "#9 image-augmentation time used:$time9 seconds"