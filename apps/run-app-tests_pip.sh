#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

echo "#1 start app test for anomaly-detection"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi
chmod +x ${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh
${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh
sed "s/nb_epoch=20/nb_epoch=2/g; s/batch_size=1024/batch_size=1008/g" ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/anomaly-detection-nyc-taxi.py > ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection/tmp_test.py

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "anomaly-detection-nyc-taxi time used:$time1 seconds"

echo "#2 start app test for object-detection"
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/object-detection/object-detection
FILENAME="${ANALYTICS_ZOO_HOME}/apps/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget https://s3-ap-southeast-1.amazonaws.com/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/object-detection/train_dog.mp4"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-data/apps/object-detection/train_dog.mp4 -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget https://s3.amazonaws.com/analytics-zoo-data/train_dog.mp4 -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi
FILENAME="/root/.imageio/ffmpeg/ffmpeg-linux64-v3.3.1"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-data/apps/object-detection/ffmpeg-linux64-v3.3.1 -P /root/.imageio/ffmpeg/
fi

export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/object-detection/object-detection.py

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time2=$((now-start))
echo "object-detection time used:$time2 seconds"

echo "anomaly-detection-nyc-taxi time used:$time1 seconds"
echo "object-detection time used:$time2 seconds"

# This should be done at the very end after all tests finish.
clear_up

