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

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection failed"
    exit $exit_status
fi

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

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time2=$((now-start))
echo "object-detection time used:$time2 seconds"

echo "#3 start app test for image-similarity"
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-similarity/image-similarity
sed "s%/tmp/images%${ANALYTICS_ZOO_HOME}/apps/image-similarity%g;s%/googlenet_places365/deploy.prototxt%/googlenet_places365/deploy_googlenet_places365.prototxt%g;s%/vgg_16_places365/deploy.prototxt%/vgg_16_places365/deploy_vgg16_places365.prototxt%g;s%/vgg_16_places365/vgg16_places365.caffemodel%/vgg_16_places365/vgg16_hybrid1365.caffemodel%g" ${ANALYTICS_ZOO_HOME}/apps/image-similarity/image-similarity.py >${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/imageClassification.tar.gz"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading images"
   
   wget $FTP_URI/analytics-zoo-data/imageClassification.tar.gz -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity
   tar -zxvf ${ANALYTICS_ZOO_HOME}/apps/image-similarity/imageClassification.tar.gz -C ${ANALYTICS_ZOO_HOME}/apps/image-similarity
   
   echo "Finished downloading images"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365/deploy_googlenet_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 deploy model"
   
   wget https://raw.githubusercontent.com/CSAILVision/places365/master/deploy_googlenet_places365.prototxt -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365
   
   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365/googlenet_places365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 weight model"
   
   wget http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365
   
   echo "Finished downloading model"
fi
FILENAME=" ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365/deploy_vgg16_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG deploy model"
   
   wget https://raw.githubusercontent.com/CSAILVision/places365/master/deploy_vgg16_places365.prototxt -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365
       
   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365/vgg16_hybrid1365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG weight model"
   
   wget http://places2.csail.mit.edu/models_places365/vgg16_hybrid1365.caffemodel -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365
   
   echo "Finished downloading model"
fi

export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time3=$((now-start))
echo "#3 image-similarity time used:$time3 seconds"

echo "anomaly-detection-nyc-taxi time used:$time1 seconds"
echo "object-detection time used:$time2 seconds"
echo "image-similarity time used:$time3 seconds"

# This should be done at the very end after all tests finish.
clear_up
