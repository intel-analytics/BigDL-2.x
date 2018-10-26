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

echo "#10 start app test for dogs-vs-cats"
start=$(date "+%s")

# Conversion to py file and data preparation

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/transfer-learning

sed "s/setBatchSize(40)/setBatchSize(56)/g; s/file:\/\/path\/to\/data\/dogs-vs-cats\/demo/demo/g;s/path\/to\/model\/bigdl_inception-v1_imagenet_0.4.0.model/demo\/bigdl_inception-v1_imagenet_0.4.0.model/g" ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/transfer-learning.py >${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/tmp.py

FILENAME="${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/bigdl_inception-v1_imagenet_0.4.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading model"

   wget $FTP_URI/analytics-zoo-models/image-classification/bigdl_inception-v1_imagenet_0.4.0.model -P demo

   echo "Finished downloading model"
fi

FILENAME="${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/train.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading dogs and cats images"
   wget  $FTP_URI/analytics-zoo-data/data/dogs-vs-cats/train.zip  -P ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/ ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/train.zip
   mkdir -p demo/dogs
   mkdir -p demo/cats
   cp ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/train/cat.7* demo/cats
   cp ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/train/dog.7* demo/dogs
   echo "Finished downloading images"
fi

# Run the example
export SPARK_DRIVER_MEMORY=2g
python ${ANALYTICS_ZOO_HOME}/apps/dogs-vs-cats/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "dogs-vs-cats failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time1=$((now-start))
echo "dogs-vs-cats time used:$time1 seconds"

echo "#5 start app test for using_variational_autoencoder_to_generate_digital_numbers"
#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers

sed "s/nb_epoch = 6/nb_epoch=2/g; s/batch_size=batch_size/batch_size=1008/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_digital_numbers.py > ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/tmp_test.py

export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/tmp_test.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "using_variational_autoencoder_to_generate_digital_numbers failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time5=$((now-start))
echo "#5 using_variational_autoencoder_to_generate_digital_numbers time used:$time5 seconds"


echo "#12 start app test for image_classification_inference"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/tfnet/image_classification_inference

sed "s%/path/to/yourdownload%${ANALYTICS_ZOO_HOME}/apps/tfnet%g;s%file:///path/toyourdownload/dogs-vs-cats/train%${ANALYTICS_ZOO_HOME}/apps/tfnet/data/minitrain%g;s%test.jpg%${ANALYTICS_ZOO_HOME}/apps/tfnet/test.jpg%g;s%imagenet_class_index.json%${ANALYTICS_ZOO_HOME}/apps/tfnet/imagenet_class_index.json%g; s/setBatchSize(16)/setBatchSize(56)/g;" ${ANALYTICS_ZOO_HOME}/apps/tfnet/image_classification_inference.py > ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/tfnet/models/*"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading model"
   
   git clone https://github.com/tensorflow/models/ ${ANALYTICS_ZOO_HOME}/apps/tfnet/models
   
   echo "Finished downloading model"
fi
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

export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "image_classification_inference failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time12=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
echo "#12 image_classification_inference time used:$time12 seconds"

# This should be done at the very end after all tests finish.
clear_up
