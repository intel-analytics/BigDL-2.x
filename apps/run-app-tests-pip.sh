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


echo "#6 start app test for using_variational_autoencoder_to_generate_faces"
#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces

sed -i "s/data_files\[\:100000\]/data_files\[\:5000\]/g; s/batch_size=batch_size/batch_size=1008/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
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

export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "using_variational_autoencoder_to_generate_faces failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time6=$((now-start))
echo "#6 using_variational_autoencoder_to_generate_faces time used:$time6 seconds"


echo "#7 start app test for using_variational_autoencoder_and_deep_feature_loss_to_generate_faces"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces

sed -i "s/data_files\[\:100000\]/data_files\[\:5000\]/g; s/batch_size=batch_size/batch_size=1008/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG model"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model --no-host-directories
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

export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "using_variational_autoencoder_and_deep_feature_loss_to_generate_faces failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time7=$((now-start))
echo "#7 using_variational_autoencoder_and_deep_feature_loss_to_generate_faces time used:$time7 seconds"

# This should be done at the very end after all tests finish.
clear_up
