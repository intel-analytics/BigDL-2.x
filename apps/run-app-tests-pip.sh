#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

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
   
    mkdir -p ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets
    touch ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/__init__.py
    touch ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/inception.py
    echo "from nets.inception_v1 import inception_v1" >> ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/inception.py
    echo "from nets.inception_v1 import inception_v1_arg_scope" >> ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/inception.py
    wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/inception_utils.py -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/
    wget https://raw.githubusercontent.com/tensorflow/models/master/research/slim/nets/inception_v1.py -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/
   
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
