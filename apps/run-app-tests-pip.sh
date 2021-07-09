#!/usr/bin/env bash

export ANALYTICS_ZOO_HOME=${ANALYTICS_ZOO_ROOT}/dist

clear_up () {
    echo "Clearing up environment. Uninstalling analytics-zoo"
    pip uninstall -y analytics-zoo
    pip uninstall -y bigdl
    pip uninstall -y pyspark
}

chmod +x ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh

set -e

RUN_PART1=0
RUN_PART2=0
RUN_PART3=0
RUN_PART4=0
if [ $1 = 1 ]; then
	RUN_PART1=1
	RUN_PART2=0
	RUN_PART3=0
	RUN_PART4=0
elif [ $1 = 2 ]; then
	RUN_PART1=0
	RUN_PART2=1
	RUN_PART3=0
	RUN_PART4=0
elif [ $1 = 3 ]; then
	RUN_PART1=0
	RUN_PART2=0
	RUN_PART3=1
	RUN_PART4=0
elif [ $1 = 4 ]; then
	RUN_PART1=0
	RUN_PART2=0
	RUN_PART3=0
	RUN_PART4=1
else
	RUN_PART1=1
	RUN_PART2=1
	RUN_PART3=1
	RUN_PART4=1
fi


if [ $RUN_PART1 = 1 ]; then
echo "#1 start app test for anomaly-detection-nyc-taxi"
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
echo "#1 anomaly-detection-nyc-taxi time used:$time1 seconds"

echo "#2 start app test for object-detection"
start=$(date "+%s")

# Conversion to py file and data preparation
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
    wget https://sourceforge.net/projects/analytics-zoo/files/analytics-zoo-models/object-detection/analytics-zoo_ssd-mobilenet-300x300_PASCAL_0.1.0.model -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
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
    wget $FTP_URI/analytics-zoo-data/apps/object-detection/train_dog.mp4  -P ${ANALYTICS_ZOO_HOME}/apps/object-detection/
fi
FILENAME="~/.imageio/ffmpeg/ffmpeg-linux64-v3.3.1"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-data/apps/object-detection/ffmpeg-linux64-v3.3.1 -P ~/.imageio/ffmpeg/
fi

# Run the example
export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/object-detection/object-detection.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "object-detection failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time2=$((now-start))
echo "#2 object-detection time used:$time2 seconds"

echo "#3 start app test for image-similarity"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-similarity/image-similarity
sed "s/setBatchSize(20)/setBatchSize(56)/g;s/setMaxEpoch(2)/setMaxEpoch(1)/g;s%/tmp/images%${ANALYTICS_ZOO_HOME}/apps/image-similarity%g;s%imageClassification%miniimageClassification%g;s%/googlenet_places365/deploy.prototxt%/googlenet_places365/deploy_googlenet_places365.prototxt%g;s%/vgg_16_places365/deploy.prototxt%/vgg_16_places365/deploy_vgg16_places365.prototxt%g;s%./samples%${ANALYTICS_ZOO_HOME}/apps/image-similarity/samples%g" ${ANALYTICS_ZOO_HOME}/apps/image-similarity/image-similarity.py >${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/miniimageClassification.tar.gz"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
   tar -zxvf ${ANALYTICS_ZOO_HOME}/apps/image-similarity/miniimageClassification.tar.gz -C ${ANALYTICS_ZOO_HOME}/apps/image-similarity
else
   echo "Downloading images"

   wget $FTP_URI/analytics-zoo-data/miniimageClassification.tar.gz -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity
   tar -zxvf ${ANALYTICS_ZOO_HOME}/apps/image-similarity/miniimageClassification.tar.gz -C ${ANALYTICS_ZOO_HOME}/apps/image-similarity

   echo "Finished downloading images"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365/deploy_googlenet_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 deploy model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/deploy_googlenet_places365.prototxt -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365

   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365/googlenet_places365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading places365 weight model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/googlenet_places365.caffemodel -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/googlenet_places365

   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365/deploy_vgg16_places365.prototxt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG deploy model"

   wget $FTP_URI/analytics-zoo-models/image-similarity/deploy_vgg16_places365.prototxt -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365

   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365/vgg16_places365.caffemodel"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG weight model"

   wget $FTP_URI/analytics-zoo-models/image-classification/vgg16_places365.caffemodel  -P ${ANALYTICS_ZOO_HOME}/apps/image-similarity/vgg_16_places365

   echo "Finished downloading model"
fi

# Run the example
export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/image-similarity/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "image-similarity failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time3=$((now-start))
echo "#3 image-similarity time used:$time3 seconds"

echo "#4 start app test for using_variational_autoencoder_to_generate_digital_numbers"
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
time4=$((now-start))
echo "#4 using_variational_autoencoder_to_generate_digital_numbers time used:$time4 seconds"

fi

if [ $RUN_PART2 = 1 ]; then
echo "#5 start app test for image-augmentation"
# timer
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation

# Run the example
export SPARK_DRIVER_MEMORY=1g
python ${ANALYTICS_ZOO_HOME}/apps/image-augmentation/image-augmentation.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "image-augmentation failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time5=$((now-start))
echo "#5 image-augmentation time used:$time5 seconds"

echo "#6 start app test for dogs-vs-cats"
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
time6=$((now-start))
echo "#6 dogs-vs-cats time used:$time6 seconds"

echo "#7 start app test for image-augmentation-3d"
# timer
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augmentation-3d

# Run the example
export SPARK_DRIVER_MEMORY=1g
python ${ANALYTICS_ZOO_HOME}/apps/image-augmentation-3d/image-augmentation-3d.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "image-augmentation-3d failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time7=$((now-start))
echo "#7 image-augmentation-3d time used:$time7 seconds"

echo "#8 start app test for image_classification_inference"
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
    wget $FTP_URI/analytics-zoo-models/image-classification/inception_utils.py -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/
    wget $FTP_URI/analytics-zoo-models/image-classification/inception_v1.py -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/models/research/slim/nets/

   echo "Finished downloading model"
fi
FILENAME="${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint/inception_v1.ckpt"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading inception_v1 checkpoint"

   wget $FTP_URI/analytics-zoo-models/image-classification/inception_v1_2016_08_28.tar.gz -P ${ANALYTICS_ZOO_HOME}/apps/tfnet/checkpoint
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
time8=$((now-start))
rm ${ANALYTICS_ZOO_HOME}/apps/tfnet/tmp.py
echo "#8 image_classification_inference time used:$time8 seconds"


echo "#9 start app test for using_variational_autoencoder_to_generate_faces"
#timer
start=$(date "+%s")
 ${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces
 sed -i "s/data_files\[\:100000\]/data_files\[\:500\]/g; s/batch_size=batch_size/batch_size=100/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi
 export SPARK_DRIVER_MEMORY=200g
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
time9=$((now-start))
echo "#9 using_variational_autoencoder_to_generate_faces time used:$time9 seconds"

fi


if [ $RUN_PART3 = 1 ]; then
echo "#10 start app test for using_variational_autoencoder_and_deep_feature_loss_to_generate_faces"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces
 sed -i "s/data_files\[\:100000\]/data_files\[\:500\]/g; s/batch_size=batch_size/batch_size=100/g" ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/using_variational_autoencoder_and_deep_feature_loss_to_generate_faces.py
FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading VGG model"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/analytics-zoo_vgg-16_imagenet_0.1.0.model --no-host-directories
   echo "Finished"
fi
 FILENAME="${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading celeba images"
   wget -P ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ $FTP_URI/analytics-zoo-data/apps/variational-autoencoder/img_align_celeba.zip --no-host-directories
   unzip -d ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/ ${ANALYTICS_ZOO_HOME}/apps/variational-autoencoder/img_align_celeba.zip
   echo "Finished"
fi
 export SPARK_DRIVER_MEMORY=200g
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
time10=$((now-start))
echo "#10 using_variational_autoencoder_and_deep_feature_loss_to_generate_faces time used:$time10 seconds"

echo "#11 start app test for recommendation-ncf"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/ncf-explicit-feedback
sed "s/end_trigger=MaxEpoch(10)/end_trigger=MaxEpoch(5)/g; s%sc.parallelize(movielens_data)%sc.parallelize(movielens_data[0:50000:])%g" ${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/ncf-explicit-feedback.py >${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/tmp.py

# Run the example
export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/recommendation-ncf/tmp.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "recommendation-ncf failed"
    exit $exit_status
fi

unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time11=$((now-start))
echo "#11 recommendation-ncf time used:$time11 seconds"

echo "#12 start app test for recommendation-wide-n-deep"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/wide_n_deep
sed "s/end_trigger=MaxEpoch(10)/end_trigger=MaxEpoch(5)/g; s/batch_size = 8000/batch_size = 8008/g" ${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/wide_n_deep.py >${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/tmp_test.py

# Run the example
export SPARK_DRIVER_MEMORY=22g
python ${ANALYTICS_ZOO_HOME}/apps/recommendation-wide-n-deep/tmp_test.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "recommendation-wide-n-deep failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time12=$((now-start))
echo "#12 recommendation-wide-n-deep time used:$time12 seconds"

echo "#13 start app test for sentiment-analysis"
start=$(date "+%s")

# Conversion to py file and data preparation
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment
sed "s/batch_size = 64/batch_size = 84/g" ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/sentiment.py >${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/tmp_test.py
FILENAME="/tmp/.bigdl/dataset/glove.6B.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading glove6B"
   wget -P /tmp/.bigdl/dataset/ $FTP_URI/analytics-zoo-data/data/glove/glove.6B.zip
   echo "Finished"
fi

# Run the example
export SPARK_DRIVER_MEMORY=12g
python ${ANALYTICS_ZOO_HOME}/apps/sentiment-analysis/tmp_test.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "sentiment-analysis failed"
    exit $exit_status
fi
unset SPARK_DRIVER_MEMORY
now=$(date "+%s")
time13=$((now-start))
echo "#13 sentiment-analysis time used:$time13 seconds"

echo "#14 start app test for anomaly-detection-hd"
#timer
start=$(date "+%s")
FILENAME="${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/realworld.zip"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists"
else
    wget $FTP_URI/analytics-zoo-data/data/HiCS/realworld.zip  -P ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd
fi
dataPath="${ANALYTICS_ZOO_HOME}/bin/data/HiCS/"
rm -rf "$dataPath"
unzip -d ${ANALYTICS_ZOO_HOME}/bin/data/HiCS/  ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/realworld.zip
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo
sed -i '/get_ipython()/d' ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
sed -i '127,273d' ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
python ${ANALYTICS_ZOO_HOME}/apps/anomaly-detection-hd/autoencoder-zoo.py
exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "anomaly-detection-hd failed"
    exit $exit_status
fi
now=$(date "+%s")
time14=$((now-start))
echo "#14 anomaly-detection-hd time used:$time14 seconds"

#echo "#15 start app test for pytorch face-generation"
##timer
#start=$(date "+%s")
#${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation
#sed -i '/get_ipython()/d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
#sed -i '/plt./d' ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
#python ${ANALYTICS_ZOO_HOME}/apps/pytorch/face_generation.py
#exit_status=$?
#if [ $exit_status -ne 0 ];
#then
#    clear_up
#    echo "pytorch face-generation failed"
#    exit $exit_status
#fi
#now=$(date "+%s")
#time15=$((now-start))
#echo "#15 pytorch face-generation time used:$time15 seconds"

fi


if [ $RUN_PART4 = 1 ]; then
echo "#16 start app test for ray paramater-server"
#timer
start=$(date "+%s")

${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/ray/parameter_server/sharded_parameter_server
python ${ANALYTICS_ZOO_HOME}/apps/ray/parameter_server/sharded_parameter_server.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "ray paramater-server failed"
    exit $exit_status
fi
now=$(date "+%s")
time16=$((now-start))

echo "#16 ray paramater-server time used:$time16 seconds"

echo "#17 start app test for chronos-network-traffic-autots-forecasting"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting

FILENAME="${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading network traffic data"

   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data

   echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting.py
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-autots-forecasting failed"
    exit $exit_status
fi
now=$(date "+%s")
time17=$((now-start))
echo "#17 chronos-network-traffic-autots-forecasting time used:$time17 seconds"

echo "#18 start app test for chronos-network-traffic-model-forecasting"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting

FILENAME="${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading network traffic data"

   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data

   echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.py
sed -i "s/epochs=20/epochs=2/g; s/epochs=10/epochs=2/g; s/epochs=50/epochs=2/g" ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.py
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_model_forecasting.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-model-forecasting failed"
    exit $exit_status
fi
now=$(date "+%s")
time18=$((now-start))
echo "#18 chronos-network-traffic-model-forecasting time used:$time18 seconds"

echo "#19 start app test for automl-nyc-taxi"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/apps/automl/nyc_taxi_dataset

chmod +x ${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh
${ANALYTICS_ZOO_HOME}/bin/data/NAB/nyc_taxi/get_nyc_taxi.sh

sed -i '/get_ipython()/d;' ${ANALYTICS_ZOO_HOME}/apps/automl/nyc_taxi_dataset.py

python ${ANALYTICS_ZOO_HOME}/apps/automl/nyc_taxi_dataset.py

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "automl nyc-taxi failed"
    exit $exit_status
fi
now=$(date "+%s")
time19=$((now-start))
echo "#19 automl-nyc-taxi time used:$time19 seconds"

echo "#20 start app test for chronos-anomaly-detect-unsupervised-forecast-based"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based

wget $FTP_URI/analytics-zoo-data/chronos-aiops/m_1932.csv -O ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/m_1932.csv
echo "Finished downloading AIOps data"
#FILENAME="${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/m_1932.csv"
#if [ -f "$FILENAME" ]
#then
#   echo "$FILENAME already exists."
#else
#   echo "Downloading AIOps data"
#
#   wget $FTP_URI/analytics-zoo-data/chronos-aiops/m_1932.csv -P ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps
#
#   echo "Finished downloading AIOps data"
#fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py
sed -i "s/epochs=20/epochs=2/g" ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised_forecast_based.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-anomaly-detect-unsupervised-forecast-based failed"
    exit $exit_status
fi
now=$(date "+%s")
time20=$((now-start))
echo "#20 chronos-anomaly-detect-unsupervised-forecast-based time used:$time20 seconds"

echo "#21 start app test for chronos-anomaly-detect-unsupervised"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised

wget $FTP_URI/analytics-zoo-data/chronos-aiops/m_1932.csv -O ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/m_1932.csv
echo "Finished downloading AIOps data"
#FILENAME="${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/m_1932.csv"
#if [ -f "$FILENAME" ]
#then
#   echo "$FILENAME already exists."
#else
#   echo "Downloading AIOps data"
#
#   wget $FTP_URI/analytics-zoo-data/chronos-aiops/m_1932.csv -P ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps
#
#   echo "Finished downloading AIOps data"
#fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.py
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/AIOps/AIOps_anomaly_detect_unsupervised.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-anomaly-detect-unsupervised failed"
    exit $exit_status
fi
now=$(date "+%s")
time21=$((now-start))
echo "#21 chronos-anomaly-detect-unsupervised time used:$time21 seconds"

echo "#22 start app test for chronos-network-traffic-impute"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_impute

FILENAME="${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading network traffic data"

   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data

   echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_impute.py
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_impute.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-network-traffic-impute failed"
    exit $exit_status
fi
now=$(date "+%s")
time22=$((now-start))
echo "#22 chronos-network-traffic-impute time used:$time22 seconds"

echo "#23 start app test for chronos-stock-prediction"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/stock_prediction

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/stock_prediction.py
sed -i "s/epochs\ =\ 50/epochs\ =\ 2/g; s/batch_size\ =\ 16/batch_size\ =\ 1024/g" ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/stock_prediction.py
cwd=$PWD
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/

# download data
if [ -d "data" ]
then
    echo "data already exists"
else
    echo "Downloading stock prediction data"

    mkdir data
    cd data
    wget https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip
    wget https://raw.githubusercontent.com/CNuge/kaggle-code/master/stock_data/merge.sh
    chmod +x merge.sh
    unzip individual_stocks_5yr.zip
    ./merge.sh
    cd ..

    echo "Finish downloading stock prediction data"
fi

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/stock_prediction.py
cd $cwd

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-stock-prediction failed"
    exit $exit_status
fi
now=$(date "+%s")
time23=$((now-start))
echo "#23 chronos-stock-prediction time used:$time23 seconds"

echo "#24 start app test for chronos-network-traffic-multivarite-multistep-tcnforecaster"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster

FILENAME="${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
    echo "$FILENAME already exists."
else
    echo "Downloading network traffic data"

    wget $FTP_URI/analytics-zoo-data/network_traffic/data/data.csv -P ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data

    echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py
sed -i "s/epochs=20/epochs=2/g" ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_multivariate_multistep_tcnforecaster.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-multivariate-multistep-tcnforecaster failed"
    exit $exit_status
fi

now=$(date "+%s")
time24=$((now-start))
echo "#24 chronos-network-traffic-multivarite-multistep-tcnforecaster time used:$time24 seconds"

echo "#25 start app test for chronos-stock-prediction-prophet"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/stock_prediction_prophet

sed -i '/get_ipython()/d; /plot./d; /plt./d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/stock_prediction_prophet.py
sed -i "s/epochs\ =\ 50/epochs\ =\ 2/g; s/batch_size\ =\ 16/batch_size\ =\ 1024/g" ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/stock_prediction_prophet.py
cwd=$PWD
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/

# download data
if [ -d "data" ]
then
    echo "data already exists"
else
    echo "Downloading stock prediction data"

    mkdir data
    cd data
    wget https://github.com/CNuge/kaggle-code/raw/master/stock_data/individual_stocks_5yr.zip
    wget https://raw.githubusercontent.com/CNuge/kaggle-code/master/stock_data/merge.sh
    chmod +x merge.sh
    unzip individual_stocks_5yr.zip
    ./merge.sh
    cd ..

    echo "Finish downloading stock prediction data"
fi

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/fsi/stock_prediction_prophet.py
cd $cwd

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos-stock-prediction-prophet failed"
    exit $exit_status
fi
now=$(date "+%s")
time25=$((now-start))
echo "#25 chronos-stock-prediction-prophet time used:$time25 seconds"

echo "#26 start app test for chronos-network-traffic-autots-forecasting-experimental"
#timer
start=$(date "+%s")
${ANALYTICS_ZOO_HOME}/apps/ipynb2py.sh ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting_experimental

FILENAME="${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data/data.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
else
   echo "Downloading network traffic data"

   wget $FTP_URI/analytics-zoo-data/network-traffic/data/data.csv -P ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/data

   echo "Finished downloading network traffic data"
fi

sed -i '/get_ipython()/d; /plot[.]/d; /plt[.]/d; /axs[.]/d' ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting_experimental.py
sed -i "s/cores=10/cores=4/g; s/epochs=5/epochs=1/g; s/n_sampling=4/n_sampling=1/g" ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting_experimental.py
cd ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/

python ${ANALYTICS_ZOO_HOME}/../pyzoo/zoo/chronos/use-case/network_traffic/network_traffic_autots_forecasting_experimental.py
cd -

exit_status=$?
if [ $exit_status -ne 0 ];
then
    clear_up
    echo "chronos network-traffic-autots-forecasting-experimental failed"
    exit $exit_status
fi
now=$(date "+%s")
time26=$((now-start))
echo "#26 chronos-network-traffic-autots-forecasting-experimental time used:$time26 seconds"

fi

# This should be done at the very end after all tests finish.
clear_up
