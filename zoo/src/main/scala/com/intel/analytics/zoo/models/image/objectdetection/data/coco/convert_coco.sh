#!/bin/bash

export ANALYTICS_ZOO_JAR=`find ${ANALYTICS_ZOO_HOME}/lib -type f -name "analytics-zoo*jar-with-dependencies.jar"`

if [ -z "${data_root}" ]; then
    echo "Please set data_root environment variable"
    exit 1
fi

# Convert Sequence File
if [ ! -d "${data_root}/seq/train" ]; then
  mkdir -p ${data_root}/seq/train
  mkdir -p ${data_root}/seq/test
fi

java -cp ${ANALYTICS_ZOO_JAR} \
 com.intel.analytics.zoo.models.image.objectdetection.common.dataset.RoiImageSeqGenerator \
 -i coco_minival -f ${data_root} -o ${data_root}/seq/coco-minival -p 28

rm ${data_root}/seq/coco-minival/.*crc
echo 'convert minival data done'

java -cp ${ANALYTICS_ZOO_JAR} \
 com.intel.analytics.zoo.models.image.objectdetection.common.dataset.RoiImageSeqGenerator \
 -i coco-testdev -f ${data_root} -o ${data_root}/seq/coco-testdev -p 28

rm ${data_root}/seq/coco-testdev/.*crc
echo 'convert testdev data done'

java -cp ${ANALYTICS_ZOO_JAR} \
     com.intel.analytics.zoo.models.image.objectdetection.common.dataset.RoiImageSeqGenerator \
      -i voc_0712_trainval -f ${data_root} -o ${data_root}/seq/train -p 28

rm ${data_root}/seq/train/.*crc
echo 'convert train data done'