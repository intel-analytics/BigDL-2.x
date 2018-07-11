#!/bin/bash

ssd_root=${HOME}/analytics-zoo/pipeline/ssd
data_root=${ssd_root}/data/coco

# Convert Sequence File
if [ ! -d "${data_root}/seq/train" ]; then
  mkdir -p ${data_root}/seq/train
  mkdir -p ${data_root}/seq/test
fi

java -cp ${ssd_root}/dist/target/object-detection-0.1-SNAPSHOT-jar-with-dependencies-and-spark.jar \
 com.intel.analytics.zoo.pipeline.common.dataset.RoiImageSeqGenerator \
 -i coco_minival -f ${data_root} -o ${data_root}/seq/coco-minival -p 28

rm ${data_root}/seq/coco-minival/.*crc
echo 'convert minival data done'

java -cp ${ssd_root}/dist/target/object-detection-0.1-SNAPSHOT-jar-with-dependencies-and-spark.jar \
 com.intel.analytics.zoo.pipeline.common.dataset.RoiImageSeqGenerator \
 -i coco-testdev -f ${data_root} -o ${data_root}/seq/coco-testdev -p 28

rm ${data_root}/seq/coco-testdev/.*crc
echo 'convert testdev data done'

java -cp ${ssd_root}/dist/target/object-detection-0.1-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.zoo.pipeline.common.dataset.RoiImageSeqGenerator \
      -i voc_0712_trainval -f ${data_root} -o ${data_root}/seq/train -p 28

rm ${data_root}/seq/train/.*crc
echo 'convert train data done'