#!/bin/bash

ssd_root=${HOME}/analytics-zoo/pipeline/ssd
data_root=${ssd_root}/data/pascal

# Convert Sequence File
if [ ! -d "${data_root}/seq/train" ]; then
  mkdir -p ${data_root}/seq/train
  mkdir -p ${data_root}/seq/test
fi

java -cp ${ssd_root}/target/pipeline-0.1-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.zoo.pipeline.common.dataset.RoiImageSeqGenerator \
     -f ${data_root}/VOCdevkit -o ${data_root}/seq/test -i voc_2007_test -p 28

rm ${data_root}/seq/test/.*crc
echo 'convert test data done'

java -cp ${ssd_root}/target/pipeline-0.1-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.zoo.pipeline.common.dataset.RoiImageSeqGenerator \
     -f ${data_root}/VOCdevkit -o ${data_root}/seq/train -i voc_0712_trainval -p 28

rm ${data_root}/seq/train/.*crc
echo 'convert train data done'