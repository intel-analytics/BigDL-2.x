#!/bin/bash

set -e

echo "#2 start example test for openvino"
#timer
start=$(date "+%s")
if [ -f analytics-zoo-models/faster_rcnn_resnet101_coco.xml ]; then
  echo "analytics-zoo-models/faster_rcnn_resnet101_coco already exists."
else
  wget -nv $FTP_URI/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco.xml \
    -P analytics-zoo-models
  wget -nv $FTP_URI/analytics-zoo-models/openvino/2018_R5/faster_rcnn_resnet101_coco.bin \
    -P analytics-zoo-models
fi
if [ -d analytics-zoo-data/data/object-detection-coco ]; then
  echo "analytics-zoo-data/data/object-detection-coco already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/data/object-detection-coco.zip -P analytics-zoo-data/data
  unzip -q analytics-zoo-data/data/object-detection-coco.zip -d analytics-zoo-data/data
fi
python ${BIGDL_ROOT}/python/orca/example/openvino/predict.py \
  --image analytics-zoo-data/data/object-detection-coco \
  --model analytics-zoo-models/faster_rcnn_resnet101_coco.xml
now=$(date "+%s")
time2=$((now - start))

echo "#2 openvino time used: $time2 seconds"



