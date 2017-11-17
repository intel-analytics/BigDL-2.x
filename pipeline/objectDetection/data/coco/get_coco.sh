#!/bin/bash

ssd_root=${HOME}/analytics/pipeline/ssd
data_root=${ssd_root}/data/coco

# Download the data.
cd $data_root

## Get images
wget http://msvocds.blob.core.windows.net/coco2014/train2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/val2014.zip
wget http://msvocds.blob.core.windows.net/coco2014/test2014.zip
wget http://msvocds.blob.core.windows.net/coco2015/test2015.zip

## Get annotations
wget http://msvocds.blob.core.windows.net/annotations-1-0-3/instances_train-val2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2014.zip
wget http://msvocds.blob.core.windows.net/annotations-1-0-4/image_info_test2015.zip
wget https://s3-ap-southeast-1.amazonaws.com/bigdl-models/ssd/instances_minival2014.json.zip
wget https://s3-ap-southeast-1.amazonaws.com/bigdl-models/ssd/instances_valminusminival2014.json.zip

# Extract the data.
unzip train2014.zip
unzip val2014.zip
unzip test2014.zip
unzip test2015.zip
unzip instances_train-val2014.zip
unzip image_info_test2014.zip
unzip image_info_test2015.zip
unzip instances_minival2014.json.zip
unzip instances_valminusminival2014.json.zip

mv instances_minival2014.json annotations/
mv instances_valminusminival2014.json annotations/