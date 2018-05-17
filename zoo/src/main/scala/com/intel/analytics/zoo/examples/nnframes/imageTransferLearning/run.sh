#!/usr/bin/env bash

$SPARK_HOME/bin/spark-submit --master local[2] \
--class com.intel.analytics.zoo.examples.nnframes.imageTransferLearning.ImageTransferLearning \
--driver-memory 10g \
./dist/lib/analytics-zoo-0.1.0-SNAPSHOT-jar-with-dependencies.jar \
--modelPath /home/yuhao/workspace/model/caffe_Inception_imagenet/bvlc_googlenet.caffemodel \
--caffeDefPath /home/yuhao/workspace/model/caffe_Inception_imagenet/deploy.prototxt \
--folder /home/yuhao/workspace/data/dogs-vs-cats/demo
