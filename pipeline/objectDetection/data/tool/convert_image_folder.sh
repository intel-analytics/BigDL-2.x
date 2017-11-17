#!/bin/bash

imageFolder=$1
output=$2

java -cp target/pipeline-0.1-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.zoo.pipeline.common.dataset.RoiImageSeqGenerator \
     -f $imageFolder -o $output