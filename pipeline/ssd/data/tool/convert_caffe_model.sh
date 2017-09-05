#!/bin/bash

caffePrototxt=$1
caffeModel=$2
bigDLModel=$3

java -cp target/pipeline-0.1-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.zoo.pipeline.common.CaffeConverter \
     --caffeDefPath $caffePrototxt --caffeModelPath $caffeModel -o $bigDLModel