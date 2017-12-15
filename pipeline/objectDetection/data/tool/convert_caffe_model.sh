#!/bin/bash

caffePrototxt=$1
caffeModel=$2
bigDLModel=$3
type=$4
name=$5
quantize=$6

java -Xmx16g -cp dist/target/object-detection-0.1-SNAPSHOT-jar-with-dependencies-and-spark.jar \
         com.intel.analytics.zoo.pipeline.common.CaffeConverter \
     --caffeDefPath $caffePrototxt --caffeModelPath $caffeModel -o $bigDLModel -t $type --name $name -q $quantize