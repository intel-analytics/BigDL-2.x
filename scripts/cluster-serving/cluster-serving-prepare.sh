#!/bin/bash

chmod a+x ./*

export PYTHONPATH=$PYTHONPATH:./analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.9.0-SNAPSHOT-cluster-serving-python.zip
mv analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.9.0-SNAPSHOT-serving.jar zoo.jar

chmod a+x cluster-serving-*
export CS_PATH=$(pwd)
#export PATH=$PATH:$CS_PATH
cp cluster-serving-* /usr/local/bin/ 

echo "cluster serving environment is ready"
