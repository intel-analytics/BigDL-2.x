#!/bin/bash

chmod a+x ./*

export PYTHONPATH=$PYTHONPATH:./analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.9.0-SNAPSHOT-python-api.zip
mv analytics-zoo-bigdl_0.10.0-spark_2.4.3-0.9.0-SNAPSHOT-serving.jar zoo.jar

chmod a+x cluster-serving-*
export CS_PATH=$(pwd)
export PATH=$PATH:$CS_PATH

echo "cluster serving environment is ready"
