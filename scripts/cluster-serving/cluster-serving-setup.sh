#!/bin/bash

chmod a+x ./*

mv *-serving.jar zoo.jar
mv *-cluster-serving-python.zip serving-python.zip

export PYTHONPATH=$PYTHONPATH:$(pwd)/serving-python.zip

chmod a+x cluster-serving-*
export CS_PATH=$(pwd)
#export PATH=$PATH:$CS_PATH
cp cluster-serving-* /usr/local/bin/

echo "cluster serving environment is ready"
