#!/bin/bash

set -x

echo "### start to prepare header files for zoo-openvino"
export version=2018_R3
wget https://github.com/opencv/dldt/archive/$version.tar.gz
tar -xzvf $version.tar.gz
mkdir include
mkdir -p ./include/common/samples
mkdir -p ./include/inference_engine
cp -r ./dldt-$version/inference-engine/src/extension ./include
cp ./dldt-$version/inference-engine/samples/common/samples/slog.hpp ./include/common/samples
cp -rf ./dldt-$version/inference-engine/include/*  ./include/inference_engine
tar -xzvf ../../src/main/resources/inference.tar.gz -C ../classes
cp -rf ./dldt-$version/model-optimizer ../classes/model_optimizer