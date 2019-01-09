#!/bin/bash

set -x

echo "### start to prepare header files for zoo-openvino"
git clone --single-branch --branch 2018_R3 https://github.com/opencv/dldt.git
mkdir include
mkdir -p ./include/common/samples
mkdir -p ./include/inference_engine
cp -r ./dldt/inference-engine/src/extension ./include
cp ./dldt/inference-engine/samples/common/samples/slog.hpp ./include/common/samples
cp -rf ./dldt/inference-engine/include/*  ./include/inference_engine
tar -xzvf ../../src/main/resources/inference.tar.gz -C ../classes