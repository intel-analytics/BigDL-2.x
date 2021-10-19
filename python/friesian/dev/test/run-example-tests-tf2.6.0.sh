#!/usr/bin/env bash

#
# Copyright 2016 The BigDL Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

export FTP_URI=$FTP_URI

set -e
mkdir -p result
mkdir -p result/stats
echo "#1 start example test for two tower train"

#timer
start=$(date "+%s")
if [ -d data/input_2tower ]; then
  echo "data/input_2tower already exists"
else
  wget -nv $FTP_URI/analytics-zoo-data/input_2tower.tar.gz -P data
  tar -xvzf data/input_2tower.tar.gz -C data
fi

python ../../example/two_tower/train_2tower.py \
    --executor_cores 4 \
    --executor_memory 50g \
    --data_dir ./data/input_2tower \
    --model_dir ./result \
    --frequency_limit 2

now=$(date "+%s")
time1=$((now - start))

echo "#1 two tower train time used: $time1 seconds"
