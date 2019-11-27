#!/usr/bin/env bash

#
# Copyright 2018 Analytics Zoo Authors.
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


# Usage: bash get_movielens-1m.sh dir
# Download and unzip MovieLens-1M dataset to the target 'dir'.
# If 'dir' is not specified, it will be downloaded to the current working directory.

if [ ! -z "$1" ]
then
   DIR=$1
   cd "$DIR"
else
   DIR=$(dirname "$0")
   cd "$DIR"
fi

if [ -f "realworld.zip" ] || [ -d "realworld" ]
then
   echo "HiCS realworld dataset already exists."
   exit
fi

echo "Downloading realworld.zip"
wget https://www.ipd.kit.edu/~muellere/HiCS/realworld.zip

echo "Unzipping realworld.zip"
unzip -q realworld.zip

echo "Finished"
