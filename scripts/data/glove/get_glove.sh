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


# Usage: bash get_glove.sh dir
# Download and unzip pre-trained glove word embeddings to the target 'dir'.
# If 'dir' is not specified, it will be downloaded to the current working directory.

if [ ! -z "$1" ]
then
   DIR=$1
   cd "$DIR"
fi

if [ -f "glove.6B.zip" ] || [ -d "glove.6B" ]
then
   echo "glove.6B already exists."
   exit
fi

echo "Downloading glove.6B.zip"
wget http://nlp.stanford.edu/data/glove.6B.zip

echo "Unzipping glove.6B.zip"
unzip -q glove.6B.zip -d glove.6B

echo "Finished"
