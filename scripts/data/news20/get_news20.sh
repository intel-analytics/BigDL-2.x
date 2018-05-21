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


# Usage: bash get_news20.sh dir
# Download and unzip 20 Newsgroup dataset to the target 'dir'.
# If 'dir' is not specified, it will be downloaded to the current working directory.

if [ ! -z "$1" ]
then
   DIR=$1
   cd "$DIR"
fi

if [ -f "20news-18828.tar.gz" ] || [ -d "20news-18828" ]
then
   echo "20 Newsgroup dataset already exists."
   exit
fi

echo "Downloading news20.tar.gz"
wget http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz

echo "Unzipping news20.tar.gz"
tar zxf 20news-18828.tar.gz

echo "Finished"
