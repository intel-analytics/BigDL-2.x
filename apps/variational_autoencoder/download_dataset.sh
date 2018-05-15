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


# Usage: bash download_dataset.sh dir
# Download dataset celeba to the target 'dir'.
# If 'dir' is not specified, it will be downloaded to the same dir with this script.

if [ ! -z "$1" ]
then
   DIR=$1
   cd "$DIR"
else
   DIR=$(dirname "$0")
   echo "Download path: $DIR"
   cd "$DIR"
fi

FILENAME="./img_align_celeba.zip"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
   exit
fi

echo "Downloading celeba images"
wget ftp://zoo:1234qwer@10.239.47.211/analytics-zoo-data/apps/variational_autoencoder/img_align_celeba.zip --no-host-directories 
unzip ./img_align_celeba.zip


echo "Finished"
