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


# Usage: bash get_ambient_temperature_system_failure.sh dir
# Download ambient_temperature_system_failure dataset to the target 'dir'.
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

FILENAME="./nyc_taxi.csv"
if [ -f "$FILENAME" ]
then
   echo "$FILENAME already exists."
   exit
fi

echo "Downloading nyc_taxi.csv"
wget https://analytics-zoo-data.s3.amazonaws.com/nyc_taxi.csv


echo "Finished"
