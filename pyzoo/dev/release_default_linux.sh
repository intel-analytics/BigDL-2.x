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

# This is the default script with maven parameters to release analytics-zoo for linux.
# Note that if the maven parameters to build analytics-zoo need to be changed,
# make sure to change this file accordingly.
# If you want to customize the release, please use release.sh and specify maven parameters instead.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR

if (( $# < 1)); then
  echo "Usage: release_default_linux.sh version"
  echo "Usage example: bash release_default_linux.sh default"
  echo "Usage example: bash release_default_linux.sh 0.6.0.dev0"
  exit -1
fi

version=$1
upload=$2
spark_version=$3
bigdl_artifactId=$4
spark_profile=$5

bash ${RUN_SCRIPT_DIR}/release.sh linux ${version} false ${upload} -Dspark.version=${spark_version} -Dbigdl.artifactId=${bigdl_artifactId} -P ${spark_profile}
