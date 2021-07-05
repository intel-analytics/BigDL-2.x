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

# This is the default script with maven parameters to release analytics-zoo for mac.
# Note that if the maven parameters to build analytics-zoo need to be changed,
# make sure to change this file accordingly.
# If you want to customize the release, please use release.sh and specify maven parameters instead.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR

if (( $# < 1)); then
  echo "Usage: release_default_mac.sh version"
  echo "Usage example: bash release_default_mac.sh default"
  echo "Usage example: bash release_default_mac.sh 0.6.0.dev0"
  exit -1
fi

version=$1

bash ${RUN_SCRIPT_DIR}/release.sh mac ${version} false true -Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.4+
