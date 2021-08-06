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

# This is the script to release analytics-zoo for mac with different spark versions.

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR

if (( $# < 1)); then
  echo "Usage: release_mac.sh version spark_version bigdl_artifactId spark_profile"
  echo "Usage example: bash release_mac.sh default 2.4.6 bigdl-SPARK_2.4 spark_2.4+"
  echo "Usage example: bash release_mac.sh 0.12.0.dev0 2.4.6 bigdl-SPARK_2.4 spark_2.4+"
  exit -1
fi

version=$1
spark_version=$2
bigdl_artifactId=$3
spark_profile=$4

bash ${RUN_SCRIPT_DIR}/release.sh mac ${version} false true -Dspark.version=${spark_version} -Dbigdl.artifactId=${bigdl_artifactId} -P ${spark_profile}
