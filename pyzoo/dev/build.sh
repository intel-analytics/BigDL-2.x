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

set -e
RUN_SCRIPT_DIR=$(cd $(dirname $0) ; pwd)
echo $RUN_SCRIPT_DIR

if (( $# < 2)); then
  echo "Usage: build.sh platform version mvn_parameters"
  echo "Usage example: bash release.sh linux default"
  echo "Usage example: bash release.sh linux 0.6.0.dev0"
  echo "If needed, you can also add other profiles such as: -Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.x"
  exit -1
fi

platform=$1
version=$2
profiles=${*:3}

bash ${RUN_SCRIPT_DIR}/release.sh ${platform} ${version} false ${profiles}
