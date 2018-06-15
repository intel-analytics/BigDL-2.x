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
export ANALYTICS_ZOO_HOME="$(cd ${RUN_SCRIPT_DIR}/../../../; pwd)"
echo $ANALYTICS_ZOO_HOME
ANALYTICS_ZOO_PYTHON_DIR="$(cd ${RUN_SCRIPT_DIR}/../../../pyzoo; pwd)"
echo $ANALYTICS_ZOO_PYTHON_DIR

if (( $# < 2)); then
  echo "Bad parameters. Usage example: bash release.sh linux without_bigdl false 0.1.0.dev2"
  exit -1
fi

platform=$1
spark_profile=$2
quick=$3
input_version=$4
analytics_zoo_version=$(python -c "exec(open('$ANALYTICS_ZOO_HOME/pyzoo/zoo/version.py').read()); print(__version__)")

if [ "$input_version" != "$analytics_zoo_version" ]; then
   echo "Your Analytics Zoo version $analytics_zoo_version is not the proposed version $input_version!"
   exit -1
fi

cd ${ANALYTICS_ZOO_HOME}
if [ "$platform" ==  "mac" ]; then
    echo "Building Analytics Zoo for mac system"
    dist_profile="-P mac -P $spark_profile"
    verbose_pname="macosx_10_11_x86_64"
elif [ "$platform" == "linux" ]; then
    echo "Building Analytics Zoo for linux system"
    dist_profile="-P $spark_profile"
    verbose_pname="manylinux1_x86_64"
else
    echo "unsupport platform"
fi

analytics_zoo_build_command="${ANALYTICS_ZOO_HOME}/make-dist.sh ${dist_profile}"
if [ "$quick" == "true" ]; then
    echo "Skip disting Analytics Zoo"
else
    echo "Dist Analytics Zoo: $analytics_zoo_build_command"
    $analytics_zoo_build_command
fi

cd $ANALYTICS_ZOO_PYTHON_DIR
sdist_command="python setup.py sdist"
echo "packing source code: ${sdist_command}"
$sdist_command

if [ -d "${ANALYTICS_ZOO_DIR}/pyzoo/dist" ]; then
   rm -r ${ANALYTICS_ZOO_DIR}/pyzoo/dist
fi

wheel_command="python setup.py bdist_wheel --universal --plat-name ${verbose_pname}"
echo "Packing python distribution:   $wheel_command"
${wheel_command}

if [ -d "${ANALYTICS_ZOO_HOME}/pyzoo/build" ]; then
   echo "Removing pyzoo/build"
   rm -r ${ANALYTICS_ZOO_HOME}/pyzoo/build
fi

if [ -d "${ANALYTICS_ZOO_HOME}/pyzoo/analytics_zoo.egg-info" ]; then
   echo "Removing pyzoo/analytics_zoo.egg-info"
   rm -r ${ANALYTICS_ZOO_HOME}/pyzoo/analytics_zoo.egg-info
fi

upload_command="twine upload dist/analytics_zoo-${analytics_zoo_version}-py2.py3-none-${verbose_pname}.whl"
echo "Please manually upload with this command:  $upload_command"

$upload_command
