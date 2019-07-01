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

if (( $# < 3)); then
  echo "Usage: release.sh platform version upload mvn_parameters"
  echo "Usage example: bash release.sh linux default true"
  echo "Usage example: bash release.sh linux 0.6.0.dev0 true -Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.x"
  echo "If needed, you can also add other profiles such as: -Dspark.version=2.4.3 -Dbigdl.artifactId=bigdl-SPARK_2.4 -P spark_2.x"
  exit -1
fi

platform=$1
version=$2
upload=$3
profiles=${*:4}

if [ "${version}" != "default" ]; then
    echo "User specified version: ${version}"
    sed -i "s/\(__version__ =\)\(.*\)/\1 \"${version}\"/" $ANALYTICS_ZOO_PYTHON_DIR/zoo/__init__.py
fi
effect_version=`cat $ANALYTICS_ZOO_PYTHON_DIR/zoo/__init__.py | grep "__version__" | awk '{print $NF}' | tr -d '"'`
echo "The effective version is: ${effect_version}"

cd ${ANALYTICS_ZOO_HOME}
if [ "$platform" ==  "mac" ]; then
    echo "Building Analytics Zoo for mac system"
    dist_profile="-P mac -P without_bigdl $profiles"
    verbose_pname="macosx_10_11_x86_64"
elif [ "$platform" == "linux" ]; then
    echo "Building Analytics Zoo for linux system"
    dist_profile="-P linux -P without_bigdl $profiles"
    verbose_pname="manylinux1_x86_64"
else
    echo "Unsupported platform. Only linux and mac are supported for now."
fi

build_command="${ANALYTICS_ZOO_HOME}/make-dist.sh ${dist_profile}"
echo "Build command: ${build_command}"
$build_command

cd $ANALYTICS_ZOO_PYTHON_DIR
sdist_command="python setup.py sdist"
echo "Packing source code: ${sdist_command}"
$sdist_command

if [ -d "${ANALYTICS_ZOO_HOME}/pyzoo/dist" ]; then
   rm -r ${ANALYTICS_ZOO_HOME}/pyzoo/dist
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

created_whl="dist/analytics_zoo-${effect_version}-py2.py3-none-${verbose_pname}.whl"
echo "whl is created at: ${created_whl}"

if [ ${upload} == true ]; then
    upload_command="twine upload ${created_whl}"
    echo "Command for uploading to pypi: $upload_command"
    $upload_command
fi

