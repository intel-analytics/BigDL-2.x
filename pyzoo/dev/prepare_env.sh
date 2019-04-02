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

SCRIPT_DIR=$(dirname ${BASH_SOURCE[0]})
echo "SCRIPT_DIR": $SCRIPT_DIR
export DL_PYTHON_HOME="$(cd ${SCRIPT_DIR}/../; pwd)"

export ANALYTICS_ZOO_ROOT="$(cd ${SCRIPT_DIR}/../..; pwd)"

echo "ANALYTICS_ZOO_ROOT:" $ANALYTICS_ZOO_ROOT
echo "SPARK_HOME:" $SPARK_HOME
echo "DL_PYTHON_HOME:" $DL_PYTHON_HOME

if [ -z ${SPARK_HOME+x} ]; then echo "SPARK_HOME is unset"; exit 1; else echo "SPARK_HOME is set to '$SPARK_HOME'"; fi

export PYSPARK_ZIP=`find $SPARK_HOME/python/lib  -type f -iname '*.zip' | tr "\n" ":"`

export PYTHONPATH=$PYSPARK_ZIP:$DL_PYTHON_HOME:$ANALYTICS_ZOO_ROOT/zoo/python_packages/sources/:$ANALYTICS_ZOO_ROOT/zoo/target/classes/spark-analytics-zoo.conf:$ANALYTICS_ZOO_ROOT/zoo/target/extra-resources/zoo-version-info.properties:$PYTHONPATH
echo "PYTHONPATH": $PYTHONPATH
export ANALYTICS_ZOO_CLASSPATH=$(find $ANALYTICS_ZOO_ROOT/zoo/target/ -name "*with-dependencies.jar" | head -n 1)
echo "ANALYTICS_ZOO_CLASSPATH": $ANALYTICS_ZOO_CLASSPATH

export BIGDL_CLASSPATH=$ANALYTICS_ZOO_CLASSPATH
echo "BIGDL_CLASSPATH": $BIGDL_CLASSPATH

export KMP_BLOCKTIME=0
export KMP_AFFINITY=granularity=fine,verbose,compact,1,0
export KMP_SETTINGS=1
export OMP_NUM_THREADS=1
