#!/usr/bin/env bash

#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# A shell script to stop all workers on a single slave
#
# Environment variables
#
#   SPARK_WORKER_INSTANCES The number of worker instances that should be
#                          running on this slave.  Default is 1.

# Usage: stop-slave.sh
#   Stops all slaves on this worker machine
function ht_enabled {
  ret=`lscpu |grep "Thread(s) per core"|awk '{print $4}'`
  if [ $ret -eq 1 ]; then
    false
  else
    true
  fi
}

if [ -z "${SPARK_HOME}" ]; then
  export SPARK_HOME="$(cd "`dirname "$0"`"/..; pwd)"
fi

. "${SPARK_HOME}/sbin/spark-config.sh"

. "${SPARK_HOME}/bin/load-spark-env.sh"

# Stop master
"${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.master.Master

_WORKER_NAME_NO=1

# Check if `numactl` exits
if type "numactl" > /dev/null 2>&1; then
  # Load NUMA configurations line-by-line and catch NUMA node no.
  IFS=$'\n'; for nnode in `numactl --hardware`; do
    if [[ ${nnode} =~ ^node\ ([0-9]+)\ cpus:\ .+$ ]]; then
      IFS=' ' _NUMA_CPUS=(${BASH_REMATCH[2]})
      _LENGTH=${#_NUMA_CPUS[@]}
      if ht_enabled; then _LENGTH=$((_LENGTH / 2)); fi
      # calculate worker num on this numa node, 12 ~ 23 core/worker
      _WORKER_NUM=$((_LENGTH / 12))
      if [[ $_WORKER_NUM -eq 0 ]]; then
        _WORKER_NUM=1
      fi
      "${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker "$_WORKER_NAME_NO"
      _WORKER_NAME_NO=$((_WORKER_NAME_NO + 1))
    fi
  done
else
  if [ "$SPARK_WORKER_INSTANCES" = "" ]; then
    "${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker 1
  else
    for ((i=0; i<$SPARK_WORKER_INSTANCES; i++)); do
      "${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker $(( $i + 1 ))
    done
  fi
fi
