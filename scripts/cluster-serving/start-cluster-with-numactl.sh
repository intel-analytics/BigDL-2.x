#!/usr/bin/env bash
set -x

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

# This script loads NUMA configurations if `numactl` exists, and
# it might start multiple slaves on this machine for NUMA awareness.
#
# Environment Variables
#
#   SPARK_WORKER_INSTANCES  The number of worker instances to run on this
#                           slave.  Default is 1.
#   SPARK_WORKER_PORT       The base port number for the first worker. If set,
#                           subsequent workers will increment this number.  If
#                           unset, Spark will find a valid port number, but
#                           with no guarantee of a predictable pattern.
#   SPARK_WORKER_WEBUI_PORT The base port for the web interface of the first
#                           worker.  Subsequent workers will increment this
#                           number.  Default is 8081.

function ht_enabled {
  #ret=`grep -o '^flags\b.*: .*\bht\b' /proc/cpuinfo | tail -1`
  #if [ x"${ret}" == "x" ]; then
  #   false
  #else
  #  true
  #fi
  ret=`lscpu |grep "Thread(s) per core"|awk '{print $4}'`
  if [ $ret -eq 1 ]; then
    false
  else
    true
  fi
}

TOTAL_CORE_NUM=`nproc`
if ht_enabled; then
  TOTAL_CORE_NUM=$((TOTAL_CORE_NUM / 2))
fi

if [ $TOTAL_CORE_NUM -lt 24 ]; then
  # use local mode
  # TODO
  echo "local mode"
  exit 1
fi

if [ -z "${SPARK_HOME}" ]; then
  export SPARK_HOME="$(cd "`dirname "$0"`"/..; pwd)"
fi

. "${SPARK_HOME}/sbin/spark-config.sh"

. "${SPARK_HOME}/bin/load-spark-env.sh"

if [ -z "${SPARK_MASTER_HOST}" ]; then
  export SPARK_MASTER_HOST=`hostname`
fi

if [ -z "${SPARK_MASTER_PORT}" ]; then
  export SPARK_MASTER_PORT=7077
fi

if [ -z "${SPARK_MASTER_WEBUI_PORT}" ]; then
  export SPARK_MASTER_WEBUI_PORT=8080
fi

grep_port=`netstat -tlpn | awk '{print $4}' | grep "\b$SPARK_MASTER_PORT\b"`
if [ -n "$grep_port" ]; then
  echo "Port $SPARK_MASTER_PORT is in use"
  exit 1
fi

grep_port=`netstat -tlpn | awk '{print $4}' | grep "\b$SPARK_MASTER_WEBUI_PORT\b"`
if [ -n "$grep_port" ]; then
  echo "Port $SPARK_MASTER_WEBUI_PORT is in use"
  exit 1
fi

# Start master node
# TODO
"${SPARK_HOME}/sbin"/spark-daemon.sh start org.apache.spark.deploy.master.Master spark-master --ip $SPARK_MASTER_HOST --port $SPARK_MASTER_PORT --webui-port $SPARK_MASTER_WEBUI_PORT "$@"

# NOTE: This exact class name is matched downstream by SparkSubmit.
# Any changes need to be reflected there.
CLASS="org.apache.spark.deploy.worker.Worker"

# if [[ $# -lt 1 ]] || [[ "$@" = *--help ]] || [[ "$@" = *-h ]]; then
#  echo "Usage: ./sbin/start-slave.sh [options] <master>"
#  pattern="Usage:"
#  pattern+="\|Using Spark's default log4j profile:"
#  pattern+="\|Registered signal handlers for"

#  "${SPARK_HOME}"/bin/spark-class $CLASS --help 2>&1 | grep -v "$pattern" 1>&2
#  exit 1
#fi

# First argument should be the master; we need to store it aside because we may
# need to insert arguments between it and the other arguments
# TODO
MASTER="spark://$SPARK_MASTER_HOST:$SPARK_MASTER_PORT"

# Determine desired worker port
if [ "$SPARK_WORKER_WEBUI_PORT" = "" ]; then
  SPARK_WORKER_WEBUI_PORT=8081
fi

# Start up the appropriate number of workers on this machine.
# quick local function to start a worker
function start_instance {
  NUMACTL=$1
  WORKER_NUM=$2
  shift
  shift

  if [ "$SPARK_WORKER_PORT" = "" ]; then
    PORT_FLAG=
    PORT_NUM=
  else
    PORT_FLAG="--port"
    PORT_NUM=$(( $SPARK_WORKER_PORT + $WORKER_NUM - 1 ))
  fi
  WEBUI_PORT=$(( $SPARK_WORKER_WEBUI_PORT + $WORKER_NUM - 1 ))

  $NUMACTL "${SPARK_HOME}/sbin"/spark-daemon.sh start $CLASS $WORKER_NUM \
     --webui-port "$WEBUI_PORT" $PORT_FLAG $PORT_NUM $MASTER "$@"
}

# Check if `numactl` exits
if type "numactl" > /dev/null 2>&1; then
  # Join an input array by a given separator
  function join_by() {
    local IFS="$1"
    shift
    echo "$*"
  }

  # Compute memory size for each NUMA node
  IFS=$'\n'; _NUMA_HARDWARE_INFO=(`numactl --hardware`)
  _NUMA_NODE_NUM=`echo ${_NUMA_HARDWARE_INFO[0]} | sed -e "s/^available: \([0-9]*\) nodes .*$/\1/"`
  _TOTAL_MEM=`grep MemTotal /proc/meminfo | awk '{print $2}'`
  # Memory size of each NUMA node = (Total memory size - 1g) / Num of NUMA nodes
  _NUMA_MEM=$((((_TOTAL_MEM - 1048576) / 1048576) / $_NUMA_NODE_NUM))
  
  _WORKER_NAME_NO=1

  # Load NUMA configurations line-by-line and set `numactl` options
  for nnode in ${_NUMA_HARDWARE_INFO[@]}; do
    if [[ ${nnode} =~ ^node\ ([0-9]+)\ cpus:\ (.+)$ ]]; then
      _NUMA_NO=${BASH_REMATCH[1]}
      IFS=' ' _NUMA_CPUS=(${BASH_REMATCH[2]})
      _LENGTH=${#_NUMA_CPUS[@]}
      if ht_enabled; then _LENGTH=$((_LENGTH / 2)); fi
      # calculate worker num on this numa node, 12 ~ 23 core/worker
      _WORKER_NUM=$((_LENGTH / 12))
      if [[ $_WORKER_NUM -eq 0 ]]; then
        _WORKER_NUM=1
      fi
      _LENGTH=$((_LENGTH / _WORKER_NUM))

      for ((i = 0; i < $((_WORKER_NUM)); i ++)); do
        core_start=$(( i * _LENGTH ))
        _NUMACTL="numactl -m ${_NUMA_NO} -C $(join_by , ${_NUMA_CPUS[@]:${core_start}:${_LENGTH}})"
        echo ${_NUMACTL}

        # Launch a worker with numactl
        export SPARK_WORKER_CORES=${_LENGTH} # core num per worker
        export SPARK_WORKER_MEMORY="$((_NUMA_MEM / _WORKER_NUM))g"  #TODO
        start_instance "$_NUMACTL" "$_WORKER_NAME_NO" "$@"
        _WORKER_NAME_NO=$((_WORKER_NAME_NO + 1))
      done
    fi
  done
else
  echo "Please install numactl package"
  if [ "$SPARK_WORKER_INSTANCES" = "" ]; then
    start_instance "" 1 "$@"
  else
    for ((i=0; i<$SPARK_WORKER_INSTANCES; i++)); do
      start_instance "" $(( 1 + $i )) "$@"
    done
  fi
fi
