#!/usr/bin/env bash

#   Stops all slaves on this worker machine
function ht_enabled {
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

_WORKER_NUM=$1
if [ $TOTAL_CORE_NUM -lt 24 ] && [ -z "${_WORKER_NUM}" ]; then
  # use local mode
  exit 1
fi

if [ -z "${SPARK_HOME}" ]; then
  export SPARK_HOME="$(cd "`dirname "$0"`"/..; pwd)"
fi

. "${SPARK_HOME}/sbin/spark-config.sh"

. "${SPARK_HOME}/bin/load-spark-env.sh"

# Stop master
"${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.master.Master spark-master

_WORKER_NAME_NO=1

# Check if `numactl` exits
if type "numactl" > /dev/null 2>&1; then
  # Load NUMA configurations line-by-line and catch NUMA node no.
  IFS=$'\n'; for nnode in `numactl --hardware`; do
    if [[ ${nnode} =~ ^node\ ([0-9]+)\ cpus:\ (.+)$ ]]; then
      IFS=' ' _NUMA_CPUS=(${BASH_REMATCH[2]})
      _LENGTH=${#_NUMA_CPUS[@]}
      if ht_enabled; then _LENGTH=$((_LENGTH / 2)); fi
      if [[ -z "${_WORKER_NUM}" ]]; then
        # calculate worker num on this numa node, 12 ~ 23 core/worker
        _WORKER_NUM=$((_LENGTH / 12))
      fi
      if [[ $_WORKER_NUM -eq 0 ]]; then
        _WORKER_NUM=1
      fi
      for ((i = 0; i < $((_WORKER_NUM)); i ++)); do
        "${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker "$_WORKER_NAME_NO"
        _WORKER_NAME_NO=$((_WORKER_NAME_NO + 1))
      done
    fi
  done
else
  echo "Please install numactl package"
#  if [ "$SPARK_WORKER_INSTANCES" = "" ]; then
#    "${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker 1
#  else
#    for ((i=0; i<$SPARK_WORKER_INSTANCES; i++)); do
#      "${SPARK_HOME}/sbin"/spark-daemon.sh stop org.apache.spark.deploy.worker.Worker $(( $i + 1 ))
#    done
#  fi
fi
