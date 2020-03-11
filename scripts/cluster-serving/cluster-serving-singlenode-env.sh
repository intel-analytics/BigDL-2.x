#!/bin/bash
#set -x
#set -e
 
SPARK_MASTER_PORT=7077
SPARK_MASTER_WEBUI_PORT=8080
REDIS_PORT=6379
WORKER_NUM=""
TOTAL_CORE_NUM=`nproc`
CPUSET=0-$((TOTAL_CORE_NUM - 1))
PREFIX=""

# Loop through arguments and process them
for arg in "$@"
do
  case $arg in
    -w=*|--worker-num=*)
    WORKER_NUM="${arg#*=}"
    shift
    ;;
    -c=*|--cpuset=*)
    CPUSET="${arg#*=}"
    shift
    ;;
    --container-prefix=*)
	  PREFIX="${arg#*=}"
	  shift
	  ;;
    -w|--worker-num)
    WORKER_NUM="$2"
    shift # Remove argument name from processing
    shift # shift # Remove argument value from processing
    ;;
    -c|--cpuset)
    CPUSET="$2"
    shift
    shift
    ;;
    --container-prefix)
    PREFIX="$2"
    shift
    shift
    ;;
    *)
    OTHER_ARGUMENTS+=("$1")
    shift # Remove generic argument from processing
    ;;
  esac
done

IFS='-'
CORE_ARR=($CPUSET)
CORE_START=${CORE_ARR[0]}
CORE_END=${CORE_ARR[1]}
unset IFS
if [ -z $CORE_START ]; then
	echo "Please check CPUSET"
	exit 1
fi

if [ $CORE_END -lt $CORE_START ]; then
	echo "Incorrect CPUSET range"
	exit 1
fi

if [ $CORE_END -ge $TOTAL_CORE_NUM ]; then
	echo "CORE number exceeds limit"
	CORE_END=$((TOTAL_CORE_NUM - 1))
	if [ $CORE_START -eq $TOTAL_CORE_NUM ]; then
		CORE_START=$((TOTAL_CORE_NUM - 1))
	fi	
fi
TOTAL_CORE_NUM=$((CORE_END - CORE_START + 1))

if [ -z $WORKER_NUM ]; then
	echo "Calculating worker num..."
	# 16 ~ 31 core/worker
	WORKER_NUM=$((TOTAL_CORE_NUM / 16))
	# total core num < 16
	if [ $WORKER_NUM -eq 0 ] && [ $CORE_START -ne 0 ]; then
		WORKER_NUM=1
	fi
fi

if [ $WORKER_NUM -gt $TOTAL_CORE_NUM ]; then
	echo "Worker number cannot exceed total core number"
	exit 1
elif [ $WORKER_NUM -lt 0 ]; then
	echo "Worker number cannot < 0"
	exit 1
fi

WORKER_CORE_NUM=$TOTAL_CORE_NUM
if [ $WORKER_NUM -ne 0 ]; then
  WORKER_CORE_NUM=$((TOTAL_CORE_NUM / WORKER_NUM))
fi

echo "worker_num: $WORKER_NUM"
echo "core_start: $CORE_START"
echo "core_end: $CORE_END"

echo "Check port..."
if [ $WORKER_NUM -ne 0 ]; then
  grep_port=`netstat -tlpn | grep "\b$SPARK_MASTER_PORT\b"`
  if [ -n "$grep_port" ]
  then
    echo "Port $SPARK_MASTER_PORT is in use"
    exit 1
  fi

  grep_port=`netstat -tlpn | grep "\b$SPARK_MASTER_WEBUI_PORT\b"`
  if [ -n "$grep_port" ]
  then
    echo "Port $SPARK_MASTER_WEBUI_PORT is in use"
    exit 1
  fi
fi

grep_port=`netstat -tlpn | grep "\b$REDIS_PORT\b"`
if [ -n "$grep_port" ]
then
    echo "Port $REDIS_PORT is in use. Please check redis port when running cluster-serving"
#    exit 1
fi

host_ip=`ifconfig | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1'|grep -v '172.17.0.1'`

if [ $WORKER_NUM -ne 0 ]; then
  echo "Start a master node..."
  docker run -itd --name "$PREFIX"cluster_master --net=host -e mode=master --add-host `hostname`:$host_ip intelanalytics/docker-spark:v1

  echo "Start worker nodes..."
  for i in $( seq 1 $WORKER_NUM ); do
    TEMP_CORE_END=$CORE_END
    if [ $i != $WORKER_NUM ]; then
      TEMP_CORE_END=$((CORE_START + WORKER_CORE_NUM - 1))
    fi
    echo "Start worker $i, cpu-set: $CORE_START - $TEMP_CORE_END"
    docker run -itd --name "$PREFIX"cluster_worker$i --net=host -e SPARK_MASTER=spark://`hostname`:$SPARK_MASTER_PORT -e mode=worker --add-host `hostname`:$host_ip --cpuset-cpus $CORE_START-$TEMP_CORE_END intelanalytics/docker-spark:v1
    CORE_START=$((TEMP_CORE_END + 1))
  done
fi

echo "Start cluster serving..."
docker run -itd --name "$PREFIX"cluster-serving --net=host -e OMP_NUM_THREADS=$WORKER_CORE_NUM -e KMP_AFFINITY=granularity=fine,compact,1,0 -e KMP_BLOCKTIME=20 intelanalytics/zoo-cluster-serving:0.7.0

if [ $WORKER_NUM -ne 0 ]; then
  echo "Master: spark://`hostname`:$SPARK_MASTER_PORT"
  echo "Data Src: $host_ip:Redis Port(Default 6379)"
else
  echo "Master: local[$TOTAL_CORE_NUM]"
  echo "Data Src: localhost:Redis Port(Default 6379)"
fi
