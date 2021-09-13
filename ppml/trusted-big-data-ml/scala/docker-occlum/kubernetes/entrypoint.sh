#!/bin/bash

# set -ex

# Check whether there is a passwd entry for the container UID
myuid=$(id -u)
mygid=$(id -g)
# turn off -e for getent because it will return error code in anonymous uid case
set +e
uidentry=$(getent passwd $myuid)
set -e

# If there is no passwd entry for the container UID, attempt to create one
if [ -z "$uidentry" ] ; then
    if [ -w /etc/passwd ] ; then
        echo "$myuid:x:$myuid:$mygid:anonymous uid:$SPARK_HOME:/bin/false" >> /etc/passwd
    else
        echo "Container ENTRYPOINT failed to add passwd entry for anonymous UID"
    fi
fi

SPARK_K8S_CMD="$1"
case "$SPARK_K8S_CMD" in
    driver | driver-py | driver-r | executor)
      shift 1
      ;;
    "")
      ;;
    *)
      echo "Non-spark-on-k8s command provided, proceeding in pass-through mode..."
      exec /sbin/tini -s -- "$@"
      ;;
esac

SPARK_CLASSPATH="$SPARK_CLASSPATH:/bin/jars/*"
env | grep SPARK_JAVA_OPT_ | sort -t_ -k4 -n | sed 's/[^=]*=\(.*\)/\1/g' > /tmp/java_opts.txt
readarray -t SPARK_EXECUTOR_JAVA_OPTS < /tmp/java_opts.txt

if [ -n "$SPARK_EXTRA_CLASSPATH" ]; then
  SPARK_CLASSPATH="$SPARK_CLASSPATH:$SPARK_EXTRA_CLASSPATH"
fi

if [ -n "$PYSPARK_FILES" ]; then
    PYTHONPATH="$PYTHONPATH:$PYSPARK_FILES"
fi

PYSPARK_ARGS=""
if [ -n "$PYSPARK_APP_ARGS" ]; then
    PYSPARK_ARGS="$PYSPARK_APP_ARGS"
fi

R_ARGS=""
if [ -n "$R_APP_ARGS" ]; then
    R_ARGS="$R_APP_ARGS"
fi

if [ "$PYSPARK_MAJOR_PYTHON_VERSION" == "2" ]; then
    pyv="$(python -V 2>&1)"
    export PYTHON_VERSION="${pyv:7}"
    export PYSPARK_PYTHON="python"
    export PYSPARK_DRIVER_PYTHON="python"
elif [ "$PYSPARK_MAJOR_PYTHON_VERSION" == "3" ]; then
    pyv3="$(python3 -V 2>&1)"
    export PYTHON_VERSION="${pyv3:7}"
    export PYSPARK_PYTHON="python3"
    export PYSPARK_DRIVER_PYTHON="python3"
fi

/opt/occlum/start_aesm.sh
case "$SPARK_K8S_CMD" in
  driver)
    CMD=(
      "$SPARK_HOME/bin/spark-submit"
      --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS"
      --deploy-mode client
      "$@"
    )
	exec /sbin/tini -s -- "${CMD[@]}"
    ;;
  driver-py)
    CMD=(
      "$SPARK_HOME/bin/spark-submit"
      --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS"
      --deploy-mode client
      "$@" $PYSPARK_PRIMARY $PYSPARK_ARGS
    )
	exec /sbin/tini -s -- "${CMD[@]}"
    ;;
    driver-r)
    CMD=(
      "$SPARK_HOME/bin/spark-submit"
      --conf "spark.driver.bindAddress=$SPARK_DRIVER_BIND_ADDRESS"
      --deploy-mode client
      "$@" $R_PRIMARY $R_ARGS
    )
	exec /sbin/tini -s -- "${CMD[@]}"
    ;;
  executor)
    /opt/build_spark_instance.sh
	cd /opt/occlum_spark/
	export OCCLUM_LOG_LEVEL=trace
  	occlum run /usr/lib/jvm/java-8-openjdk-amd64/bin/java \
  		"${SPARK_EXECUTOR_JAVA_OPTS[@]}" \
  		-Xms$SPARK_EXECUTOR_MEMORY \
  		-Xmx$SPARK_EXECUTOR_MEMORY \
  		-Dos.name=Linux \
  		-Dio.netty.availableProcessors=64 \
  		-cp "$SPARK_CLASSPATH" \
  		org.apache.spark.executor.CoarseGrainedExecutorBackend \
  		--driver-url $SPARK_DRIVER_URL \
  		--executor-id $SPARK_EXECUTOR_ID \
  		--cores $SPARK_EXECUTOR_CORES \
  		--app-id $SPARK_APPLICATION_ID \
  		--hostname $SPARK_EXECUTOR_POD_IP >/root/log/2.log 2>&1 
    ;;

  *)
    echo "Unknown command: $SPARK_K8S_CMD" 1>&2
    exit 1
esac
