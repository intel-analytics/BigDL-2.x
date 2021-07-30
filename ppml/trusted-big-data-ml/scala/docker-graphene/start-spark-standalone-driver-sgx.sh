#!/bin/bash

spark_master=$SPARK_MASTER
driver_port=$SPARK_DRIVER_PORT
block_manager_port=$SPARK_BLOCK_MANAGER_PORT
driver_host=$SPARK_DRIVER_IP
driver_block_manager_port=$SPARK_DRIVER_BLOCK_MANAGER_PORT
export secure_password=`openssl rsautl -inkey /ppml/trusted-big-data-ml/work/password/key.txt -decrypt </ppml/trusted-big-data-ml/work/password/output.bin`

export SPARK_HOME=/ppml/trusted-big-data-ml/work/spark-2.4.6

bash ppml-spark-submit.sh \
    --master $spark_master \
    --conf spark.driver.port=$driver_port \
    --conf spark.scheduler.maxRegisteredResourcesWaitingTime=5000000 \
    --conf spark.worker.timeout=600 \
    --conf spark.executor.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
    --conf spark.driver.extraClassPath=/ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
    --conf spark.starvation.timeout=250000 \
    --conf spark.blockManager.port=$block_manager_port \
    --conf spark.driver.host=$driver_host \
    --conf spark.driver.blockManager.port=$driver_block_manager_port \
    --conf spark.network.timeout=1900s \
    --conf spark.executor.heartbeatInterval=1800s \
    --class com.intel.analytics.bigdl.models.lenet.Train \
    --executor-cores 4 \
    --total-executor-cores 4 \
    --executor-memory 12G \
    /ppml/trusted-big-data-ml/work/bigdl-jar-with-dependencies.jar \
    -f /ppml/trusted-big-data-ml/work/data \
    -b 64 -e 1 | tee ./spark-driver-sgx.log
