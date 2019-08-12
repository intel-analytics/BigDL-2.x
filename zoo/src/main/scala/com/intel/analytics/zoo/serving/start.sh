#!/bin/bash

# --------------config

echo $ModelFolder

/opt/work/redis-5.0.5/src/redis-server --port $REDIS_PORT > /opt/work/redis.log &
echo "redis server started, please check log in /opt/work/redis.log"

${SPARK_HOME}/bin/spark-submit --master local[*] --driver-memory 32g --conf "spark.driver.extraJavaOptions=-Dbigdl.engineType=mkldnn -Dbigdl.mklNumThreads=1"  --jars ./packages/spark-redis-2.4.0-SNAPSHOT-jar-with-dependencies.jar --class com.intel.analytics.zoo.serving.ZooServing ./packages/serving-0.1.0-SNAPSHOT-jar-with-dependencies.jar -f ${ModelFolder} -b 64


