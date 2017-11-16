#!/bin/bash
table_name=image
overwrite=true
export HADOOP_CONF_DIR=$HADOOP_HOME/etc/hadoop
export YARN_CONF_DIR=$HADOOP_HOME/etc/hadoop
spark-2.1.0-bin-hadoop2.7/bin/spark-submit \
--master yarn \
--deploy-mode client \
--executor-cores 28 \
--num-executors 28 \
--executor-memory 50g \
--driver-memory 50g \
--conf "spark.serializer=org.apache.spark.serializer.JavaSerializer" \
--class com.intel.analytics.zoo.pipeline.common.HBaseLoader \
pipeline-0.1-SNAPSHOT-jar-with-dependencies.jar \
-f test1 \
-t ${table_name} \
-b 100 \
--overwrite $overwrite

if [ "$overwrite" = "true" ]; then
  hive -hiveconf table=${table_name} -f data/tool/create_external_hive_table.sql
  echo create table ${table_name} in Hive done
  echo total records: ===============================
  hive -S -e "select count(uri) from ${table_name}"
fi;


